# MoE mmap-tiering: why the 512-expert model is unusably slow, and what to do

Target hardware this was reasoned about: i7-12800H (6P + 8E, AVX2 + AVX-VNNI),
RTX 3080 Ti Laptop 16 GiB, 62 GiB RAM, store on NVMe (`/media/b/Hyena`, ROTA=0).
Model: Qwen3.8-Flash-Next-Uncensored IQ4_XS, 48 layers x 512 experts, H=2560, I=640,
composite `iq4_xs+iq4_nl` experts (2.54 MiB/expert, 62.4 GiB of banks). Store present at
`/media/b/Hyena/qwen38-experts-store` (48 x (891 MB gate_up + 472 MB down)).

## TL;DR

The staged path is slow because, cold, each expert is read one 4 KiB page-fault at a time
(`MADV_RANDOM` disables readahead), there is no page-cache warm-up and no default pinning,
and the host-driven copy adds per-layer device syncs + a per-row Python loop. Measured on
this box against the real store:

| read path | throughput | per 1.66 MiB gate_up expert |
|---|---|---|
| O_DIRECT 4 KiB random (what `MADV_RANDOM` page faults approximate) | **53 MiB/s** | **~31 ms** |
| O_DIRECT large (256 KiB chunks) | **1253 MiB/s** | **1.33 ms** |
| O_DIRECT sequential whole file | 4194 MiB/s | - |
| mmap row copy, cache warm (`MADV_RANDOM`) | 9198 MiB/s | 0.18 ms |
| buffered pread into a buffer, cache warm | 11689 MiB/s | 0.14 ms |

So the cold read pattern alone is ~20x slower than it needs to be, and a single cold token
(480 routed experts x 2.54 MiB ~= 1.2 GiB) costs ~15 s at 53 MiB/s. A long prompt is worse;
that is the "completely unresponsive" symptom. The single cheapest fix (read whole experts
in large chunks / prefetch them) is worth ~20x on cold and is small.

| # | Fix | Impact | Effort | Risk |
|---|---|---|---|---|
| A | Read whole experts with large reads / `MADV_WILLNEED`, not per-page faults under `MADV_RANDOM` | High (20x cold) | S | Low |
| B | Warm the page cache at startup (whole store, ~46 GiB of 62 fits in RAM) | High (first token/prefill) | S | Low |
| C | Batched staged copy (double-buffered ring, one D2H sync, no per-chunk drain) (implemented) | Med-High | M | Med |
| D | CPU/hybrid MoE reading the mmap store directly (no PCIe, no ring) (implemented) | High (matches ik_llama.cpp) | M-L | Med |
| E | Support composite `iq4_xs+iq4_nl` in the CPU MoE extension | High (prereq for D) | M | Low-Med |
| F | Seed the warm/pin set from the REAP top-384 JSON (no calibration run needed) | Med-High | S-M | Med |
| G | Run `ft experts stats` + `--expert-usage-file` (+ `--expert-pin-budget`) | Med | S to run / S to wire | Low |
| H | Make staged decode CUDA-graph-capturable | Med-High | L | High |
| I | Tune `FREETOKEN_EXPERT_PREFETCH`, prefetch same-layer routed experts (implemented) | Med | S | Low |
| J | Usage-ordered repack (hot banks for the pin build) (implemented) | Med | M | Low |
| K | `cudaHostRegister` the warm mmap ranges instead of copying to anonymous pinned | Med | L | High |
| L | Remove the default `MADV_RANDOM` (or make it policy) | Med (moot with A) | S | Low |
| M | Add a microbenchmark harness so ring/size/usage changes get real A/B numbers (implemented) | Enables all | S-M | None |
| N | Surface the `FREETOKEN_EXPERT_*` knobs as flags and warn (not info) when `auto` picks mmap (partially: `--expert-prefetch` + the warning) | Low | S | None |

`Impact` is on this model's slowness, `Effort` is S (< half day), M (1-2 days), L (3+ days).

## Status

- **A - implemented.** `ExpertSource.read_rows_into` backs the staged ring fill and the
  pinned-subset build: pinned rows copy from their warm buffer, everything else is one
  buffered whole-expert `preadv` through the page cache, instead of a per-page fault on the
  mapping (`python/freetoken/moe/expert_source.py`). The staged whole-layer prefill path
  `MADV_WILLNEED`s the bank before its pageable H2D copy
  (`python/freetoken/moe/offload_cache.py`). L (drop the default `MADV_RANDOM`) was not done:
  the fixed paths no longer depend on it, so it stays a low-risk follow-up.
- **B - implemented.** `--expert-warm` (`FREETOKEN_EXPERT_WARM=1` the env fallback) runs
  `MmapExpertSource.warm_cache()` at load: bounded multi-threaded buffered sequential reads of
  every bank range, before the pinned subset is built. Wired through
  `moe/expert_banks.py`, `engine/config.py` and `server/args.py`.
- **C - implemented.** `_copy_missing_staged` (`moe/offload_cache.py`) now learns
  `num_indices` + `src_indices` with **one** D2H sync (pinned mirrors of both, copied
  together) instead of two, and stages a layer's misses through a **double-buffered** pinned
  ring with per-buffer CUDA events, so the previous chunk's pinned->CUDA copy is still in
  flight while the host fills the other buffer. The per-chunk full-stream `synchronize` is
  gone; the ring fill still calls `ExpertSource.read_rows_into` (one buffered whole-expert
  pread per miss). The old `FREETOKEN_EXPERT_RING_ROWS` read is now shared
  (`offload_cache.staging_ring_rows()` / `STAGING_RING_BUFFERS`) so the pin-budget carve-out
  and the allocated ring cannot drift.
- **M - implemented.** `benchmarks/bench_expert_store.py` microbenchmarks, without a model
  load:
  - the store read paths (`mmap_random`, `willneed`, `pread_whole`, `pread_prefetch`,
    `o_direct`) cold/warm, as MiB/s and ms/expert;
  - `_copy_missing_staged` swept over ring rows x miss count, with `--legacy` timing the
    pre-C loop and `--no-prefetch` timing the same-layer-prefetch-off loop (same
    invocation, interleaved), both also under `--cold`;
  - a per-token / per-1000-token-prefill projection from the measured whole-expert reads.
  It synthesizes a tiny store when `--store` is omitted, so it runs anywhere.
- **F - implemented.** `--expert-warm-file` (`FREETOKEN_EXPERT_WARM_FILE`) takes a
  retained-expert plan (JSON `layer -> [expert ids]`, e.g. the REAP top-384 dump) and pins
  exactly those experts per layer, budget-capped, with no calibration run
  (`moe/usage.py: load_warm_plan` / `select_warm_pins`, wired through
  `moe/expert_banks.py`, `engine/config.py` and `server/args.py`). It carries no frequency
  information, so it guarantees residency of the retained set rather than a ranking; when
  `--expert-usage-file` is also given, the warm file still sets the pin plan and the usage
  file drives prefetch. Budget leftover after the plan is not spent on unplanned experts.
  **Caveat (observed): a large pin starves the page cache the whole-layer prefill reads, which
  made this host much slower - see "F caveat" under Suggested fixes and size the pin
  deliberately.**
- **G - implemented (wiring + bug fix).** `ft experts stats` now forwards
  `--expert-source` / `--expert-store` / `--expert-warm` and `--ple-source` / `--ple-backend`
  into the calibration `LLM(...)` (`experts/__main__.py`), so a split GGUF can calibrate
  straight from its mmap store with its external PLE table instead of failing on the missing
  default `<shard>.experts` path. Adding the CLI test
  (`tests/moe/test_expert_store.py`) caught a latent bug: it passed
  `SamplingParams(max_new_tokens=...)`, which is now mapped to `max_tokens`. Still open: the
  offline pass runs eagerly (it is much faster with D's CPU decode over the store), and the
  live histogram is not wired to `--moe-collect-stats`.
- **E - implemented.** The C++ `CpuMoeExecutor` now takes a per-role weight format
  (`weight_format` + `down_weight_format`), keeps two dot kernels (`q4dot_gu`/`q4dot_dn`)
  and per-role row strides (`q4_gu_row_bytes`/`q4_dn_row_bytes`), and checks H/I
  divisibility against each role's own block size (a composite can impose a 256-block on
  gate_up only). The shared Q8_0-per-32 activation quantization is unchanged, which is what
  makes the pair cheap: every GGUF W4A8 dot consumes the same int8 grid, so one
  `quant_q8_0` pass feeds both roles. Python resolves a `'+'` tag into roles
  (`moe/cpu_executor.py: _role_formats` / `cpu_moe_format_supported`), so
  `_resolve_gguf_banks` passes both format ids, `CpuMoeExecutor` accepts the composite, and
  `engine._cpu_moe_executor_viable` no longer rejects it. `benchbw.py` synthesizes packed
  GGUF/composite banks for the per-dtype tuning bench and
  `bench_profile._QUANT_TO_BENCH_FORMAT` maps the tags, so `--moe-strategy auto` can pick
  hybrid for them. A stale prebuilt `_cpu_moe.so` is detected via the
  `supports_down_weight_format` marker: single formats keep the old ctor, a composite fails
  with a rebuild instruction, and auto degrades to offload. Covered by
  `tests/moe/test_cpu_moe_gguf_quants.py` (CPU-vs-GPU
  equivalence on `iq4_xs+iq4_nl`, `iq4_nl+iq4_xs` and the singles).
- **D - implemented.** `--moe-strategy cpu` (and hybrid) can now run the routed experts on the
  CPU straight over the mmap store, so decode stops moving the working set over PCIe and
  stops staging through the pinned ring. `_select_expert_source` no longer forces the pinned
  source for a non-offload strategy or `--moe-cpu-layers`: `auto` picks the store when the
  banks exceed the pin budget, and `--expert-source mmap` is honored for `cpu`/`hybrid`
  (`engine.py`). A pure-CPU mmap boot also skips the `--moe-strategy cpu` OS-lock split
  (there are no pins to size) and keeps CUDA graphs, because the CPU executor is
  graph-capturable while staged H2D is not (`_finish_mmap_source` now only disables graphs
  for `decode_target != "cpu"`). The cold-read floor is handled by a per-routed-expert
  `MADV_WILLNEED`: `CpuMoeExecutor` detects a pageable (`expert_source.resident == False`)
  source and enables `_cpu_moe.set_mmap_prefetch(True)`, so `submit()` madvises each routed
  expert's whole gate_up/down byte range before the pool reads it (the topk ids are already
  D2H'd by then), turning the 4 KiB fault stream into readahead. A stale `_cpu_moe.so`
  without the setter falls back to the old per-page-fault path. Covered by
  `tests/moe/test_cpu_moe_gguf_quants.py` (CPU-over-mmap vs GPU on the singles and
  composites) and `tests/moe/test_expert_store.py` (source selection + graph policy).
- **I - implemented (same-layer prefetch + depth flag; N's flag/warning folded in).**
  The staged copy already learns its misses with the one D2H read, so a pageable source
  is now `MADV_WILLNEED`ed for every miss *before* the first ring pread, letting the
  kernel's readahead overlap the per-expert reads instead of issuing them cold and
  serially (`_copy_missing_staged` -> `source.prefetch(layer, misses,
  include_pinned=False)`). `include_pinned=False` also applies to the next-layer usage
  plan: pinned rows are served from the pin buffer, so WILLNEEDing their pages would
  only refill the page cache with pages nothing reads (the "F caveat" mechanism in
  miniature) - only the whole-layer prefill copy keeps the default, because it reads
  the mmap views rather than the pins. The next-layer depth is now `--expert-prefetch N`
  (flag > `FREETOKEN_EXPERT_PREFETCH` env > default 4), and **0 disables all prefetch**
  (the plan and the same-layer miss WILLNEED together); the old env read had a
  `max(1, ...)` clamp, so 0 silently meant 1 and there was no way to turn prefetch off.
  Covered by `tests/moe/test_expert_store.py` (same-layer call + pinned skip + depth
  resolution + flag parsing). Measured on this host against the real store (ring 8,
  interleaved on/off via the bench's new `--no-prefetch` column, 11 reps):
  warm 8 misses +~0.01 ms, 32 +~0.27 ms, 128 +~0.29 ms, 512 **-0.85 ms** (readahead
  ahead of the preads over a whole layer); best-effort-cold 32 +~0.5 ms, 128 +~1.0 ms.
  Read-path A/B agrees: `pread_prefetch` vs `pread_whole` is within noise cold
  (1.133 vs 1.148 ms/expert at 32 experts) and warm (0.171 vs 0.216). Verdict: on this
  NVMe a whole-expert buffered pread already reaches drive bandwidth, so the same-layer
  WILLNEED is a wash-to-slightly-negative - kept on by default (decode-scale cost is
  noise, and it should pay on higher-latency/colder storage), with `--expert-prefetch 0`
  as the A/B kill switch for a served workload. Do not raise the *next-layer* depth
  without a served A/B either; the flag exists so that A/B is reproducible.
- **N - partially implemented (the parts I needs).** `--expert-prefetch` surfaces the
  depth; the mmap auto-pick that disables decode CUDA graphs now logs a **warning**
  (was info) naming the `--moe-strategy cpu` remedy, so a silently-eager default boot is
  visible (`engine._finish_mmap_source`). `FREETOKEN_EXPERT_RING_ROWS` stays env-only
  deliberately: the bench measured the default 8 fastest, so a flag would only invite
  unhelpful tuning.
- **J - implemented (hot banks + `--hot-only` patch); measured ~1.0x on this NVMe.**
  `ft experts repack` now takes `--hot-prefix K` with exactly one ranking input
  (`--usage-file` from `ft experts stats`, or `--warm-file`, e.g. the REAP top-K JSON)
  and writes a per-`(layer, role)` **hot bank** (`layer-000.gate_up.hot.bin`): the top-K
  experts' bytes contiguously, hottest first, duplicated on disk (K * expert_bytes per
  layer). The index records `hot_file` + `hot_ids` per bank (additive; old stores and
  old engines interop both ways). `--hot-only` patches hot banks into an existing store
  without rewriting the main banks, after verifying the store's fingerprint against
  this checkpoint - the first place the fingerprint (which the engine still does not
  check at load, gguf.md "Known limits") is enforced. `MmapExpertSource._build_pins`
  then fills the pinned warm subset from the hot bank when the plan densely covers it:
  plan == hot prefix in order is ONE sequential `preadv` straight into the pin buffer
  (no bounce tensor), a dense-but-permuted plan reads the covering prefix into a
  bounce tensor and scatters in RAM, plans sparser than half the hot bank fall back to
  the per-expert path, and out-of-hot experts always come from the main bank. After a
  hot read the source `POSIX_FADV_DONTNEED`s the hot file: the pin buffer owns the data,
  and the duplicated pages must not squat the page cache (the F caveat's mechanism).
  Why hot banks instead of the doc's original permuted-row layout: every consumer of
  `ExpertSource.all_layer_views` treats host row `e` as logical expert `e`
  (`_materialize_layer_kernel`'s `src_indices = off` / `slot_for_id[base + e] = e`, the
  LRU `lru_ensure` resolving `src_indices` as host rows, the CPU executor's pointer
  tables indexing by routed id, `copy_missing`'s positional whole-layer copy) - a
  permuted file cannot present a logical-order `[E, ...]` view without a copy, so the
  permutation lives in duplicated hot banks that only the pin build reads. Serving
  paths deliberately keep reading the main banks. Measured (`bench_expert_store.py
  --section pin`, all-real-byte store at the model's 2.64 MiB experts + the real
  store's 2.54 MiB): cold pin build hot 651 ms vs scattered 642 ms per 384-expert
  layer (**0.98x**); the real store's cold scattered baseline is 886 ms/layer
  (~1.1 GiB/s), warm 308 ms (~3.2 GiB/s). On these NVMe devices a scattered sequence
  of 1.3-2.6 MiB reads at queue depth 1 already runs at the sequential rate, so the
  layout win is confined to storage where sequential beats random (spinning disks,
  network stores) or small-request regimes - J is correct, tested infrastructure, not
  a speedup on this host. Do not build hot banks for this model expecting tok/s.

Local C A/B against the real store (`layer-000`, `gate_up`, cache_size 512, ring rows 8,
`--legacy`, page-warm, 21 reps). Only the chunked cases move; a single-chunk layer is
unchanged by construction:

| misses | chunks (ring 8) | new ms | legacy ms | speedup |
|---|---|---|---|---|
| 8 | 1 | 3.60 | 3.53 | 0.98x |
| 32 | 4 | 11.44 | 16.25 | 1.42x |
| 128 | 16 | 42.53 | 67.00 | 1.58x |

Ring scaling with 128 misses (new code): ring 8 **42.2 ms**, ring 32 43.7 ms, ring 128
55.4 ms. The double buffer makes the default ring of 8 the best of the three; do not raise
`FREETOKEN_EXPERT_RING_ROWS` without re-running the bench (a one-chunk ring loses the
fill/copy overlap and the smaller PCIe batches pipeline better here).

Local A/B against the real store (`layer-000.gate_up`, 32 random experts x 1.66 MiB, page
cache dropped between runs). This box cannot be made fully cold, so it understates the
table's 53 MiB/s cold baseline:

| path | cold | warm |
|---|---|---|
| mmap-view row copy (`MADV_RANDOM` faults) | 672 MiB/s | 14050 MiB/s |
| `read_rows_into` (buffered whole-expert pread) | 1443 MiB/s | 6073 MiB/s |

The cold read is ~2.1x faster here (the 20x is the truly-cold O_DIRECT gap from the table
above). The warm read is ~2x slower than the mmap memcpy but stays >6 GiB/s, well clear of the
cold floor, and with B the steady state is warm. C (the double-buffered ring, one D2H sync)
followed; `benchmarks/bench_expert_store.py` reproduces all of these numbers.

## Root causes (with references)

### 1. `MADV_RANDOM` + per-page faults on cold experts (dominant)
`MmapExpertSource._mmap` calls `mm.madvise(mmap.MADV_RANDOM)` (`moe/expert_source.py:157`),
which sets the kernel readahead window to zero. A cold expert read via
`warm_row -> self._view(layer, role)[expert] -> ring[c].copy_(row)`
(`moe/expert_source.py:188`, `moe/offload_cache.py:1208`) then faults 4 KiB at a time. On
this NVMe that is 53 MiB/s instead of >1 GiB/s. During prefill the engine copies whole layers
from the pageable mmap views (`moe/offload_cache.py:1229-1234`), so a long prompt touches
~62 GiB at 53 MiB/s.

### 2. No page-cache warming and no default pinning
`_build_pins` only runs when a `pin_plan` exists, and a plan only exists with
`--expert-usage-file` (`moe/expert_banks.py:352-364`). With no usage file the whole store
relies on the page cache, which is never pre-warmed: it fills 4 KiB at a time on demand.
`_prefetch_next_layer` also no-ops without a usage file (`moe/offload_cache.py:1152-1164`).
There is no "warm the store once at boot" step anywhere.

### 3. Host-synchronized, row-at-a-time staged copy
_(fixed: C, see Status; the analysis below is the pre-C baseline)_
`_copy_missing_staged` (`moe/offload_cache.py:1185-1222`) does, per MoE layer, per bank:
- `int(self.num_indices.item())` - a device sync (`:1193`),
- `self.src_indices[:n].cpu().tolist()` - a second device sync (`:1198`),
- a Python loop with one `warm_row` + pinned copy per expert (`:1205-1216`),
- one H2D + one `index_copy_` per ring chunk,
- `torch.cuda.current_stream().synchronize()` whenever another chunk follows (`:1222`).

With `FREETOKEN_EXPERT_RING_ROWS=8` (`:298`) and ~10 misses/layer this is ~1 extra sync/layer
on top of the two per-layer syncs, i.e. ~100-200 full-stream syncs per token, each one
draining the pipeline. This is exactly the "staged decode is eager" tradeoff, but the eager
sync count is far higher than needed.

### 4. CUDA graphs are disabled for the mmap source
`_finish_mmap_source` sets `cuda_graph_max_bs = 0` (`engine/engine.py:1494-1497`). For a
48-layer model that is a large constant Python/launch overhead on top of #3. It is disabled
because the staged path runs host code that a captured graph would not replay. _(D changed
this: the disable is now skipped for `decode_target == "cpu"`, where decode never stages -
the CPU executor is graph-capturable; only hybrid/GPU staged decode stays eager.)_

### 5. GPU offload moves the working set over PCIe every token
Even fully warm, the staged path copies every slot-cache miss host->device each token. On a
laptop 3080 Ti (often PCIe 4.0 x8) that is a few GB/s; a top-10-of-512 routing with a ~46
slots/layer cache can still miss 20-40% x 1.2 GiB/token. CPU MoE avoids that traffic
entirely, which is why `--moe-strategy cpu`/`hybrid` is attractive here (see D/E).

## Suggested fixes

### A. Read whole experts in large chunks (or `MADV_WILLNEED` them) - Impact High, Effort S
_(implemented: `ExpertSource.read_rows_into`, see Status)_
Do not let a cold expert be read by 4 KiB page faults. Two equivalent options:
- In the staged copy, before touching an expert, `MADV_WILLNEED` its whole
  `[expert_offset, +stride)` range; `MmapExpertSource.prefetch` already does this and is
  currently only called for next-layer usage-ranked experts (`moe/expert_source.py:200-231`).
  Always call it for the current layer's `src_ids` before the ring fill.
- Better for the cold path: read the expert with a buffered large `pread` (or `preadv` in
  256 KiB-1 MiB chunks) straight into the pinned ring slot, instead of indexing the mmap
  view. Still goes through the page cache (so the warm tier is populated), but issues
  sequential large I/O. Measured 1253 MiB/s vs 53 MiB/s.
A natural home is a new `ExpertSource.read_rows_into(layer, role, ids, dst_pinned)` that the
staged path uses; the mmap view stays for the pinned warm subset.
`docs/gguf.md:123-125` flags the O_DIRECT cold-path option as deliberately benchmark-gated:
A/B a buffered `pread` against O_DIRECT for both cold-first-read and repeat-read regimes (they
win in opposite regimes - O_DIRECT never populates the page cache) with the M harness before
picking a default.

### B. Warm the page cache at startup - Impact High, Effort S
_(implemented: `--expert-warm` / `FREETOKEN_EXPERT_WARM=1`, see Status)_
Add a boot step (flag, e.g. `--expert-warm` / `FREETOKEN_EXPERT_WARM=1`) that sequentially
reads the store once (`os.posix_fadvise(POSIX_FADV_WILLNEED)` per file, or a bounded
multi-threaded sequential read) before serving. On this box ~46 GiB of the 62 GiB fits in
page cache; at the measured 4.2 GiB/s that is ~11 s of boot for a mostly-warm server, versus
15+ minutes of on-demand 4 KiB faults. Page cache is reclaimable, so this does not risk OOM
the way pinning does.

### C. Batch the staged copy properly (the real M2 ring) - Impact Med-High, Effort M
_(implemented, see Status)_
This is the exact follow-up already recorded in `docs/gguf.md:109-113` ("staged decode copies
miss rows one at a time ... instead of a batched gather"; "each staged layer does two device
syncs ... read both from one pinned host buffer"), so it is the intended next step.
Replace the per-expert Python loop + per-chunk sync with:
1. one device->pinned read of `num_indices` and `src_indices` (ideally update a pinned host
   mirror asynchronously so there is at most one sync per layer, not two),
2. one vectorized gather `index_select`/`cat` of all miss rows for a bank into the pinned
   ring (or directly through A's `read_rows_into`),
3. one H2D of the whole batch and one `index_copy_`,
4. no per-chunk `torch.cuda.synchronize` (use a double-buffered ring with events, or size the
   ring so a layer's misses fit in one chunk).

Also worth doing while here: skip the two syncs entirely on the all-hit case by deferring the
`num_indices` read (currently `num_indices.item()` happens before the early-out).

Implemented as: (1) a single pinned D2H of both plus one `synchronize`; (2) `read_rows_into`
per chunk; (3)+(4) a double-buffered ring with per-buffer events and no stream drain. The
all-hit case still pays one D2H (the device owns the count and the staged path is eager), but
the two syncs are now one. Steps 1 and 4 are what the measured 1.4-1.6x on chunked layers
comes from; the per-row Python fallback is only the no-`read_rows_into` path now.

### D. CPU/hybrid MoE reading the mmap store directly - Impact High, Effort M-L
_(implemented, see Status)_
The fastest thing on this machine is probably to compute MoE experts on the CPU from the
mapped store, exactly as ik_llama.cpp does: no PCIe, no pinned host banks, no ring. This also
sidesteps the pin budget entirely (the store lives in reclaimable page cache). Scope: the
mmap store is native-GGUF only (`docs/gguf.md:130-132`), which covers this model.
Feasibility check against the code:
- `CpuMoeExecutor._make_table` uses `t.data_ptr()` on the per-layer `[E, ...]` tensors
  (`moe/cpu_executor.py:337-350`). The mmap source already exposes those as `uint8` views via
  `MmapExpertSource.all_layer_views` (`moe/expert_source.py:182-186`), and
  `_store_expert_banks` returns them as `sources` (`moe/expert_banks.py:375-381`). So the C++
  GEMV can read the mmap pages in place; a page fault simply blocks one worker thread.
- The blockers are policy/wiring, not data layout: `_select_expert_source` forces `pinned`
  whenever `moe_cpu_layers` is set or the strategy is not offload/hybrid
  (`engine/engine.py:1437-1438`), and `_cpu_moe_executor_viable` rejects the composite format
  (`engine/engine.py:1320-1322`).
- Implemented as: `_select_expert_source` allows mmap for cpu/hybrid (auto over budget, or
  explicit), `decode_target == "cpu"` keeps CUDA graphs and skips the OS-lock split, and the
  executor turns on C++ per-expert `MADV_WILLNEED`. The GPU slot cache is **not** skipped: the
  prefill path still streams whole layers into it (for `--moe-strategy cpu` it is the fixed
  2-layer prefill double buffer), so prefill still crosses PCIe - D removes the per-token
  decode traffic, not the prefill stream. Prefill over the mmap store copies the mapped pages,
  so keep `--expert-warm` (or a small pin) for it; the "F caveat" applies unchanged.
- Expect a CPU-bound result: at H=2560, I=640, top-10 x 48 layers, the GEMV is ~2.4 GMAC/token;
  a 6P+8E AVX-VNNI core does this in the 10-30 tok/s class, and it removes the PCIe term.
  This is the most likely way to match ik_llama.cpp, but it is more work than A-C.

### E. Composite expert support in the CPU extension - Impact High (with D), Effort M
_(implemented, see Status)_
`docs/gguf.md` "Known limits" previously recorded "CPU/hybrid MoE only supports single-type
(non-composite) expert formats"; this section is what removing that limit took.
`CpuMoeExecutor` took one
`weight_format`; a composite tag was rejected
(`moe/cpu_executor.py:178-183`). The change is smaller than it looks because gate_up and down
only differ in the **weight** format - activations are Q8_0 per-32 for every GGUF W4A8 dot
(`iq4_nl_dot_i8_*` / `iq4_xs_dot_i8_*` both index `asb[b]`; `cpu_moe_ext.cpp:1363-1458`).
Concretely:
- C++ (`cpu_moe_ext.cpp`): add a `down_weight_format` ctor arg; store `gu_fmt`/`dn_fmt`; two
  `q4dot_fn`s; `q4_gu_row_bytes = gguf_row_bytes(gu_fmt, H)`, `q4_dn_row_bytes =
  gguf_row_bytes(dn_fmt, I)`; per-role block-divisibility checks (the single-format check
  is now per role); `gemm1_dot` uses the gu dot (`:1852-1855`), `gemm2_dot` the dn dot
  (`:1875-1877`). Keep `use_q4a8` and `quant_q8_0` unchanged. _(done)_
- Python: in `_resolve_gguf_banks` (already computes `gate_up_type, down_type` from the tag)
  pass both format ids; relax the `CpuMoeExecutor` gate; relax `_cpu_moe_executor_viable`.
  _(done)_
- Also add the GGUF formats to `bench_profile._QUANT_TO_BENCH_FORMAT` (and `benchbw.py`) so
  `--moe-strategy auto` can ever choose hybrid for them. _(done)_
Roughly a day including a CPU-vs-GPU equivalence test (there is already
`tests/moe/test_cpu_moe_gguf_quants.py` to extend).

### F. Seed the warm/pin set from the REAP top-384 JSON - Impact Med-High, Effort S-M
_(implemented: `--expert-warm-file`, see Status)_
`docs/qwen3.8-flash-next-top-384-experts-according-to-sh0wie.json` is 48 layers x exactly 384
expert ids, **sorted ascending**, i.e. it is the *retained set* of a REAPed sibling, not a
ranked importance list. 384/512 x 62.4 GiB = 45.7 GiB, which fits the ~51 GiB pin budget
(`docs/gguf.md:71-81`). So it can be used as a prior: pin the retained 384 per layer,
leave the 128 dropped ones to the page cache. Implementation options, cheapest first:
- A tiny adapter that emits a `UsageData`-compatible JSON with counts 2 (retained) / 1
  (dropped), consumed by the existing `--expert-usage-file` path
  (`moe/expert_banks.py:352-364` + `moe/usage.py:select_pins`). Caveat: `select_pins` sizes by
  `expert_bytes` and a per-layer floor, so this pins exactly the retained set only if the
  budget allows; that is the intent.
- Or a new `--expert-warm-file` that directly sets the pin plan. **This is what was built**:
  the loader reads the REAP JSON as-is (`layer -> [ids]`), `select_warm_pins` pins the listed
  experts under the same per-layer floor as `select_pins` but never spends leftover budget on
  unplanned experts (there is no ranking to order them by). Out-of-range/duplicate ids are
  dropped. The adapter option was not built - the dedicated flag avoids conflating a
  residency plan with routing counts, and `--expert-usage-file` can still be passed alongside
  to drive prefetch.
Caveat: this is a different (REAPed) checkpoint; if REAP renumbered experts the mapping is
wrong, and it carries no frequency information, so it mostly guarantees residency of 75% of
experts rather than optimal ranking. It also does not fix #1/#3, so it is a complement, not a
substitute.

### F caveat. A large pin competes with the page cache prefill needs (observed)

`--expert-warm-file` is a residency prior, and on this host a large one is actively harmful
because the **whole-layer prefill path does not read the pinned subset**. `copy_missing`'s
`_pending_whole_layer` branch `MADV_WILLNEED`s the bank and then copies `per_layer[layer_id]`
- the **mmap view** - into the slot cache (`moe/offload_cache.py:1309-1318`). Only the staged
*decode* path consults the pins (`_copy_missing_staged` -> `read_rows_into`,
`offload_cache.py:1279-1284`). Every pinned byte is therefore a byte the page cache cannot use
for the very pages prefill reads.

Observed with the REAP top-384 plan on this 62 GiB host (store 62.4 GiB):

- `expert store: pinned warm subset 17858 experts (44.28 GiB)` (budget-capped; ~49 GiB was
  available because `_pin_budget_bytes` is 90% of `MemAvailable`).
- The store was largely page-cache resident before the run - the `Free memory before loading
  model: 15.38 GiB` line means MemAvailable was mostly reclaimable cache.
- Building the pinned subset evicted that cache, so every prefill re-read whole layers cold:
  `input throughput` 0.10-0.36 tok/s and clients timing out before a single token (the HTTP
  200 is the SSE response starting; tokens never arrive for minutes).
- Adding `--expert-warm` (warm cache first, then pin) was better than without it but still far
  worse than no pinning; removing `--expert-warm-file` restored the previous
  slow-but-responsive behaviour.

Guidance until the prefill path is made pin-aware or D/E removes the whole-layer PCIe stream:

- Prefer the page cache over pins: `--expert-source mmap --expert-warm`, with no
  `--expert-warm-file` and no `--expert-usage-file`.
- If pinning, size it so the cache keeps room: `--expert-pin-fraction 0.1` /
  `--expert-pin-budget 5`, not the ~44 GiB default.
- To use a G usage file for prefetch but not its pins, zero the budget:
  `--expert-usage-file <usage.json> --expert-pin-fraction 0` (`select_pins` returns empty, but
  `_finish_mmap_source` still attaches the ranked one-layer prefetch).
- Real fix: make the whole-layer prefill copy pinned rows from their pinned buffer and mmap
  only the non-pinned remainder (a small `copy_missing` change). D/E removes the decode
  dependence on the prefill stream but **not** prefill itself, which still copies the mapped
  pages for every layer - so this caveat (and `--expert-warm`) still applies with a pin.

### G. Calibration usage file (`ft experts stats`) - Impact Med, Effort S to run
_(implemented: store-flag forwarding + a latent `max_tokens` fix + a CLI test; see Status)_
G and F feed the same pin/prefetch policy but are not interchangeable: F is an unranked
retained set (residency only), while the usage file is frequency-ranked and is the only source
of the one-layer-ahead prefetch plan (`usage.prefetch_plan`). With both given, the warm file
sets the pins and the usage file drives prefetch. A usage file also lets you get prefetch
without pins via `--expert-pin-fraction 0` (see the "F caveat").
The documented M3 path works, with two caveats found in the code:
- `ft experts stats` does not forward `--expert-source`/`--expert-store`
  (`experts/__main__.py:50-57`). For this model the auto path would raise the pin-budget error
  unless `FREETOKEN_EXPERT_STORE` points at the store (then it auto-selects mmap). Small
  improvement: add `--expert-store`/`--expert-source` to the subcommand. **Done** (also
  `--expert-warm` and `--ple-source`/`--ple-backend`; this model needs the external PLE
  table); they are now passed to the calibration `LLM(...)`.
- It runs the model eagerly, so it is as slow as serving until A/B are fixed. Run it after A
  (and with a short `--max-new-tokens`) or rely on F. Still true: the offline pass is eager.
- The live counters cannot replace it: `--moe-collect-stats` accumulates only `lru_stats`
  (miss rate), the per-`(layer, expert)` histogram needs `collect_decode_freq` set
  programmatically and has no dump endpoint (`docs/gguf.md:126-129`), so the offline pass is
  the only route. It also has no test coverage (`docs/gguf.md:138-139`). **Coverage added**:
  a CLI test pins the forwarded flags, and it exposed a latent
  `SamplingParams(max_new_tokens=...)` TypeError (the flag is now mapped to `max_tokens`), so
  the command was unusable before this fix. The live-histogram dump remains open.
The output improves hit rate and overlap, not raw cold-read volume.

### H. Make staged decode graph-capturable - Impact Med-High, Effort L
The host-driven gather/sync is fundamentally uncapturable. Options: keep CUDA graphs for all
non-MoE work and only run MoE eagerly; or move the miss list to a graph-safe path (device-side
pinned-mirror read + captureable copy descriptors). Long-term and risky; revisit after D.

### I. Prefetch tuning / same-layer prefetch - Impact Med, Effort S
_(implemented, see Status: same-layer miss WILLNEED before the ring fill, pinned rows
skipped, next-layer depth via `--expert-prefetch` with 0 = all prefetch off; measured a
wash on this NVMe, so the flag doubles as the kill switch for a served A/B)_
`FREETOKEN_EXPERT_PREFETCH` defaults to 4 (`engine/engine.py:1501`) and only fires with a usage
file. Once A exists, prefetch the *current* layer's `src_ids` ranges (they are known before the
copy) and consider a larger next-layer depth. Cheap to A/B.

### J. Usage-ordered repack - Impact Med, Effort M
_(implemented, see Status, with one redesign forced by the code and one honest
measurement: physical row permutation would break the per-layer host-bank contract
``row == logical expert id`` - ``_materialize_layer_kernel`` writes ``src_indices = off``
and ``slot_for_id[base + e] = e``, the LRU ``lru_ensure`` resolves ``src_indices`` as
host rows, and the CPU executor's pointer tables index by routed id - so the store keeps
logical row order and adds per-``(layer, role)`` **hot banks** instead: duplicated
top-K bytes, contiguous, consumed only by the pin build. Measured ~1.0x on this NVMe -
at QD1 a scattered sequence of multi-MiB preads already runs at the sequential rate, so
the win is confined to storage where sequential beats random (spinning rust, network) or
to small-request regimes.)_
The store is already expert-contiguous per `(layer, role)`, so a whole-expert read is one
extent (`moe/expert_store.py:328-360`). If a usage ranking is available, repacking with hot
experts first (and optionally split into `hot.bin`/`cold.bin` per layer) makes prefetch and
readahead more effective and lets the warm subset be a contiguous prefix. Adds an optional
permutation to `ft experts repack`; index must record the permutation.

### K. `cudaHostRegister` the warm mmap ranges - Impact Med, Effort L
Instead of copying the warm subset into anonymous pinned buffers (which duplicates RAM and
still needs the ring copy), register the warm expert ranges of the mapping so they get a
device alias and the existing device-side gather (`fast_index_copy_multi_jit`) can read them.
Risks: page-locking is irreversible, expert strides are not page-aligned, and registering too
much defeats reclaim. Attractive long-term memory savings; not for v1.

### L. Drop the default `MADV_RANDOM` - Impact Med (moot with A), Effort S, Risk Low
_(not done: A's pread path and the prefill `MADV_WILLNEED` already avoid per-page faults)_
`MADV_RANDOM` is deliberate (`moe/expert_source.py:154-157`) to avoid pulling adjacent experts
no routing asked for. But with A doing explicit whole-expert reads/willneed, the global random
policy only hurts. Make it `MADV_NORMAL` by default, or apply `MADV_RANDOM` only to the mmap
view used for sparse pinned-subset copies.

### M. Add a microbenchmark harness - Effort S-M
_(implemented: `benchmarks/bench_expert_store.py`, see Status)_
Do not try to A/B via the server until A/B are fixed: it is too slow and noisy to isolate
anything. Add `benchmarks/` entries that, without loading the model,
1. cold-read N experts from a store via each path (mmap+MADV_RANDOM, WILLNEED, large pread) and
   report MiB/s and ms/expert,
2. report `_copy_missing_staged` cost for ring rows 8/32/512 and n=8/32/512 with the page cache
   both cold and warm,
3. project per-token and per-1000-token-prefill time from the observed miss rate.
Then the server-level A/B (ring rows, usage file, pin budget) only needs to confirm, not
discover. The existing per-layer telemetry (`staged_tier_stats`,
`decode_miss_stats_per_layer`; `moe/offload_cache.py:1047-1118`) covers hit-rate breakdown but
is host-side and only meaningful once #1/#3 are fixed.

## Answers to the specific questions

**Will a starting expert set help? Yes, with a caveat.**
The REAP JSON (F) is a good zero-cost warm/pin set (384 of 512 fits the pin budget), and it
avoids the slow calibration run. Residency does not help while cold experts cost 4 KiB faults
(A) or while the staged copy syncs per layer (C) - fix those first, then F. A, C and F are
now implemented: pass `--expert-warm-file docs/qwen3.8-flash-next-top-384-experts-according-to-sh0wie.json`
with `--expert-source mmap`. But a full-size pin (44 GiB here) evicts the page cache the
whole-layer prefill path reads and makes serving much slower; on this 62 GiB host prefer
`--expert-warm` with no pin, a small `--expert-pin-fraction`, or a pin-aware prefill (see the
"F caveat" above).

**`--moe-strategy cpu` / hybrid: worth unblocking?**
Yes, probably the largest steady-state win here, because it removes the PCIe term (root cause
5) and matches a RAM-resident working set. Work to unblock: E (composite in the CPU ext, ~1
day, low risk because activations are already shared) + wiring the mmap source to the CPU
executor (D) + bench-profile entries. **E, the bench-profile entries and D are done** (see
Status): `--moe-strategy cpu` / `hybrid` with `--expert-source mmap` (or `auto` over the pin
budget) runs the experts on the CPU reading the mapped store, and each routed expert is
`MADV_WILLNEED`ed before its GEMV so cold reads do not fault per page. The CPU path needs no
pinned banks, so it also solves the "banks don't fit the pin budget" problem rather than
working around it - the page cache is the warm tier. Prefill still streams whole layers over
PCIe, so `--expert-warm` (or a small pin) is still worth it.

**Is the "gather all miss rows, one H2D, one index_copy_" fix worth it?**
Yes - it is C, and it is the correct M2 implementation (the TODO in `docs/gguf.md:109-113`
says the same). It removes the per-row Python loop and the per-chunk sync, and cuts the
per-layer host reads from two to one. Measured on the real store at ring 8 (Status, C): 1.0x
at 8 misses (one chunk, so nothing to overlap), 1.4x at 32, 1.6x at 128. But note it is a
*latency/CPU* fix: it does not change the 53 MiB/s cold-read floor. Do A first so C's gains
are visible.

**`FREETOKEN_EXPERT_RING_ROWS=64` did nothing - how to measure properly?**
Because at ~10 misses/layer the default ring of 8 already needs only 1-2 chunks, the per-chunk
sync was not the dominant term; the dominant term was cold reads (#1) and the two per-layer
syncs (#3). Use M's `benchmarks/bench_expert_store.py --section staged --legacy`: it sweeps
ring rows x miss count and times the pre-C loop against the new one directly. On the real
store the new code at ring 8 was the fastest of 8/32/128 for 128 misses (Status, C), so the
default is already the right size; re-run the bench after changing it. To isolate cold reads
rather than the copy, use the read section with `--cold`, and a fixed short prompt through
the server capturing TTFT and `staged_tier_stats`.

**Does the usage file + pin budget route help?** Yes, it improves hit rate and enables
next-layer prefetch, but it improves *overlap*, not raw cold volume. Combine with A/B.

## Other observations

- The store's `index.json` `fingerprint` is not checked against the model at load
  (`docs/gguf.md:133-135`); if you rebuild/move the store, verify format/geometry manually.
- `FREETOKEN_EXPERT_RING_ROWS` is now read in one place
  (`offload_cache.staging_ring_rows()`) and `_resolve_expert_pin_budget` carves out
  `STAGING_RING_BUFFERS` (2) buffers, so the budget carve-out and the allocated ring cannot
  drift (`docs/gguf.md:116-117`). Changing the ring size still requires re-running M.
- The ring is allocated per bank as `2 * staging_rows * row_shape` (double-buffered); a large
  ring is real pinned RAM (`--expert-pin-budget` already carves it out) - size it deliberately,
  and note the bench showed the default 8 beating 32/128 for 128 misses.
- `MADV_WILLNEED` is async; issuing it immediately before the copy may not win much, but it
  does switch the kernel from 4 KiB faults to readahead. For a hard guarantee, do the large
  `pread` into the ring (A).

## Cross-check with `docs/gguf.md` known limits

`docs/gguf.md:104-166` already tracks most of this; the mapping, plus the few extras it
surfaces:

- `:109-113` staged decode is row-at-a-time and does two device syncs per layer -> fixed by C
  (one D2H sync + a double-buffered ring).
- `:114-115` an auto-selected mmap source disables decode CUDA graphs with only an info log.
  Raise that to a warning so a default boot that silently loses graphs is visible. D keeps
  graphs for `--moe-strategy cpu` (no staged decode); the info log still covers the staged
  (offload/hybrid) case.
- `:123-125` the cold-path O_DIRECT option and `mincore` counters are benchmark-gated and the
  ring fill is currently a page-cache copy -> the A/B point in A.
- `:126-129` the per-`(layer, expert)` histogram is not wired to `--moe-collect-stats` and has
  no dump endpoint, so `--expert-usage-file` needs the offline `ft experts stats` pass -> G,
  and a reason F's zero-calibration JSON is attractive.
- `:130-132` the mmap store is native-GGUF only and needs contiguous expert layers -> D/E
  scope.
- `:136-137` prefetch depth and ring rows are env-only, with no CLI flags -> N.
- `:138-139` `ft experts stats` still runs the full model eagerly; its store/PLE flag
  forwarding and a `max_tokens` fix landed, with CLI coverage -> G.
- `:149-152` CPU Q5_K GEMV redundantly recomputes the per-32 activation sum per output row
  (`gguf_asum32`). Not on this model's formats (gate/up IQ4_XS, down IQ4_NL), but relevant if
  E is later extended to Q5_K.

## Suggested order of work

1. A (large/whole-expert reads) + L (drop MADV_RANDOM default) - smallest change, ~20x cold.
   _A done; L left as follow-up (moot for the fixed paths)._
2. B (startup page-cache warm) - first token and prefill. _Done._
3. C (batched staged copy, one sync/layer) + I (current-layer prefetch) + N (flags/warning so
   the A/B runs are reproducible). _C, I and N's flag+warning are done. I measured as a
   wash-to-slightly-negative on this NVMe (see Status): same-layer prefetch is on by default
   with `--expert-prefetch 0` as the kill switch._
4. F (REAP-seeded warm set) or G (stats) - residency/prefetch. _F done
   (`--expert-warm-file`); G's store-flag forwarding + test done, the eager calibration run
   is still eager (until it runs on cpu/hybrid over the store, which D now enables)._
5. E + D (composite CPU ext, then CPU/hybrid over the mmap store) - the big architectural win.
   _E and D done (plus bench-profile entries): `--moe-strategy cpu`/`hybrid` reads the mmap
   store directly with per-routed-expert prefetch. Prefill still streams whole layers._
6. J, then H, then K as follow-ups (J compounds with the same usage file I now consumes;
   H only pays on staged GPU decode, which `--moe-strategy cpu` avoids entirely; K last,
   per its own risk notes). _J done: hot banks + `--hot-only`, but measured ~1.0x on
   this NVMe (see Status) - infrastructure for other storage classes, not a win here.
   Remaining: H (only for staged GPU decode), K._

M (the `benchmarks/bench_expert_store.py` harness) is the measurement gate for all of the
above; run it before and after any ring/source policy change.

## Appendix: how the evidence was gathered

Benchmarks are plain `os.preadv`/`mmap` reads against
`/media/b/Hyena/qwen38-experts-store` (no model load), `random.sample` expert ids:
- O_DIRECT 4 KiB random over `layer-000.gate_up.bin`: 53 MiB/s, 74 us/4 KiB.
- O_DIRECT 256 KiB chunks, whole 1.66 MiB experts: 1253 MiB/s, 1.33 ms/expert.
- O_DIRECT sequential whole file: 4194 MiB/s.
- Warm mmap single-row `copy_` under `MADV_RANDOM`: 9198 MiB/s (0.18 ms/expert).
- Warm buffered `pread` per expert: 11689 MiB/s (0.14 ms/expert).

Note the cold numbers are O_DIRECT (cache bypassing), which is the right lower bound for a
page-cache miss; `MADV_RANDOM` page faults were not measured directly because the store was
already mostly resident in page cache on this box during testing.

This is now reproducible without a model load via
`benchmarks/bench_expert_store.py --store /media/b/Hyena/qwen38-experts-store --section read
--cold`. Its `o_direct` column matches the 256 KiB row above (~1.3 GiB/s, ~1.22 ms/expert) and
its `pread_whole` column the buffered whole-expert row. Its `--cold` is best-effort
(`madvise(MADV_DONTNEED)` on the mapping + `fadvise(DONTNEED)`); without root the page cache is
rarely fully evicted, so the faulting paths read faster than the true 53 MiB/s cold lower bound
- trust the O_DIRECT column for the honest cold number. The `--section staged --legacy` mode
produces the C A/B numbers in Status directly against the cache.


## Update: implemented since these notes

See `docs/mmap-tiering-performance.md` for the full status and numbers.

- **A+B (done):** whole-expert buffered `preadv` reads (`ExpertSource.read_rows_into`) on the
  staged/pin paths, and `--expert-warm` / `FREETOKEN_EXPERT_WARM=1` page-cache warm-up.
- **The staged-copy fix above (C, done):** `_copy_missing_staged` now reads `num_indices` +
  `src_indices` with one D2H sync and stages through a double-buffered pinned ring with
  per-buffer events, with no per-chunk stream drain. Measured on the real store, ring 8:
  1.0x at 8 misses, 1.4x at 32, 1.6x at 128. The per-row Python loop is now only the
  fallback for a source without `read_rows_into`.
- **Measure properly (M, done):** `benchmarks/bench_expert_store.py` needs no model load.
  `--section read --cold` measures cold/warm read paths and `--section staged --legacy`
  times the old staged loop vs the new one, so ring/size/usage changes get A/B numbers
  without the server. On the real store the default ring of 8 measured fastest (do not
  raise `FREETOKEN_EXPERT_RING_ROWS` without re-running it).
- **Warm set without calibration (F, done):** `--expert-warm-file` /
  `FREETOKEN_EXPERT_WARM_FILE` accepts a retained-expert plan (JSON `layer -> [expert ids]`,
  e.g. the REAP top-384 dump) and pins those experts per layer, budget-capped, with no
  `ft experts stats` run. A usage file can still be passed alongside to drive prefetch.
- **Calibration run usable on the store (G, done):** `ft experts stats` forwards
  `--expert-source` / `--expert-store` / `--expert-warm` and `--ple-source` / `--ple-backend`
  to the calibration model and maps its `--max-new-tokens` to `SamplingParams.max_tokens` (a
  latent TypeError is fixed). Its output feeds `--expert-usage-file` for a ranked pin set plus
  one-layer prefetch.
- **Pinning vs page cache (operational):** the whole-layer prefill path reads the mmap views,
  not the pinned buffers, so a large pin (44 GiB here) evicts the cache prefill needs and makes
  serving much slower - see the "F caveat" under Suggested fixes. Prefer `--expert-warm` with
  no pin or a small `--expert-pin-fraction` until the prefill path is pin-aware; D/E removed
  the decode dependence on this stream but prefill still copies the mapped pages.
- **Composite CPU experts (E, done):** the CPU MoE extension takes a per-role weight format
  and keeps a dot kernel + row stride per role, so `iq4_xs+iq4_nl` (and any pair of
  native-GGUF W4A8 tags) computes on the CPU over the same packed banks, sharing the Q8_0/32
  activation quantization. `benchbw`/`bench_profile` carry the tags, so `--moe-strategy auto`
  can pick hybrid for them; `tests/moe/test_cpu_moe_gguf_quants.py` adds CPU-vs-GPU parity for
  the composite.
- **CPU/hybrid over the mmap store (D, done):** `--moe-strategy cpu` (and `hybrid`) can read
  the mapped banks directly. `auto` picks the store when the banks exceed the pin budget and
  `--expert-source mmap` is honored for a non-offload strategy; a pure-CPU mmap boot skips the
  OS-lock split and keeps CUDA graphs (only staged GPU decode must run eagerly). The cold-read
  floor is handled by `_cpu_moe.set_mmap_prefetch`, which `MADV_WILLNEED`s each routed
  expert's gate_up/down range before the pool computes it. Prefill still streams whole layers
  into the GPU slot cache, so keep `--expert-warm` (or a small pin); H (graph-capturable staged
  decode) and K (`cudaHostRegister`) remain as follow-ups, and L stays
  moot (A's `pread`/`WILLNEED` paths do not depend on the `MADV_RANDOM` default, and
  `MADV_WILLNEED` ignores the VMA's random policy, so I composes with it).
- **Same-layer prefetch + the depth flag (I, done; N's flag/warning folded in):** the staged
  copy `MADV_WILLNEED`s its misses before the ring fill (pinned rows skipped), the usage-ranked
  next-layer depth is `--expert-prefetch N` (flag > `FREETOKEN_EXPERT_PREFETCH` > 4, and 0
  disables all prefetch), and the mmap graph-disable is now a warning, not an info. Measured a
  wash on this NVMe (see Status); the flag doubles as the A/B kill switch.
- **Hot banks for the pin build (J, done; measured ~1.0x here):** `ft experts repack
  --hot-prefix K --usage-file|--warm-file [--hot-only]` writes contiguous duplicated top-K
  banks per (layer, role); the index records the ranking, `--hot-only` patches an existing
  store after a fingerprint check, and `_build_pins` fills the warm subset with one
  sequential pread when the plan densely covers the hot bank (dropping the duplicated
  pages afterwards). The row-permutation design was dropped because the per-layer
  host-bank contract (`row == expert id`) is load-bearing for the whole-layer prefill
  copy, the LRU remap and the CPU executor; see Status for the measured numbers and the
  honest verdict - keep it off this model's build recipes.

