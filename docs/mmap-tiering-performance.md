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
| C | Batched staged copy (M2 proper): one gather + one H2D + one scatter, no per-chunk sync | Med-High | M | Med |
| D | CPU/hybrid MoE reading the mmap store directly (no PCIe, no ring) | High (matches ik_llama.cpp) | M-L | Med |
| E | Support composite `iq4_xs+iq4_nl` in the CPU MoE extension | High (prereq for D) | M | Low-Med |
| F | Seed the warm/pin set from the REAP top-384 JSON (no calibration run needed) | Med-High | S-M | Med |
| G | Run `ft experts stats` + `--expert-usage-file` (+ `--expert-pin-budget`) | Med | S to run / S to wire | Low |
| H | Make staged decode CUDA-graph-capturable | Med-High | L | High |
| I | Tune `FREETOKEN_EXPERT_PREFETCH`, prefetch same-layer routed experts | Med | S | Low |
| J | Usage-ordered repack (hot experts contiguous per layer) | Med | M | Low |
| K | `cudaHostRegister` the warm mmap ranges instead of copying to anonymous pinned | Med | L | High |
| L | Remove the default `MADV_RANDOM` (or make it policy) | Med (moot with A) | S | Low |
| M | Add a microbenchmark harness so ring/size/usage changes get real A/B numbers | Enables all | S-M | None |
| N | Surface the `FREETOKEN_EXPERT_*` knobs as flags and warn (not info) when `auto` picks mmap | Low | S | None |

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

Local A/B against the real store (`layer-000.gate_up`, 32 random experts x 1.66 MiB, page
cache dropped between runs). This box cannot be made fully cold, so it understates the
table's 53 MiB/s cold baseline:

| path | cold | warm |
|---|---|---|
| mmap-view row copy (`MADV_RANDOM` faults) | 672 MiB/s | 14050 MiB/s |
| `read_rows_into` (buffered whole-expert pread) | 1443 MiB/s | 6073 MiB/s |

The cold read is ~2.1x faster here (the 20x is the truly-cold O_DIRECT gap from the table
above). The warm read is ~2x slower than the mmap memcpy but stays >6 GiB/s, well clear of the
cold floor, and with B the steady state is warm. C (one batched gather + one H2D/sync per
layer) is the next step and is unchanged by this work.

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
because the staged path runs host code that a captured graph would not replay.

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

### D. CPU/hybrid MoE reading the mmap store directly - Impact High, Effort M-L
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
- Needed: allow the mmap source with `decode_target in ("cpu", "hybrid")`; set
  `cpu_layer_ids` to all/selected layers; attach the executor; skip the GPU slot cache
  allocation when nothing uses it. Add per-routed-expert `MADV_WILLNEED` prefetch (the
  topk_ids are already D2H'd to pinned buffers in `decode_submit`) so cold CPU reads are
  large, not 4 KiB faults (otherwise A is still required).
- Expect a CPU-bound result: at H=2560, I=640, top-10 x 48 layers, the GEMV is ~2.4 GMAC/token;
  a 6P+8E AVX-VNNI core does this in the 10-30 tok/s class, and it removes the PCIe term.
  This is the most likely way to match ik_llama.cpp, but it is more work than A-C.

### E. Composite expert support in the CPU extension - Impact High (with D), Effort M
`docs/gguf.md:146` records "CPU/hybrid MoE only supports single-type (non-composite) expert
formats"; this section is what removing that limit takes. `CpuMoeExecutor` takes one
`weight_format`; a composite tag is rejected
(`moe/cpu_executor.py:178-183`). The change is smaller than it looks because gate_up and down
only differ in the **weight** format - activations are Q8_0 per-32 for every GGUF W4A8 dot
(`iq4_nl_dot_i8_*` / `iq4_xs_dot_i8_*` both index `asb[b]`; `cpu_moe_ext.cpp:1363-1458`).
Concretely:
- C++ (`cpu_moe_ext.cpp`): add a `down_weight_format` ctor arg; store `gu_fmt`/`dn_fmt`; two
  `q4dot_fn`s; `q4_gu_row_bytes = gguf_row_bytes(gu_fmt, H)`, `q4_dn_row_bytes =
  gguf_row_bytes(dn_fmt, I)`; per-role block-divisibility checks (currently `:1733-1740` is
  single-format); `gemm1_dot` uses the gu dot (`:1852-1855`), `gemm2_dot` the dn dot
  (`:1875-1877`). Keep `use_q4a8` and `quant_q8_0` unchanged.
- Python: in `_resolve_gguf_banks` (already computes `gate_up_type, down_type` from the tag)
  pass both format ids; relax the `CpuMoeExecutor` gate; relax `_cpu_moe_executor_viable`.
- Also add the GGUF formats to `bench_profile._QUANT_TO_BENCH_FORMAT` (and `benchbw.py`) so
  `--moe-strategy auto` can ever choose hybrid for them (`docs/gguf.md:147-148`).
Roughly a day including a CPU-vs-GPU equivalence test (there is already
`tests/moe/test_cpu_moe_gguf_quants.py` to extend).

### F. Seed the warm/pin set from the REAP top-384 JSON - Impact Med-High, Effort S-M
`docs/qwen3.8-flash-next-top-384-experts-according-to-sh0wie.json` is 48 layers x exactly 384
expert ids, **sorted ascending**, i.e. it is the *retained set* of a REAPed sibling, not a
ranked importance list. 384/512 x 62.4 GiB = 45.7 GiB, which fits the ~51 GiB pin budget
(`docs/gguf.md:71-81`). So it can be used as a prior today: pin the retained 384 per layer,
leave the 128 dropped ones to the page cache. Implementation options, cheapest first:
- A tiny adapter that emits a `UsageData`-compatible JSON with counts 2 (retained) / 1
  (dropped), consumed by the existing `--expert-usage-file` path
  (`moe/expert_banks.py:352-364` + `moe/usage.py:select_pins`). Caveat: `select_pins` sizes by
  `expert_bytes` and a per-layer floor, so this pins exactly the retained set only if the
  budget allows; that is the intent.
- Or a new `--expert-warm-file` that directly sets the pin plan.
Caveat: this is a different (REAPed) checkpoint; if REAP renumbered experts the mapping is
wrong, and it carries no frequency information, so it mostly guarantees residency of 75% of
experts rather than optimal ranking. It also does not fix #1/#3, so it is a complement, not a
substitute.

### G. Calibration usage file (`ft experts stats`) - Impact Med, Effort S to run
The documented M3 path works, with two caveats found in the code:
- `ft experts stats` does not forward `--expert-source`/`--expert-store`
  (`experts/__main__.py:50-57`). For this model the auto path would raise the pin-budget error
  unless `FREETOKEN_EXPERT_STORE` points at the store (then it auto-selects mmap). Small
  improvement: add `--expert-store`/`--expert-source` to the subcommand.
- It runs the model eagerly, so it is as slow as serving until A/B are fixed. Run it after A
  (and with a short `--max-new-tokens`) or rely on F.
- The live counters cannot replace it: `--moe-collect-stats` accumulates only `lru_stats`
  (miss rate), the per-`(layer, expert)` histogram needs `collect_decode_freq` set
  programmatically and has no dump endpoint (`docs/gguf.md:126-129`), so the offline pass is
  the only route. It also has no test coverage (`docs/gguf.md:138-139`).
The output improves hit rate and overlap, not raw cold-read volume.

### H. Make staged decode graph-capturable - Impact Med-High, Effort L
The host-driven gather/sync is fundamentally uncapturable. Options: keep CUDA graphs for all
non-MoE work and only run MoE eagerly; or move the miss list to a graph-safe path (device-side
pinned-mirror read + captureable copy descriptors). Long-term and risky; revisit after D.

### I. Prefetch tuning / same-layer prefetch - Impact Med, Effort S
`FREETOKEN_EXPERT_PREFETCH` defaults to 4 (`engine/engine.py:1501`) and only fires with a usage
file. Once A exists, prefetch the *current* layer's `src_ids` ranges (they are known before the
copy) and consider a larger next-layer depth. Cheap to A/B.

### J. Usage-ordered repack - Impact Med, Effort M
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

**Will a starting expert set help? Yes, but not as the first move.**
The REAP JSON (F) is a good zero-cost warm/pin set (384 of 512 fits the pin budget), and it
avoids the slow calibration run. But residency does not help while cold experts cost 4 KiB
faults (A) and while the staged copy syncs per layer (C); fix those first, then F.

**`--moe-strategy cpu` / hybrid: worth unblocking?**
Yes, probably the largest steady-state win here, because it removes the PCIe term (root cause
5) and matches a RAM-resident working set. Work to unblock: E (composite in the CPU ext, ~1
day, low risk because activations are already shared) + wiring the mmap source to the CPU
executor (D) + bench-profile entries. It does not require the banks to be pinned, so it also
solves the "banks don't fit the pin budget" problem rather than working around it. It will not
be fast if cold reads still fault per page, so A remains a prerequisite.

**Is the "gather all miss rows, one H2D, one index_copy_" fix worth it?**
Yes - it is C, and it is the correct M2 implementation (the TODO in `docs/gguf.md:109-113`
says the same). It removes the per-row Python loop and the per-chunk sync, and cuts the
per-layer host reads from two to one. But note it is a *latency/CPU* fix: it does not change
the 53 MiB/s cold-read floor. Do A first so C's gains are visible.

**`FREETOKEN_EXPERT_RING_ROWS=64` did nothing - how to measure properly?**
Because at ~10 misses/layer the default ring of 8 already needs only 1-2 chunks, the per-chunk
sync was not the dominant term; the dominant term was cold reads (#1) and the two per-layer
syncs (#3). Measure with M: a standalone `_copy_missing_staged` bench with cold/warm cache and
ring 8/32/512, plus a fixed short prompt through the server capturing TTFT and
`staged_tier_stats`. If the model is too slow to reach steady state, use `--max-new-tokens 1`
and time TTFT only; that isolates cold-read cost.

**Does the usage file + pin budget route help?** Yes, it improves hit rate and enables
next-layer prefetch, but it improves *overlap*, not raw cold volume. Combine with A/B.

## Other observations

- The store's `index.json` `fingerprint` is not checked against the model at load
  (`docs/gguf.md:133-135`); if you rebuild/move the store, verify format/geometry manually.
- `_resolve_expert_pin_budget` and `OffloadMoeCache.__post_init__` read
  `FREETOKEN_EXPERT_RING_ROWS` independently (both default 8); if you change the ring size,
  the budget carve-out can drift (`docs/gguf.md:116-117`).
- The ring is allocated per bank as `staging_rows * row_shape`; a large ring is real pinned
  RAM (`--expert-pin-budget` already carves it out) - size it deliberately.
- `MADV_WILLNEED` is async; issuing it immediately before the copy may not win much, but it
  does switch the kernel from 4 KiB faults to readahead. For a hard guarantee, do the large
  `pread` into the ring (A).

## Cross-check with `docs/gguf.md` known limits

`docs/gguf.md:104-166` already tracks most of this; the mapping, plus the few extras it
surfaces:

- `:109-113` staged decode is row-at-a-time and does two device syncs per layer -> fix C.
- `:114-115` an auto-selected mmap source disables decode CUDA graphs with only an info log.
  Raise that to a warning so a default boot that silently loses graphs is visible.
- `:123-125` the cold-path O_DIRECT option and `mincore` counters are benchmark-gated and the
  ring fill is currently a page-cache copy -> the A/B point in A.
- `:126-129` the per-`(layer, expert)` histogram is not wired to `--moe-collect-stats` and has
  no dump endpoint, so `--expert-usage-file` needs the offline `ft experts stats` pass -> G,
  and a reason F's zero-calibration JSON is attractive.
- `:130-132` the mmap store is native-GGUF only and needs contiguous expert layers -> D/E
  scope.
- `:136-137` prefetch depth and ring rows are env-only, with no CLI flags -> N.
- `:138-139` `ft experts stats` runs the full model eagerly and has no test coverage -> G.
- `:149-152` CPU Q5_K GEMV redundantly recomputes the per-32 activation sum per output row
  (`gguf_asum32`). Not on this model's formats (gate/up IQ4_XS, down IQ4_NL), but relevant if
  E is later extended to Q5_K.

## Suggested order of work

1. A (large/whole-expert reads) + L (drop MADV_RANDOM default) - smallest change, ~20x cold.
   _A done; L left as follow-up (moot for the fixed paths)._
2. B (startup page-cache warm) - first token and prefill. _Done._
3. C (batched staged copy, one sync/layer) + I (current-layer prefetch) + N (flags/warning so
   the A/B runs are reproducible).
4. F (REAP-seeded warm set) or G (stats) - residency/prefetch.
5. E + D (composite CPU ext, then CPU/hybrid over the mmap store) - the big architectural win.
6. H, J, K as follow-ups.

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
