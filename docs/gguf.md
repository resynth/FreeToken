# GGUF support

FreeToken loads GGUF checkpoints natively (no conversion) for the architectures registered
in `models/gguf/config.py:GGUF_ARCH_TO_REGISTRY`:

| GGUF `general.architecture` | Registry spec |
|---|---|
| `gemma4` | `Gemma4GGUFForCausalLM` |
| `qwen4exp` (Qwen3.8-Flash-Next) | `Qwen4ExpGGUFForCausalLM` |

The shared plumbing lives in `models/gguf/`:
- `reader.py` — metadata, split-shard resolution, `GgufTensor` (torch shape + ggml type +
  packed `[rows, row_bytes]` bytes), `write_metadata_gguf` for `ft checkpoint`.
- `dequant.py` — pure-torch reference dequant + the blocked-shape metadata the packed paths use.
- `tokenizer.py` — builds a HF fast tokenizer from `tokenizer.ggml.*`.
- `config.py` — the `GgufConfigShim` the registry sees, and the arch -> registry map.

## Supported quant types

| ggml type | block / bytes | dense | experts | CPU MoE |
|---|---|---|---|---|
| F32 / F16 / BF16 | — | dense fallback | — | — |
| Q4_0 | 32 / 18 | MMVQ, MMQ, dequant | yes | yes |
| Q5_K | 256 / 176 | MMVQ, MMQ, dequant | yes | yes |
| Q6_K | 256 / 210 | MMVQ, MMQ, dequant | yes | no |
| Q8_0 | 32 / 34 | MMVQ, MMQ, dequant | yes | no |
| IQ4_NL | 32 / 18 | MMVQ, dequant | yes | yes |
| IQ4_XS | 256 / 136 | MMVQ, dequant | yes | yes |

Dense dispatch (`layers/gguf.py`): `_MMVQ` (small-batch GEMV) and `_DEQUANT`
(dequant-then-matmul) cover every type above; `_MMQ` (large-batch) only covers the types
the vendored `ggml_mul_mat_a8` actually implements. **Never add a type to a dispatch set
whose kernel switch lacks it** — the MMQ entry point has no default and silently returns
NaNs (IQ4_NL/IQ4_XS are deliberately MMVQ/dequant only).

## Routed experts

Experts stay as packed block bytes (`moe/gguf_experts.py`) and are dequantized inside the
borrowed `ggml_moe_a8_vec` kernel. Bank layout is one `gate_up [E, 2I, row_bytes(H)]` and one
`down [E, H, row_bytes(I)]` per layer, so the checkpoint may quantize gate/up and down
differently; that is a **composite format tag** `<gate_up>+<down>` (e.g. `iq4_xs+iq4_nl`).
`expert_quant` / `moe_weight_format` hold that tag, and the engine/cache/kernel dispatch read
the per-role types from it.

The expert banks are read once at startup into **pinned host RAM** and streamed to a
GPU LRU slot cache (`--moe-cache-size` / `--moe-cache-auto`). When the full packed expert
set does not fit pinned host RAM (on plain Linux the pin budget is 90% of `MemAvailable`;
`FREETOKEN_PIN_BUDGET_GB` overrides), repack it with `ft experts repack <gguf>` and boot
with `--expert-source mmap`: the store is mapped read-only, the OS page cache is the warm
tier and the SSD the cold tier, so RAM becomes the working set rather than the total.
Pass `--drop-ple` when the checkpoint carries its own `per_layer_token_embd.weight` (the
qwen4exp GGUFs do; ~28.8 GiB here) - the engine serves the fp8 table from `--ple-source`,
so archiving the GGUF copy only wastes disk.
An optional `ft experts stats` usage file pins the top experts per layer and prefetches
them one layer ahead. A calibration-free alternative is `--expert-warm-file`, a retained
expert plan (a JSON `layer -> [expert ids]`, e.g. the REAP top-K dump shipped as
`docs/qwen3.8-flash-next-top-384-experts-according-to-sh0wie.json`): the listed experts are
pinned per layer, budget-capped, and the rest fall back to the page cache. Both can be
combined - the warm file sets the pin plan while the usage file still drives prefetch.

Caution: the whole-layer prefill path copies the mmap views, not the pinned buffers, so a
large pin trades away the page cache prefill depends on. On a 62 GiB host a full REAP pin
(~44 GiB) makes prefill much slower; prefer `--expert-warm` with no pin file, or cap the pin
with `--expert-pin-fraction`. See the "F caveat" in
[mmap-tiering-performance.md](mmap-tiering-performance.md).
Without either, an oversized model still stops with a clear error instead of OOM-crashing the
host.

## Qwen3.8-Flash-Next (`qwen4exp`)

- Config from the `qwen4exp.*` KVs (`models/qwen4_exp/gguf.py:parse_gguf_config`); 48 layers,
  36 GDN + 12 QSA, 512 routed experts (releases vary).
- Tiny tensors (norms, routers, hyper-connection mixes, conv weights) become dense bf16;
  packed projections become `GGUFLinear`/`GGUFEmbedding` (`convert_qwen4_exp_to_gguf`). A
  group whose parts differ in quant type (e.g. QSA q=Q6_K with k/v=Q8_0) stays dense bf16.
- GDN is built split: packed `in_proj_qkvz` + dense `in_proj_ba` (the checkpoint ships b/a as
  F32) and `ssm_*` recurrences.
- **PLE**: the fp8 n-gram table is not taken from the GGUF. It is loaded from the original
  safetensors with `--ple-source <repo-or-dir>` (`--ple-backend disk|pinned`); the source's
  own `ngram_embedding.shard_<i>.weight` count defines the table. `ple.layer_multipliers` and
  the n-gram head sizes/offsets are derived and reproduce the GGUF's own values.
- How many IQ4_XS experts fit (52 GiB available):
  Base geometry (H=2560, I=640; gate/up IQ4_XS + down IQ4_NL) = 2.54 MiB/expert, 48 layers, so banks scale as:
  N experts   banks
   128        15.2 GiB
   256        30.5 GiB
   288        34.3 GiB
   320        38.1 GiB
   352        41.9 GiB
   384        45.7 GiB
   512        60.9 GiB   <- served from the mmap store, see below
  With ~56 GiB MemAvailable and ~4–6 GiB for the process/dense/KV, the pinned path's practical ceiling is ~352–384 experts; 320 is comfortable. expert_used_count (top-k=10) must stay ≤ N.

To serve the 512-expert IQ4_XS GGUF on a ~62 GiB host (banks exceed the pin budget):
```
ft experts repack <gguf> --out <store> --drop-ple
ft serve --model <gguf> --ple-source Saren/Qwen3.8-Flash-Next-ple-table-fp8 \
    --expert-source mmap --expert-store <store> --moe-cache-auto --expert-warm
```
See [mmap-expert-tiering.md](mmap-expert-tiering.md) "Running with the store": staged decode
is eager (CUDA graphs off), pass an explicit `--moe-cache-size` if decode OOMs. `--expert-warm`
pre-loads the store into the page cache, which is the warm tier the whole-layer prefill reads.

Pinning is optional and, on this host, a large pin hurts: `--expert-warm-file` pins the
REAP-retained 384/layer (45.7 GiB) and `--expert-usage-file` (from `ft experts stats`) pins a
usage-ranked set, but both evict the page cache prefill needs. Start without either, or cap the
pin with `--expert-pin-fraction`; to calibrate, run `ft experts stats --model <gguf> --calib
<text> --out <usage.json> --expert-source mmap --expert-store <store> --expert-warm
--ple-source <fp8-ple>`. Details and the measured diagnosis are in the "F caveat" of
[mmap-tiering-performance.md](mmap-tiering-performance.md).

## Verification

- `tests/models/test_gguf_dequant.py` — dequant bit-exact vs a literal `ggml-quants.c`
  transcription and gguf-py (synthetic + real checkpoint).
- `tests/models/test_gguf_reader_split.py` — split shards (synthetic + real).
- `tests/moe/test_gguf_experts.py`, `tests/layers/test_gguf_dispatch.py` — expert/service
  wiring and dispatch-set guards.
- `tests/moe/test_cpu_moe_gguf_quants.py` — CPU W4A8 vs GPU.
- `tests/models/test_gguf_qwen4exp_config.py` — qwen4exp config/plan/PLE-source (synthetic;
  real checkpoint behind `FREETOKEN_QWEN4EXP_GGUF`).

## Known limits / TODOs

- Offload pins the whole expert set by default; the disk-backed mmap tier
  (`ft experts repack` + `--expert-source mmap`, see `mmap-expert-tiering.md`) now serves
  sets that exceed the pin budget. Deferred follow-ups from the v1 review:
  - Staged decode now fills a double-buffered pinned ring with whole-expert buffered
    `preadv` rows (`ExpertSource.read_rows_into`, no per-page mmap faults) and reads
    `num_indices`/`src_indices` with one D2H sync, with no per-chunk stream drain
    (`offload_cache.py:_copy_missing_staged`). The per-row Python loop survives only as the
    fallback for a source without `read_rows_into`. Read-path and copy A/B numbers:
    `benchmarks/bench_expert_store.py` (see `mmap-tiering-performance.md` C/M).
  - An auto-selected mmap source disables decode CUDA graphs with only an info log
    (`engine._finish_mmap_source`); warn explicitly when `--expert-source auto` picks mmap.
  - `FREETOKEN_EXPERT_RING_ROWS` is read once (`offload_cache.staging_ring_rows()`), and
    `_resolve_expert_pin_budget` carves out both staging buffers, so the budget and the
    allocated ring can no longer drift. Still env-only with no CLI flag, and the default 8
    measured fastest in the bench; change it only with that bench's numbers.
  - The repack writer's GGUF tensor-name -> role mapping duplicates
    `moe/gguf_experts.py:load_gguf_expert_sources`; a new naming or fused form must be
    updated in both.
  - `moe/expert_source.py:PinnedExpertSource` and the `ExpertSource.resident`/`expert_bytes`
    members are defined but unused.
  - The cold-path O_DIRECT option and `mincore`-based cold-read counters from the plan are
    not implemented; ring fill is a buffered whole-expert page-cache read
    (`ExpertSource.read_rows_into`). Both are benchmark-gated in the plan, not committed
    behaviour; `benchmarks/bench_expert_store.py` now measures the read paths (`o_direct`
    among them) so a default can be picked with numbers. Startup page-cache warming is
    `--expert-warm` / `FREETOKEN_EXPERT_WARM=1`.
  - Online per-`(layer, expert)` counters are not wired to `--moe-collect-stats`: the server
    path only accumulates `lru_stats` (miss rate), while the histogram needs
    `collect_decode_freq` set programmatically and has no dump endpoint, so
    `--expert-usage-file` currently needs the offline `ft experts stats` pass. A
    calibration-free alternative is `--expert-warm-file`, but it only seeds residency (a
    static retained set, no ranking and no prefetch).
  - The mmap store is native-GGUF only (`q4_0`/`q5_K`/`iq4_nl`/`iq4_xs` and composite tags)
    and requires contiguous expert layers; nvfp4/mxfp4/fp8-block and HF/FTW checkpoints keep
    the pinned path, and leading-dense expert layouts are rejected by the repack.
  - `ft checkpoint` does not emit an expert store, and the engine checks the store's
    format/geometry but not its `fingerprint`/`source_path`, so a same-geometry store built
    from a different checkpoint is accepted silently.
  - Prefetch depth (`FREETOKEN_EXPERT_PREFETCH`) and ring rows
    (`FREETOKEN_EXPERT_RING_ROWS`) are env-only, with no CLI flags.
- `ft experts stats` now forwards `--expert-source` / `--expert-store` / `--expert-warm` and
  `--ple-source` / `--ple-backend` to the calibration model (a split GGUF's store is not at
  the default `<shard>.experts` path, and qwen4exp needs its external fp8 PLE table), and its
  `--max-new-tokens` is mapped to `SamplingParams.max_tokens` (it previously raised a
  TypeError). It still runs the full model eagerly, so it is slow until D/E; its CLI-level
  test coverage is now in `tests/moe/test_expert_store.py`.

  - Online adaptive re-pinning (calibration/production drift detection) stays out of scope
    for v1 per `mmap-expert-tiering.md`, gated on per-layer telemetry showing divergence.
- Expert format must be uniform across layers (one tag per model). Unsloth "dynamic" quants
  that vary the type per layer are rejected for now.
- FTW conversion is not wired for `qwen4exp`: a metadata-only GGUF has no tensor table, so the
  quant plan cannot be derived; it would need the plan persisted at convert time.
- CPU/hybrid MoE only supports single-type (non-composite) expert formats.
- `moe/bench_profile.py` / `benchbw.py` have no entries for the new formats, so
  `--moe-strategy auto` stays on GPU offload unless overridden.
- CPU Q5_K GEMV recomputes the per-32 activation sum inside every output-row dot
  (`cpu_moe_ext.cpp: gguf_asum32`), so it does ~`2I + H` redundant sums per token/route.
  Precomputing the sums in the activation-quantization pass would remove it; left undone
  because it changes the W4A8 dot signature and needs A/B numbers.
- Two qwen4_exp tests assert exact equality where only closeness holds, so they fail
  deterministically (measured; both are test-hygiene, not runtime bugs):
  - `test_qsa_backend.py::test_chunked_prefill_matches_one_shot` compares a split prefill to
    a one-shot with `torch.equal`; the dual-source compress reduces the pooled group in a
    different order, giving max abs diff ~1.5e-3 at output scale ~0.1 (~1.5%), inside the
    `rtol=2e-2` the same file uses against the fp32 oracle. Should be a closeness check.
  - `test_ple.py::test_track_snapshot_equals_a_prefill_stopped_at_the_boundary` compares the
    boundary snapshot to a truncated prefill with `torch.equal`, at ~1.3e-7 relative fp32
    noise. Should be `allclose`.
- The two `needs_weights` tests in `tests/models/test_gguf_qwen4exp_config.py` hardcode the
  full Qwen3.8-Flash-Next checkpoint (512 experts, PLE on layer 1, layer-0 `in_proj_qkvz`
  fused as `qweight`), so they fail when `FREETOKEN_QWEN4EXP_GGUF` points at a variant such as
  the 160-expert no-PLE coder finetune (its layer-0 qkv/z types differ, so `in_proj_qkvz`
  stays dense). Assert derived properties or gate the tests to the matching file.
