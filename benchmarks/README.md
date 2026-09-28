# benchmarks

Run from the repo root with `PYTHONPATH=python:.`, pinned to one GPU
(`CUDA_VISIBLE_DEVICES=0`). Each script's `--help` / docstring has the details.

**`bench_decode_moe.py`** — bs=1 decode tok/s of a served MoE model. Spawns `ft serve`
per backend and times token arrivals over streamed `/v1/chat/completions`, so numbers
include the full serving path. AIME-25 prompt, checkpoint-recommended sampling.

```bash
python benchmarks/bench_decode_moe.py --model /path/to/model --backend offload,cpu,hybrid
```

**`bench_load_weight_generic.py`** — expert-bank load time: serial vs parallel O_DIRECT
vs pre-repacked FTW, each mode in its own subprocess. Linux-only; stages the FTW under
`/var/tmp` (`--ftw-dir` overrides; roughly checkpoint-sized).

```bash
python benchmarks/bench_load_weight_generic.py --model /path/to/model
```

**`bench_offload_cache_copy.py`** — synthetic (no checkpoint): per-layer decode expert
copy cost (`ensure_experts` + `copy_missing`), swept over bank layout x cache slots x
batch size x miss rate.

```bash
python benchmarks/bench_offload_cache_copy.py
```

**`bench_expert_store.py`** — the mmap expert-store tier (no model load): the read paths
(`mmap_random`, `willneed`, whole-expert buffered `preadv`, O_DIRECT) cold/warm, and
`_copy_missing_staged` swept over ring rows x miss count with `--legacy` timing the pre-C
loop for an A/B, plus a per-token / per-1000-token-prefill projection. Synthesizes a tiny
store when `--store` is omitted.

```bash
python benchmarks/bench_expert_store.py --store /path/to/store --cold
python benchmarks/bench_expert_store.py --store /path/to/store --section staged --legacy
```

For host RAM vs PCIe bandwidth and the offload/hybrid backend pick, use `ft bench bw`
instead — it writes the JSON profile the engine reads.
