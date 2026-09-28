cd ~/ai/FreeToken && source .venv/bin/activate

FREETOKEN_ALLOW_CUDA_MISMATCH=1 .venv/bin/ft serve \
  --model /media/b/Hyena/gguf/mradermacher/qwen3.8-flash-coder-85gb-bf16-i1-GGUF/qwen3.8-flash-coder-85gb-bf16.i1-IQ4_XS.gguf \
  --dtype bfloat16 --moe-strategy offload --text-model-only \
  --max-running-requests 1 --max-seq-len-override 4096 --cuda-graph-max-bs 1 \
  --host 127.0.0.1 --port 8000


source .venv/bin/activate
ft serve --model /media/b/Hyena/gguf/orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF/Qwen3.8-Flash-Next-Uncensored-IQ4_XS-00001-of-00003.gguf \
  --ple-source /home/b/.cache/huggingface/hub/models--Saren--Qwen3.8-Flash-Next-ple-table-fp8/snapshots/50511b0a41aa1d34b8beb7e5d4bb06a0b650dc14 \
  --expert-source mmap --expert-store /media/b/Hyena/qwen38-experts-store --moe-cache-auto


.venv/bin/ft serve \
  --model /media/b/Hyena/gguf/orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF/Qwen3.8-Flash-Next-Uncensored-IQ4_XS-00001-of-00003.gguf \
  --ple-source /home/b/.cache/huggingface/hub/models--Saren--Qwen3.8-Flash-Next-ple-table-fp8/snapshots/50511b0a41aa1d34b8beb7e5d4bb06a0b650dc14 \
  --expert-source mmap --expert-store /media/b/Hyena/qwen38-experts-store --moe-cache-auto \
  --text-model-only --max-running-requests 1 --max-seq-len-override 4096 --cuda-graph-max-bs 1 \
  --host 127.0.0.1 --port 8000


cd ~/ai/FreeToken && source .venv/bin/activate
.venv/bin/ft shell --server http://127.0.0.1:8000



It is unusably slow!  ik_llama.cpp is way faster with the same model on this same computer.  When testing a smaller model that doesn't require this mmap-expert-tiering, /media/b/Hyena/gguf/mradermacher/qwen3.8-flash-coder-85gb-bf16-i1-GGUF/qwen3.8-flash-coder-85gb-bf16.i1-IQ4_XS.gguf, it is faster than in ik_llama.cpp.  I think our mmap-expert-tiering is causing the extreme slowness

Things to try now (no code change)
- FREETOKEN_EXPERT_RING_ROWS=64 (or 512+) — directly removes most of the per-8-row syncs. Cheapest experiment but seems to not work.  Measure it properly — ring rows 8/32/512, and with/without ft experts stats + --expert-usage-file — so we have real A/B numbers before changing any code.
- ft experts stats --model <gguf> --calib <text> then serve with --expert-usage-file <usage.json> (plus --expert-pin-budget) — the doc's M3 pinned warm subset + MADV_WILLNEED prefetch; improves hit rate/overlap, not raw volume.
- --moe-strategy cpu (or hybrid) would likely be fastest here by avoiding the PCIe traffic, but the docs note CPU/hybrid MoE reject the composite iq4_xs+iq4_nl expert format this file uses (docs/gguf.md:95), so it's blocked today.

A fix to try
The staged copy should gather all of a layer's miss rows in one index_select/cat into the ring, do one H2D and one index_copy_, and drop the per-expert Python loop and per-chunk sync — i.e. actually implement the M2 ring rather than the row-at-a-time version. Longer term, staged decode wants to be graph-capturable. That's a perf change, so per AGENTS.md it needs A/B numbers against main on this exact checkpoint.