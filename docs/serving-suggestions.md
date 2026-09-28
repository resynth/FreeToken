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
  --expert-warm-file /home/b/ai/FreeToken/docs/qwen3.8-flash-next-top-384-experts-according-to-sh0wie.json \
  --expert-warm --moe-cache-auto \
  --text-model-only --max-running-requests 1 --max-seq-len-override 4096 --cuda-graph-max-bs 1 \
  --host 127.0.0.1 --port 8000

# F
  ft serve --model /media/b/Hyena/gguf/orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF/Qwen3.8-Flash-Next-Uncensored-IQ4_XS-00001-of-00003.gguf \
  --ple-source /home/b/.cache/huggingface/hub/models--Saren--Qwen3.8-Flash-Next-ple-table-fp8/snapshots/50511b0a41aa1d34b8beb7e5d4bb06a0b650dc14 \
  --expert-source mmap --expert-store /media/b/Hyena/qwen38-experts-store \
  --expert-warm-file /home/b/ai/FreeToken/docs/qwen3.8-flash-next-top-384-experts-according-to-sh0wie.json --expert-warm \
  --moe-cache-auto --text-model-only --max-running-requests 1 \
  --host 127.0.0.1 --port 8000

# D: CPU decode over the mmap store (no per-token PCIe, CUDA graphs stay on).
# No pins: --moe-strategy cpu fixes the slot cache to a two-layer prefill buffer.
  ft serve --model /media/b/Hyena/gguf/orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF/Qwen3.8-Flash-Next-Uncensored-IQ4_XS-00001-of-00003.gguf \
  --ple-source /home/b/.cache/huggingface/hub/models--Saren--Qwen3.8-Flash-Next-ple-table-fp8/snapshots/50511b0a41aa1d34b8beb7e5d4bb06a0b650dc14 \
  --moe-strategy cpu --expert-source mmap --expert-store /media/b/Hyena/qwen38-experts-store \
  --expert-warm --text-model-only --max-running-requests 1 \
  --host 127.0.0.1 --port 8000

# D alternative: hybrid (GPU fetches a share of each step's misses, CPU computes the rest).
  ft serve --model /media/b/Hyena/gguf/orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF/Qwen3.8-Flash-Next-Uncensored-IQ4_XS-00001-of-00003.gguf \
  --ple-source /home/b/.cache/huggingface/hub/models--Saren--Qwen3.8-Flash-Next-ple-table-fp8/snapshots/50511b0a41aa1d34b8beb7e5d4bb06a0b650dc14 \
  --moe-strategy hybrid --expert-source mmap --expert-store /media/b/Hyena/qwen38-experts-store \
  --expert-warm --moe-cache-auto --text-model-only --max-running-requests 1 \
  --host 127.0.0.1 --port 8000


cd ~/ai/FreeToken && source .venv/bin/activate
.venv/bin/ft shell --server http://127.0.0.1:8000

