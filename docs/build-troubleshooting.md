# Build troubleshooting

Notes from building FreeToken from source on a machine whose toolchain did not match the
wheels. Install steps are in [install.md](install.md); this file covers the failure modes.

## CUDA toolkit major must match torch
`torch` is pinned to the cu130 wheel set, so `nvcc` must be 13.x. `_check_toolchain`
(`python/freetoken/kernel/_toolchain.py`) refuses to compile across CUDA majors because
nvcc-built binaries link `libcudart.so.<major>` while only torch's runtime is guaranteed
loadable.

- Symptom: `RuntimeError: nvcc 12.8 would build kernels linking libcudart.so.12, but torch
  2.11.0+cu130 ships CUDA 13.0`.
- Fix: install a CUDA 13.x toolkit and point the build at it (`CUDA_HOME=/usr/local/cuda-13.0`,
  `PATH=$CUDA_HOME/bin:$PATH`). `_nvcc_path()` prefers `torch.utils.cpp_extension.CUDA_HOME`.
- Escape hatch: `FREETOKEN_ALLOW_CUDA_MISMATCH=1` (unsupported; may fail at runtime).

## Host compiler: gcc for `setup.py`, clang for the JIT kernels
- `setup.py` host-only extensions (`_pinned_tensor`, `_cpu_moe`, `_row_store`) need **GCC**:
  `cpu_moe_ext.cpp` uses `__builtin_cpu_supports("avxvnni")`, a GCC-only feature string clang
  rejects. Build with `CXX=g++-13 CC=gcc-13`.
- The gguf/CUDA JIT (`kernel/gguf.py:_host_compiler`) prefers **clang++** and passes
  `-ccbin clang++` to nvcc (nvcc+gcc-13 trips a non-conformant `typename decltype` in torch's
  `List_inl.h`). It sets `CXX` itself; keep clang++ on PATH.
- Do not mix: a gcc build with clang-only flags (`-fdebug-default-version=4`) fails with
  `unrecognized command-line option`, and a clang build of `_cpu_moe` fails on `avxvnni`.

## A Python whose sysconfig bakes in clang flags
Some uv-managed standalone CPython builds were compiled with clang and a custom prefix, and
their `sysconfig` records that:

```
$ python -c "import sysconfig; print(sysconfig.get_config_var('CFLAGS'))"
... -fdebug-default-version=4 -fPIC -I/tools/deps/include ...
```

distutils copies `CFLAGS`/`CPPFLAGS` from `sysconfig` into **every** extension build, and
`unset CFLAGS` cannot remove them. Fix: create the venv from a system Python whose sysconfig
is clean (`uv venv --python /usr/bin/python3.12`) and check with the command above.

## Build and verify

```bash
source .venv/bin/activate
CXX=g++-13 CC=gcc-13 FREETOKEN_ALLOW_CUDA_MISMATCH=1 uv pip install -e ".[accel]"
CXX=g++-13 CC=gcc-13 FREETOKEN_ALLOW_CUDA_MISMATCH=1 python setup.py build_ext --inplace
python -c "import freetoken.kernel._cpu_moe, freetoken.kernel._pinned_tensor, freetoken.kernel._row_store; print('ext ok')"
```

CUDA/Triton kernels are JIT-compiled on first use; the first `ft serve` (or a GPU test) will
build them, so give it a few minutes.

## `ft: cannot execute: required file not found` (venv moved)
Console scripts bake their venv's interpreter into the shebang. If the venv was created at
`<repo>/.venv` and the checkout was later moved or renamed (or a second checkout was made),
every generated script points at the old path. Symptom:

```
$ ft serve ...
bash: /home/b/ai/FreeToken-OG/.venv/bin/ft: cannot execute: required file not found
$ source .venv/bin/activate
bash: .venv/bin/activate: No such file or directory
```

Check with `head -1 $(command -v ft)`; it names the missing interpreter. Fix by restoring the
path the venv expects (`ln -s <current-venv> <old-venv-path>`) or by recreating the venv at
the baked-in path. `python -m freetoken.cli <command>` (or any `<venv>/bin/python -m
freetoken.cli ...`) bypasses the console script entirely, and `ft serve` accepts `--model`
as an alias of `--model-path`.

## Expert-bank memory (offload)
The offload cache pins the whole packed expert set in host RAM at startup; page-locked memory
cannot be reclaimed or swapped, so a model whose banks exceed available RAM used to OOM the
host. The engine budgets 90% of `MemAvailable` (Linux; `FREETOKEN_PIN_BUDGET_GB` overrides)
and, when the banks exceed it, serves them from a repacked mmap store instead (`ft experts
repack` + `--expert-source mmap`); without a store it still fails fast with a clear message.
See [mmap-expert-tiering.md](mmap-expert-tiering.md) for the hot/warm/cold design.
