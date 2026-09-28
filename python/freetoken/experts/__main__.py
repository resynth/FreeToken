"""``ft experts``: repack routed experts for the mmap source and rank their usage.

Subcommands:
  repack  Build a fixed-stride expert store from a GGUF (verbatim, no dequant).
  stats   Run a calibration corpus through the model and write a usage file.
"""

from __future__ import annotations

import argparse
import sys
import time


def _repack(args: argparse.Namespace) -> int:
    from freetoken.moe.expert_store import repack_gguf_experts

    t0 = time.time()
    index = repack_gguf_experts(
        args.model, args.out, drop_ple=args.drop_ple,
        usage_file=args.usage_file, warm_file=args.warm_file,
        hot_prefix=args.hot_prefix, hot_only=args.hot_only,
    )
    total = index.num_layers * index.num_experts * index.expert_bytes()
    hot = next((len(loc.hot_ids) for banks in index.banks.values() for loc in banks.values() if loc.hot_ids), 0)
    print(
        f"repacked {index.num_layers} layers x {index.num_experts} experts "
        f"({index.quant_format}) -> {args.out}\n"
        f"  {total / 2**30:.2f} GiB, fingerprint {index.fingerprint}, "
        f"{time.time() - t0:.1f}s"
        + (f", hot banks {hot}/layer" if hot else "")
    )
    print(f"serve with: ft serve --model {args.model} --expert-source mmap --expert-store {args.out}")
    if hot:
        ranked = args.usage_file or args.warm_file
        hint = "--expert-warm-file" if args.warm_file else "--expert-usage-file"
        print(
            f"pin {hot}/layer sequentially with: ft serve ... {hint} {ranked} "
            "(the pin build reads the hot banks as one sequential pread)"
        )
    return 0


def _stats(args: argparse.Namespace) -> int:
    import torch

    from freetoken.core import SamplingParams
    from freetoken.gpu_select import assign_gpu, bind_assigned_gpu
    from freetoken.moe.usage import UsageData

    try:
        assign_gpu(args.gpu)
        bind_assigned_gpu()
    except (ValueError, RuntimeError) as exc:
        print(f"ft experts stats: {exc}", file=sys.stderr)
        return 2

    from freetoken.llm import LLM

    with open(args.calib, encoding="utf-8") as f:
        text = f.read()
    dtype = getattr(torch, args.dtype)
    llm = LLM(
        args.model,
        dtype=dtype,
        moe_strategy="offload",
        moe_collect_stats=True,
        cuda_graph_bs=[],
        cuda_graph_max_bs=0,
        # Forwarded so calibration can use the mmap store; a split GGUF's store is not at
        # the default `<shard>.experts` path, so the auto path would otherwise error.
        expert_source=args.expert_source,
        expert_store=args.expert_store,
        expert_warm=args.expert_warm,
        ple_source=args.ple_source,
        ple_backend=args.ple_backend,
    )
    cache = getattr(llm.engine, "moe_offload_cache", None)
    if cache is None:
        print(f"ft experts stats: {args.model} has no offloaded expert cache", file=sys.stderr)
        return 2
    # The routing histogram is host-accumulated before the LRU kernel rewrites ids, so it
    # is only accurate without a captured decode graph (which is why graphs are disabled).
    cache.collect_decode_freq = True
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens, ignore_eos=True)
    llm.generate([text], sampling)
    usage = UsageData.from_freq(cache.decode_freq, source=args.calib)
    out = args.out or (args.model + ".usage.json")
    usage.save(out)
    totals = usage.totals()
    print(
        f"wrote {out}: {usage.num_layers} layers x {usage.num_experts} experts, "
        f"{sum(totals)} routed activations"
    )
    hint = f"--expert-usage-file {out} --expert-source mmap"
    if args.expert_store:
        hint += f" --expert-store {args.expert_store}"
    print(f"serve with: {hint}")
    return 0


def _add_common(p: argparse.ArgumentParser) -> None:
    from freetoken.gpu_select import single_gpu_arg

    p.add_argument("--gpu", type=single_gpu_arg, default=None, help="GPU UUID or nvidia-smi index (as ft serve --gpu)")


def _print_help(file) -> None:
    print(
        """usage: ft experts <subcommand> [args]

Subcommands:
  repack  Build a fixed-stride expert store from a GGUF (verbatim, no dequant)
  stats   Rank expert usage from a calibration corpus and write a usage file

Use "ft experts <subcommand> --help" for subcommand-specific options.""",
        file=file,
    )


def main(argv: list[str] | None = None, prog: str = "ft experts") -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        _print_help(sys.stdout if args else sys.stderr)
        return 0 if args else 2
    sub, rest = args[0], args[1:]
    if sub == "repack":
        p = argparse.ArgumentParser(prog=f"{prog} repack", description="Repack GGUF routed experts into a fixed-stride store.")
        p.add_argument("model", help="source .gguf (or a shard of a split GGUF)")
        p.add_argument("--out", required=True, help="output expert-store dir")
        p.add_argument("--drop-ple", action="store_true", help="do not archive the in-GGUF per-layer token embedding")
        p.add_argument(
            "--usage-file", default=None,
            help="ranked per-(layer, expert) counts from `ft experts stats`; with --hot-prefix, "
                 "also orders the hot banks (hottest first)",
        )
        p.add_argument(
            "--warm-file", default=None,
            help="retained-expert plan (JSON layer -> [expert ids], e.g. a REAP top-K dump); "
                 "with --hot-prefix, orders the hot banks by that plan instead",
        )
        p.add_argument(
            "--hot-prefix", type=int, default=0, metavar="K",
            help="also write per-(layer, role) hot banks holding the top-K experts "
                 "contiguously (duplicated bytes, K * expert_bytes per layer), so the "
                 "pinned warm subset builds with one sequential read; needs exactly one "
                 "of --usage-file / --warm-file",
        )
        p.add_argument(
            "--hot-only", action="store_true",
            help="patch hot banks into an existing store instead of rewriting the main "
                 "banks; verifies the store's fingerprint against this checkpoint",
        )
        return _repack(p.parse_args(rest))
    if sub == "stats":
        p = argparse.ArgumentParser(prog=f"{prog} stats", description="Rank expert usage from a calibration corpus.")
        p.add_argument("--model", required=True, help="checkpoint dir or .gguf")
        p.add_argument("--calib", required=True, help="calibration text file")
        p.add_argument("--out", default=None, help="usage file to write (default <model>.usage.json)")
        p.add_argument("--max-new-tokens", type=int, default=128, help="decode tokens to run")
        p.add_argument("--dtype", default="bfloat16", help="model dtype (default bfloat16)")
        p.add_argument(
            "--expert-source",
            default="auto",
            choices=("auto", "pinned", "mmap"),
            help="Expert source for the calibration run (as `ft serve --expert-source`); "
                 "'auto' serves a repacked store when the banks exceed the pin budget.",
        )
        p.add_argument(
            "--expert-store",
            default=None,
            help="Repacked expert store for the calibration run; needed for a split GGUF "
                 "whose store is not at the default `<shard>.experts` path.",
        )
        p.add_argument(
            "--expert-warm",
            action="store_true",
            help="Sequentially read the whole store once before calibrating (page-cache warm), "
                 "so the routed prefill reads resident pages.",
        )
        p.add_argument(
            "--ple-source",
            default=None,
            help="External fp8 PLE table (repo id or dir), as `ft serve --ple-source`; required "
                 "by qwen4exp GGUFs that do not carry the table inline.",
        )
        p.add_argument(
            "--ple-backend",
            default="disk",
            choices=("disk", "pinned"),
            help="PLE backend for the calibration run (default disk).",
        )
        _add_common(p)
        return _stats(p.parse_args(rest))
    print(f"unknown ft experts subcommand: {sub}", file=sys.stderr)
    _print_help(sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
