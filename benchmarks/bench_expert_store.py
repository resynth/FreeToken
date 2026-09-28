"""Expert-store read-path + staged-copy microbenchmark (no model load).

Two sections, both against a repacked expert store (`ft experts repack`):

1. **Read paths** -- N random experts of one `(layer, role)` bank, through each path the
   mmap tier can use:
   - `mmap_random`  copy the mmap view row under `MADV_RANDOM` (the per-4-KiB-fault path),
   - `willneed`     `MADV_WILLNEED` the expert ranges first, then the mmap copy,
   - `pread_whole`  one buffered whole-expert `preadv` per miss (`ExpertSource.read_rows_into`),
   - `o_direct`     whole experts via O_DIRECT 256-KiB chunks (aligned extents only).
   Cold (best-effort page-cache drop) and warm are both reported as MiB/s and ms/expert.

2. **Staged copy** -- `OffloadMoeCache._copy_missing_staged` over the store's real bank
   shapes, swept over ring rows x miss count, cold/warm. This is the host-driven H2D path
   that C rewrote; use it to size `FREETOKEN_EXPERT_RING_ROWS` instead of A/B-ing the
   server, which is too slow and noisy to isolate anything before A/B land.

A final projection turns the measured whole-expert throughput into per-token and
per-1000-token-prefill wall time.

A bare GGUF has no store; `--store` is required for real numbers. Without it the script
synthesizes a tiny local store so the plumbing can be smoke-tested anywhere.

Run:
  PYTHONPATH=python python benchmarks/bench_expert_store.py --store /path/to/store
  PYTHONPATH=python python benchmarks/bench_expert_store.py            # synthetic store
"""

from __future__ import annotations

import argparse
import json
import mmap
import os
import random
import statistics
import tempfile
import time

import torch

from freetoken.moe.expert_source import MmapExpertSource
from freetoken.moe.expert_store import (
    STORE_FORMAT,
    STORE_VERSION,
    ExpertStoreIndex,
)

MiB = 1 << 20


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--store", default=None, help="expert store dir; omit to synthesize one")
    p.add_argument("--section", choices=["all", "read", "staged"], default="all")
    p.add_argument("--gpu", default=None, help="GPU UUID or nvidia-smi index (staged section)")
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--role", default="gate_up")
    p.add_argument("--experts", type=int, default=32, help="read paths: how many random experts")
    p.add_argument("--repeat", type=int, default=7)
    p.add_argument(
        "--cold", action="store_true",
        help="best-effort page-cache drop before each sample (fadvise DONTNEED; "
             "rarely fully cold for a live mmap)",
    )
    p.add_argument("--aggressive-drop", action="store_true",
                   help="also write /proc/sys/vm/drop_caches (needs root)")
    p.add_argument("--ring-rows", type=int, nargs="+", default=[8, 32, 512])
    p.add_argument("--miss-counts", type=int, nargs="+", default=[8, 32, 512])
    p.add_argument("--per-token-experts", type=int, default=480,
                   help="routed experts per decode token, for the projection")
    p.add_argument("--legacy", action="store_true",
                   help="staged section: also time the pre-C copy loop (two syncs, per-chunk "
                        "stream drain) for an A/B")
    return p.parse_args()


# --------------------------------------------------------------------------- store helpers


def make_synthetic_store(root: str, *, layers: int = 2, experts: int = 512,
                         hidden: int = 256, inter: int = 64) -> None:
    """A tiny valid store with the same file/index layout as `ft experts repack`."""
    os.makedirs(root, exist_ok=True)
    gu_rows, gu_row_bytes = 2 * inter, (hidden // 32) * 18
    dn_rows, dn_row_bytes = hidden, (inter // 32) * 18
    rng = random.Random(0)
    index_layers = []
    for layer in range(layers):
        gu = os.path.join(root, f"layer-{layer:03d}.gate_up.bin")
        dn = os.path.join(root, f"layer-{layer:03d}.down.bin")
        with open(gu, "wb") as f:
            f.write(bytes(rng.getrandbits(8) for _ in range(min(experts * gu_rows * gu_row_bytes, 1 << 22))))
            f.truncate(experts * gu_rows * gu_row_bytes)
        with open(dn, "wb") as f:
            f.write(bytes(rng.getrandbits(8) for _ in range(min(experts * dn_rows * dn_row_bytes, 1 << 22))))
            f.truncate(experts * dn_rows * dn_row_bytes)
        index_layers.append({"layer": layer, "banks": {
            "gate_up": {"file": os.path.basename(gu), "offset": 0,
                        "stride": gu_rows * gu_row_bytes, "rows": gu_rows, "row_bytes": gu_row_bytes},
            "down": {"file": os.path.basename(dn), "offset": 0,
                     "stride": dn_rows * dn_row_bytes, "rows": dn_rows, "row_bytes": dn_row_bytes},
        }})
    doc = {
        "format": STORE_FORMAT, "version": STORE_VERSION, "align": 4096,
        "quant_format": "iq4_nl", "num_layers": layers, "num_experts": experts,
        "hidden_size": hidden, "intermediate_size": inter, "roles": ["gate_up", "down"],
        "fingerprint": "synthetic", "layers": index_layers,
    }
    with open(os.path.join(root, "index.json"), "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, sort_keys=True)
        f.write("\n")


def drop_page_cache(path: str, aggressive: bool) -> None:
    """Best-effort eviction of a bank file's clean page-cache pages."""
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    except (OSError, AttributeError):
        pass
    if aggressive:
        try:
            with open("/proc/sys/vm/drop_caches", "w", encoding="ascii") as f:
                f.write("3\n")
        except OSError:
            pass


def _time(fn, repeat: int, before=None) -> float:
    fn()  # warm-up / fault-in
    samples = []
    for _ in range(repeat):
        if before is not None:
            before()
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


# --------------------------------------------------------------------------- read paths


class _ODirectReader:
    """Whole-expert O_DIRECT reads (256-KiB chunks) into a page-aligned scratch mapping."""

    def __init__(self, path: str, loc):
        self.fd = -1
        self.loc = loc
        self.chunk = 256 << 10
        self.scratch = None
        if not getattr(os, "O_DIRECT", 0):
            return
        try:
            self.fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
        except OSError:
            return
        self.scratch = mmap.mmap(-1, self.chunk)

    def aligned(self, ids) -> bool:
        # O_DIRECT wants the offset, length and buffer address on the device block size
        return (
            self.loc.stride % 512 == 0
            and all(self.loc.expert_offset(e) % 4096 == 0 for e in ids)
        )

    def read_many(self, ids) -> None:
        mv = memoryview(self.scratch)
        for e in ids:
            off = self.loc.expert_offset(e)
            done = 0
            while done < self.loc.stride:
                want = min(self.chunk, self.loc.stride - done)
                got = os.preadv(self.fd, [mv[:want]], off + done)
                if got <= 0:
                    raise OSError(f"O_DIRECT short read at {off + done}")
                done += got

    def close(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
        if self.scratch is not None:
            self.scratch.close()


def bench_read_paths(store_dir: str, args: argparse.Namespace) -> dict[str, float]:
    """Measure each read path; return the median ms/expert for whole-expert reads."""
    index = ExpertStoreIndex.load(store_dir)
    loc = index.location(args.layer, args.role)
    n = min(args.experts, index.num_experts)
    ids = random.Random(1234).sample(range(index.num_experts), n)
    source = MmapExpertSource.open(store_dir)
    view = source.layer_views(args.layer)[args.role]
    row = torch.empty((loc.rows, loc.row_bytes), dtype=torch.uint8)
    dst = torch.empty((n, loc.rows, loc.row_bytes), dtype=torch.uint8)
    bank_path = source._path(args.layer, args.role)
    expert_mib = loc.stride / MiB

    def drop() -> None:
        # Unmap the source's own view first so fadvise can actually free the page cache;
        # MADV_DONTNEED on a read-only mapping also restores the per-page fault cost.
        mm = source._mmap(args.layer, args.role)
        if hasattr(mm, "madvise"):
            try:
                mm.madvise(mmap.MADV_DONTNEED)
            except (OSError, ValueError):
                pass
        drop_page_cache(bank_path, args.aggressive_drop)

    before = drop if args.cold else None

    def mmap_copy() -> None:
        for e in ids:
            row.copy_(view[e])

    def willneed_copy() -> None:
        source.prefetch(args.layer, ids)
        for e in ids:
            row.copy_(view[e])

    def pread_whole() -> None:
        source.read_rows_into(args.layer, args.role, ids, dst)

    reader = _ODirectReader(bank_path, loc)
    odirect_ok = reader.fd >= 0 and reader.aligned(ids)

    def o_direct() -> None:
        reader.read_many(ids)

    paths: list[tuple[str, object]] = [
        ("mmap_random", mmap_copy),
        ("willneed", willneed_copy),
        ("pread_whole", pread_whole),
        ("o_direct", o_direct if odirect_ok else None),
    ]

    print(f"\nread paths: layer {args.layer}, {args.role!r}, {n} experts x {expert_mib:.2f} MiB "
          f"(cold={args.cold})")
    print(f"{'path':<12} {'cold MiB/s':>10} {'cold ms/ex':>10} "
          f"{'warm MiB/s':>10} {'warm ms/ex':>10}")
    print("-" * 58)
    result: dict[str, float] = {}
    for name, fn in paths:
        if fn is None:
            print(f"{name:<12} {'-':>10} {'-':>10} {'-':>10} {'-':>10}")
            continue
        warm_s = _time(fn, args.repeat)
        cold_s = _time(fn, args.repeat, before=before) if before is not None else float("nan")
        cold_bw = (n * expert_mib / cold_s) if cold_s == cold_s else float("nan")
        warm_bw = n * expert_mib / warm_s
        cold_per = cold_s / n * 1e3 if cold_s == cold_s else float("nan")
        warm_per = warm_s / n * 1e3
        cold_cell = f"{cold_bw:>10.1f}" if cold_bw == cold_bw else f"{'-':>10}"
        cold_ms_cell = f"{cold_per:>10.3f}" if cold_per == cold_per else f"{'-':>10}"
        print(f"{name:<12} {cold_cell} {cold_ms_cell} "
              f"{warm_bw:>10.1f} {warm_per:>10.3f}")
        if name in ("pread_whole", "o_direct"):
            result[f"{name}_cold_ms"] = cold_per
            result[f"{name}_warm_ms"] = warm_per

    reader.close()
    source.close()
    return result


def print_projection(store_bytes: int, expert_bytes: int, args: argparse.Namespace,
                     times: dict[str, float]) -> None:
    """Per-token / per-1000-token-prefill time from the measured whole-expert reads.

    A 1000-token prefill materializes every layer's whole bank once (non-overlap), so it
    touches the whole store; decode touches ``per-token-experts`` experts per token.
    """
    total_experts = store_bytes / expert_bytes
    print(f"\nprojection ({store_bytes / 2**30:.2f} GiB store, "
          f"{args.per_token_experts} experts/token):")
    for path in ("pread_whole", "o_direct"):
        warm = times.get(f"{path}_warm_ms")
        if warm is None:
            continue
        for label, ms in (("cold", times.get(f"{path}_cold_ms", float("nan"))), ("warm", warm)):
            if ms != ms:
                continue
            per_token = args.per_token_experts * ms / 1e3
            prefill = total_experts * ms / 1e3
            print(f"  {path:<11} {label:<4} per-token {per_token * 1e3:9.1f} ms   "
                  f"per-1000-token prefill {prefill:9.1f} s")


# --------------------------------------------------------------------------- staged copy


def _legacy_copy_missing_staged(cache, layer_id: int) -> None:
    """Pre-C staged copy: two device syncs, one chunk at a time, full stream drain between."""
    n = int(cache.num_indices.item())
    if n <= 0:
        return
    cache._ensure_staging()  # buffer 0 stands in for the old single ring
    evict = cache.evict_slots[:n]
    src_ids = cache.src_indices[:n].cpu().tolist()
    source = cache.expert_source
    read_rows = getattr(source, "read_rows_into", None)
    for role, (per_layer, slot_cache) in zip(cache.bank_schema, cache.banks):
        ring = cache._staging_ring[role][0]
        device = cache._staging_device[role][0]
        rows = ring.shape[0]
        for start in range(0, n, rows):
            m = min(rows, n - start)
            chunk = src_ids[start : start + m]
            if read_rows is not None:
                read_rows(layer_id, role, chunk, ring[:m])
            else:
                for c in range(m):
                    ring[c].copy_(per_layer[layer_id][int(chunk[c])])
            device[:m].copy_(ring[:m], non_blocking=True)
            slot_cache.index_copy_(0, evict[start : start + m].long(), device[:m])
            if start + m < n:
                torch.cuda.current_stream(cache.device).synchronize()


def bench_staged(store_dir: str, args: argparse.Namespace, device: torch.device) -> None:
    from freetoken.moe.offload_cache import _BANK_SCHEMAS, OffloadMoeCache

    index = ExpertStoreIndex.load(store_dir)
    layers = index.num_layers
    cache_size = max(index.num_experts, max(args.miss_counts))
    if index.quant_format not in _BANK_SCHEMAS:
        print(f"staged: store format {index.quant_format!r} has no generic bank schema; skipping")
        return

    print(f"\nstaged copy: _copy_missing_staged ({layers} layers x {index.num_experts} experts, "
          f"cache_size {cache_size}, device {device})")
    columns = f"{'ring_rows':>9} {'misses':>7} {'new ms':>9} {'MiB/s':>9}"
    if args.legacy:
        columns += f" {'legacy ms':>10} {'legacy MiB/s':>13} {'speedup':>8}"
    print(columns)
    print("-" * len(columns))

    for ring_rows in args.ring_rows:
        os.environ["FREETOKEN_EXPERT_RING_ROWS"] = str(ring_rows)
        source = MmapExpertSource.open(store_dir)
        cache = OffloadMoeCache(
            num_layers=layers, num_experts=index.num_experts, cache_size=cache_size,
            device=device, quant_format=index.quant_format,
        )
        cache.set_bank_sources(
            source.all_layer_views(),
            layer_residency=["mmap"] * layers,
            expert_source=source,
        )
        for misses in args.miss_counts:
            misses = min(misses, index.num_experts, cache_size)
            cache.evict_slots[:misses] = torch.arange(misses, dtype=torch.int32, device=device)
            cache.src_indices[:misses] = torch.tensor(
                list(range(misses)), dtype=torch.int32, device=device
            )

            def run() -> None:
                cache.num_indices.fill_(misses)
                cache._copy_missing_staged(0)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)

            def run_legacy() -> None:
                cache.num_indices.fill_(misses)
                _legacy_copy_missing_staged(cache, 0)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)

            ms = _time(run, args.repeat) * 1e3
            moved = misses * index.expert_bytes() / MiB
            line = f"{ring_rows:>9} {misses:>7} {ms:>9.3f} {moved / (ms / 1e3):>9.1f}"
            if args.legacy:
                ms_legacy = _time(run_legacy, args.repeat) * 1e3
                line += (f" {ms_legacy:>10.3f} {moved / (ms_legacy / 1e3):>13.1f} "
                         f"{ms_legacy / ms:>7.2f}x")
            print(line, flush=True)
        del cache
        source.close()
        if device.type == "cuda":
            torch.cuda.empty_cache()


# --------------------------------------------------------------------------- main


def main() -> None:
    args = parse_args()
    tmp = None
    store_dir = args.store
    if store_dir is None:
        tmp = tempfile.mkdtemp(prefix="ft-expert-store-bench-")
        make_synthetic_store(tmp)
        store_dir = tmp
        print(f"no --store: synthesized {store_dir}")

    index = ExpertStoreIndex.load(store_dir)
    expert_bytes = index.expert_bytes()
    store_bytes = expert_bytes * index.num_experts * index.num_layers
    print(f"store: {store_dir} ({index.quant_format}, {index.num_layers} layers x "
          f"{index.num_experts} experts, {expert_bytes / MiB:.2f} MiB/expert, "
          f"{store_bytes / 2**30:.2f} GiB)")

    if args.section in ("all", "read"):
        times = bench_read_paths(store_dir, args)
        print_projection(store_bytes, expert_bytes, args, times)

    if args.section in ("all", "staged"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda" and args.gpu is not None:
            from freetoken.gpu_select import assign_gpu, bind_assigned_gpu, single_gpu_arg

            assign_gpu(single_gpu_arg(args.gpu))
            device = bind_assigned_gpu()
        bench_staged(store_dir, args, device)

    if tmp is not None:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
