"""Repacked, fixed-stride expert store for the mmap expert source.

Random access must be cheap, so each ``(layer, role)`` expert bank is repacked into one
file whose leading dim is the expert index:

```
experts/
  index.json                 # geometry, per-(layer, role) file/offset/stride, source fingerprint
  layer-000.gate_up.bin
  layer-000.down.bin
  ...
```

``row_bytes`` is constant per role and ``expert`` is the leading dim, so expert ``e`` of
role ``r`` in layer ``l`` is the byte range ``[offset(l, r) + e * stride(r), +stride(r))``
-- a single ``mmap`` slice or ``pread``. The repack writer streams the GGUF expert tensors
verbatim (no dequant), preallocates each output file (``posix_fallocate``) and writes
sequentially so the store lands in contiguous extents.

``repack_gguf_experts(..., hot_prefix=K, usage_file=...)`` additionally writes a per-
``(layer, role)`` *hot bank* (``layer-000.gate_up.hot.bin``): the top-K experts by usage
rank, contiguously, duplicating those bytes. The hot bank exists only so the pinned warm
subset builds with one sequential read; every serving path keeps reading the main bank,
because duplicated pages would only evict the page cache the main banks need. The row
order of the main banks is *not* permuted: the per-layer host bank contract
(``ExpertSource.all_layer_views`` -> ``row == logical expert id``) is load-bearing for the
whole-layer prefill copy, the LRU ``src_indices`` remap and the CPU executor's pointer
tables.

The store is read-only at serving time; see
:class:`~freetoken.moe.expert_source.MmapExpertSource`.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Callable, Iterable

from freetoken.utils import init_logger

logger = init_logger(__name__)

STORE_FORMAT = "freetoken_experts"
STORE_VERSION = 1
INDEX_NAME = "index.json"
ALIGN = 4096  # fallocate/write alignment; matches the O_DIRECT block elsewhere
_WRITE_CHUNK = 8 << 20

# GGUF tensor names that carry a per-layer token embedding table (PLE). The store holds
# expert banks; PLE is archived only when --drop-ple is not passed.
_PLE_SUFFIXES = ("per_layer_token_embd.weight",)


def default_store_dir(model_path: str) -> str | None:
    """Where ``ft experts repack`` writes and the engine looks for the store.

    ``--expert-store`` overrides this; ``FREETOKEN_EXPERT_STORE`` is the env fallback.
    A bare GGUF file maps to ``<file>.experts``; a checkpoint dir to ``<dir>/experts``.
    """
    env = os.environ.get("FREETOKEN_EXPERT_STORE")
    if env:
        return env
    if not model_path:
        return None
    if os.path.isfile(model_path) and model_path.endswith(".gguf"):
        return model_path + ".experts"
    if os.path.isdir(model_path):
        return os.path.join(model_path, "experts")
    return None


def is_expert_store(path: str | None) -> bool:
    """A directory carrying a readable expert-store index."""
    if not path or not os.path.isdir(path):
        return False
    return os.path.isfile(os.path.join(path, INDEX_NAME))


@dataclass(frozen=True)
class BankLocation:
    """One ``(layer, role)`` bank file and the addressing within it."""

    file: str
    offset: int
    stride: int  # expert_bytes
    rows: int
    row_bytes: int
    # J: an optional hot bank holding a per-layer usage-ranked subset of the same
    # experts contiguously (hottest first). It exists only so the pinned warm subset
    # can be built with one sequential read; every serving path keeps reading the
    # main bank (duplicated pages would only evict the page cache the main banks need).
    hot_file: str | None = None
    hot_ids: tuple[int, ...] = ()  # logical expert ids in hot-bank physical order

    @property
    def expert_bytes(self) -> int:
        return self.rows * self.row_bytes

    def expert_offset(self, expert: int) -> int:
        return self.offset + expert * self.stride


@dataclass(frozen=True)
class ExpertStoreIndex:
    """The parsed ``index.json``."""

    quant_format: str
    num_layers: int
    num_experts: int
    hidden_size: int
    intermediate_size: int
    roles: tuple[str, ...]
    banks: dict[int, dict[str, BankLocation]]
    fingerprint: str = ""
    source_path: str | None = None
    ple: dict | None = None
    align: int = ALIGN

    @classmethod
    def load(cls, store_dir: str) -> "ExpertStoreIndex":
        path = os.path.join(store_dir, INDEX_NAME)
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
        if doc.get("format") != STORE_FORMAT:
            raise ValueError(
                f"{store_dir}: not an expert store (format={doc.get('format')!r}, "
                f"expected {STORE_FORMAT!r})"
            )
        if int(doc.get("version", -1)) != STORE_VERSION:
            raise ValueError(
                f"{store_dir}: expert store version {doc.get('version')} unsupported "
                f"(expected {STORE_VERSION})"
            )
        banks: dict[int, dict[str, BankLocation]] = {}
        for entry in doc["layers"]:
            layer = int(entry["layer"])
            banks[layer] = {
                role: BankLocation(
                    file=loc["file"],
                    offset=int(loc.get("offset", 0)),
                    stride=int(loc["stride"]),
                    rows=int(loc["rows"]),
                    row_bytes=int(loc["row_bytes"]),
                    hot_file=loc.get("hot_file"),
                    hot_ids=tuple(int(e) for e in loc.get("hot_ids", ())),
                )
                for role, loc in entry["banks"].items()
            }
        return cls(
            quant_format=str(doc["quant_format"]),
            num_layers=int(doc["num_layers"]),
            num_experts=int(doc["num_experts"]),
            hidden_size=int(doc["hidden_size"]),
            intermediate_size=int(doc["intermediate_size"]),
            roles=tuple(doc["roles"]),
            banks=banks,
            fingerprint=str(doc.get("fingerprint", "")),
            source_path=doc.get("source_path"),
            ple=doc.get("ple"),
            align=int(doc.get("align", ALIGN)),
        )

    def location(self, layer: int, role: str) -> BankLocation:
        try:
            return self.banks[layer][role]
        except KeyError as exc:
            raise KeyError(f"expert store has no ({layer}, {role!r}) bank") from exc

    def expert_bytes(self) -> int:
        """Bytes of one expert across every role (the per-expert cost)."""
        return sum(self.location(0, role).expert_bytes for role in self.roles)

    def resolve_file(self, store_dir: str, location: BankLocation) -> str:
        # the index is checkpoint-supplied and may be crafted; keep every bank file inside
        # the store dir (realpath also resolves symlink escapes)
        base = os.path.realpath(store_dir)
        cand = os.path.realpath(os.path.join(base, location.file))
        if cand != base and not cand.startswith(base + os.sep):
            raise ValueError(
                f"expert store {store_dir!r}: bank file {location.file!r} escapes the store dir"
            )
        return cand


def _fingerprint(entries: Iterable[dict]) -> str:
    """Cheap identity of the source tensor table (names/types/shapes), not the weight bytes."""
    payload = json.dumps(list(entries), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _write_bytes(fd: int, data, *, chunk: int = _WRITE_CHUNK) -> int:
    """Write ``data`` (buffer or numpy view) to ``fd`` in large sequential chunks."""
    import numpy as np

    arr = np.asarray(data).reshape(-1).view(np.uint8)
    total = int(arr.nbytes)
    mv = memoryview(arr)
    off = 0
    while off < total:
        off += os.write(fd, mv[off : off + chunk])
    return total


def _preallocated_file(path: str, size: int) -> int:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.posix_fallocate(fd, 0, size)
    except AttributeError:
        os.ftruncate(fd, size)
    except OSError as exc:
        if exc.errno in (errno.ENOSPC, errno.EDQUOT):
            os.close(fd)
            os.unlink(path)
            raise
        # other POSIX filesystems (unsupported fallocate) fall back to a plain truncate;
        # the writes still land, they just may not be one contiguous extent
        os.ftruncate(fd, size)
    return fd


def _write_bank(path: str, size: int, write) -> None:
    """Write one bank file atomically: ``*.tmp`` -> fsync -> rename."""
    tmp = path + ".tmp"
    fd = _preallocated_file(tmp, size)
    try:
        write(fd)
        os.fsync(fd)
    except BaseException:
        os.close(fd)
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    os.close(fd)
    os.replace(tmp, path)


def _hot_ranking(
    *, usage_file: str | None, warm_file: str | None, hot_prefix: int,
    num_layers: int, num_experts: int,
) -> list[list[int]] | None:
    """Per-layer expert ids to place in the hot banks (hottest first); None = no hot banks.

    Exactly one ranking input is accepted: ``usage_file`` (a ranked ``ft experts stats``
    dump) or ``warm_file`` (an unranked retained-expert plan such as a REAP top-K dump --
    contiguity is what the pin build wants, not the order within the set). A layer the
    plan omits gets an empty list, i.e. no hot bank.
    """
    from freetoken.moe.usage import load_usage, load_warm_plan

    if hot_prefix <= 0:
        if usage_file or warm_file:
            raise ValueError("--usage-file/--warm-file rank the hot banks; pass --hot-prefix too")
        return None
    if (usage_file is None) == (warm_file is None):
        raise ValueError(
            "--hot-prefix needs exactly one ranking input: --usage-file (ranked counts) "
            "or --warm-file (a retained-expert plan)"
        )
    k = min(hot_prefix, num_experts)
    ranking: list[list[int]] = []
    if usage_file is not None:
        usage = load_usage(usage_file)
        if usage.num_layers != num_layers or usage.num_experts != num_experts:
            raise ValueError(
                f"usage file {usage_file!r} holds {usage.num_layers} x {usage.num_experts} "
                f"counts but the model has {num_layers} x {num_experts} experts"
            )
        return [usage.rank(layer)[:k] for layer in range(num_layers)]
    plan = load_warm_plan(warm_file)
    out_of_range = [layer for layer in plan if layer >= num_layers]
    if out_of_range:
        raise ValueError(
            f"warm plan {warm_file!r} names layer {max(out_of_range)} but the model has "
            f"{num_layers} MoE layers"
        )
    for layer in range(num_layers):
        ids: list[int] = []
        for e in plan.get(layer, []):
            if not 0 <= int(e) < num_experts:
                raise ValueError(
                    f"warm plan {warm_file!r}: expert {e} is out of range for layer {layer}"
                )
            if int(e) not in ids:
                ids.append(int(e))
        ranking.append(ids[:k])
    return ranking


def _verify_patchable(
    existing: ExpertStoreIndex, out_dir: str, *, num_layers: int, num_experts: int,
    hidden: int, inter: int, quant_format: str, fingerprint: str,
) -> None:
    """--hot-only safety: the existing store must be this checkpoint's own repack."""
    if not existing.fingerprint:
        raise ValueError(
            f"--hot-only: {out_dir} carries no fingerprint; run a full `ft experts repack` first"
        )
    problems = []
    if existing.fingerprint != fingerprint:
        problems.append(f"fingerprint {existing.fingerprint} != this checkpoint's {fingerprint}")
    if (existing.num_layers, existing.num_experts, existing.hidden_size,
            existing.intermediate_size) != (num_layers, num_experts, hidden, inter):
        problems.append(
            f"geometry {existing.num_layers} x {existing.num_experts} of "
            f"{existing.hidden_size}/{existing.intermediate_size} != "
            f"{num_layers} x {num_experts} of {hidden}/{inter}"
        )
    if existing.quant_format != quant_format:
        problems.append(f"quant_format {existing.quant_format!r} != {quant_format!r}")
    if problems:
        raise ValueError(
            f"--hot-only: {out_dir} was not built from this checkpoint "
            f"({'; '.join(problems)}); run a full repack"
        )


def repack_gguf_experts(
    model_path: str,
    out_dir: str,
    *,
    drop_ple: bool = False,
    usage_file: str | None = None,
    warm_file: str | None = None,
    hot_prefix: int = 0,
    hot_only: bool = False,
    progress: Callable[[str], None] | None = None,
) -> ExpertStoreIndex:
    """Repack a GGUF's routed experts into ``out_dir`` verbatim (no dequant).

    With ``hot_prefix > 0`` and exactly one ranking input (``usage_file``: ranked
    counts; ``warm_file``: a retained-expert plan), additionally writes a per-
    ``(layer, role)`` *hot bank*: the top experts contiguously, hottest first,
    duplicating those bytes. The hot bank exists only so the pinned warm subset
    builds with one sequential read (`MmapExpertSource._build_pins`); serving
    paths keep reading the main banks. ``hot_only`` patches hot banks into an
    existing store after verifying it was built from this checkpoint.

    Returns the written index. Raises for a checkpoint whose expert tensors are missing,
    mixed, or non-contiguous across layers.
    """
    from freetoken.models.gguf.dequant import row_bytes
    from freetoken.models.gguf.reader import iter_gguf_tensors
    from freetoken.moe.gguf_experts import gguf_expert_format, gguf_expert_role_types

    def say(msg: str) -> None:
        logger.info_rank0(msg)
        if progress is not None:
            progress(msg)

    os.makedirs(out_dir, exist_ok=True)
    quant_format = gguf_expert_format(model_path)
    gate_up_type, down_type = gguf_expert_role_types(quant_format)

    # Collect the per-(layer, role) GGUF tensors (memmap views, no copy) and the source
    # tensor table for the fingerprint. Separate gate/up are fused while writing.
    layers: dict[int, dict[str, object]] = {}
    table: list[dict] = []
    ple_tensor = None
    for t in iter_gguf_tensors(model_path):
        name = t.name
        table.append({"name": name, "type": t.ggml_type, "shape": list(t.shape)})
        if not name.startswith("blk."):
            if any(name.endswith(suffix) for suffix in _PLE_SUFFIXES):
                ple_tensor = t
            continue
        try:
            layer = int(name.split(".")[1])
        except (IndexError, ValueError):
            continue
        slot = layers.setdefault(layer, {})
        if name.endswith("ffn_gate_up_exps.weight"):
            slot["gate_up"] = t
        elif name.endswith("ffn_down_exps.weight"):
            slot["down"] = t
        elif name.endswith("ffn_gate_exps.weight"):
            slot["gate"] = t
        elif name.endswith("ffn_up_exps.weight"):
            slot["up"] = t
    if not layers:
        raise ValueError(f"{model_path}: no routed expert tensors (ffn_*_exps.weight) found")

    incomplete = [
        layer for layer, slot in layers.items()
        if "down" not in slot or ("gate_up" not in slot and not ("gate" in slot and "up" in slot))
    ]
    if incomplete:
        raise ValueError(f"{model_path}: incomplete expert tensors for layers {sorted(incomplete)}")
    num_layers = max(layers) + 1
    if set(layers) != set(range(num_layers)):
        raise ValueError(f"{model_path}: expert layers are not contiguous 0..{num_layers - 1}")

    first = layers[0]
    if "gate_up" in first:
        # torch shape (E, 2I, H): the last axis is the ggml fastest dim (hidden)
        hidden = int(first["gate_up"].shape[-1])
        inter = int(first["gate_up"].shape[1]) // 2
    else:
        # gate shape (E, I, H), down shape (E, H, I)
        hidden = int(first["gate"].shape[-1])
        inter = int(first["down"].shape[-1])
    num_experts = int(first["down"].shape[0])

    h_bytes = row_bytes(hidden, gate_up_type)
    i_bytes = row_bytes(inter, down_type)

    def dims_ok(t, want_rows: int, want_row_bytes: int) -> bool:
        return (
            t.shape[0] == num_experts
            and int(t._raw.shape[1]) == want_row_bytes
            and int(t._raw.shape[0]) == num_experts * want_rows
        )

    for layer, slot in layers.items():
        checks = []
        if "gate_up" in slot:
            checks.append((slot["gate_up"], 2 * inter, h_bytes))
        else:
            checks.append((slot["gate"], inter, h_bytes))
            checks.append((slot["up"], inter, h_bytes))
        if "down" in slot:
            checks.append((slot["down"], hidden, i_bytes))
        for t, want_rows, want_row_bytes in checks:
            if not dims_ok(t, want_rows, want_row_bytes):
                raise ValueError(
                    f"{model_path}: expert tensor {t.name} shape {tuple(t.shape)} is inconsistent "
                    f"with {num_experts} experts of {want_rows} rows x {want_row_bytes} bytes "
                    f"(layer {layer})"
                )

    fingerprint = _fingerprint(table)
    hot_ids_per_layer = _hot_ranking(
        usage_file=usage_file, warm_file=warm_file, hot_prefix=hot_prefix,
        num_layers=num_layers, num_experts=num_experts,
    )
    existing = None
    if hot_only:
        if hot_ids_per_layer is None:
            raise ValueError("--hot-only needs --hot-prefix with --usage-file or --warm-file")
        if drop_ple:
            raise ValueError("--hot-only writes hot banks only; run a full repack to change PLE archiving")
        existing = ExpertStoreIndex.load(out_dir)
        _verify_patchable(
            existing, out_dir, num_layers=num_layers, num_experts=num_experts,
            hidden=hidden, inter=inter, quant_format=quant_format, fingerprint=fingerprint,
        )

    index_banks: dict[int, dict[str, dict]] = {}
    written: set[str] = set()
    if existing is not None:
        # keep the existing main banks and PLE entry verbatim; only the hot banks change
        for layer in range(num_layers):
            index_banks[layer] = {
                role: {
                    "file": loc.file, "offset": loc.offset, "stride": loc.stride,
                    "rows": loc.rows, "row_bytes": loc.row_bytes,
                }
                for role, loc in existing.banks[layer].items()
            }
            written.update(loc.file for loc in existing.banks[layer].values())
        if existing.ple and existing.ple.get("file"):
            written.add(existing.ple["file"])

    for layer in range(num_layers):
        slot = layers[layer]
        if existing is None:
            gu_file = f"layer-{layer:03d}.gate_up.bin"
            gu_size = num_experts * 2 * inter * h_bytes

            def _write_gate_up(fd: int, slot=slot) -> None:
                if "gate_up" in slot:
                    _write_bytes(fd, slot["gate_up"]._raw)
                else:
                    gate = slot["gate"]._raw.reshape(num_experts, inter, h_bytes)
                    up = slot["up"]._raw.reshape(num_experts, inter, h_bytes)
                    for e in range(num_experts):
                        _write_bytes(fd, gate[e])
                        _write_bytes(fd, up[e])

            _write_bank(os.path.join(out_dir, gu_file), gu_size, _write_gate_up)
            written.add(gu_file)

            dn_file = f"layer-{layer:03d}.down.bin"
            dn_size = num_experts * hidden * i_bytes
            _write_bank(os.path.join(out_dir, dn_file), dn_size, lambda fd: _write_bytes(fd, slot["down"]._raw))
            written.add(dn_file)

            index_banks[layer] = {
                "gate_up": {
                    "file": gu_file, "offset": 0, "stride": 2 * inter * h_bytes,
                    "rows": 2 * inter, "row_bytes": h_bytes,
                },
                "down": {
                    "file": dn_file, "offset": 0, "stride": hidden * i_bytes,
                    "rows": hidden, "row_bytes": i_bytes,
                },
            }

        if hot_ids_per_layer is not None and (ids := hot_ids_per_layer[layer]):
            # the hot banks duplicate the top experts' bytes, contiguously, so the pin
            # build reads them sequentially instead of one scattered pread per expert
            gu_hot = f"layer-{layer:03d}.gate_up.hot.bin"
            dn_hot = f"layer-{layer:03d}.down.hot.bin"

            def _write_hot_gate_up(fd: int, slot=slot, ids=ids) -> None:
                if "gate_up" in slot:
                    rows = slot["gate_up"]._raw.reshape(num_experts, 2 * inter, h_bytes)
                    for e in ids:
                        _write_bytes(fd, rows[e])
                else:
                    gate = slot["gate"]._raw.reshape(num_experts, inter, h_bytes)
                    up = slot["up"]._raw.reshape(num_experts, inter, h_bytes)
                    for e in ids:
                        _write_bytes(fd, gate[e])
                        _write_bytes(fd, up[e])

            def _write_hot_down(fd: int, slot=slot, ids=ids) -> None:
                rows = slot["down"]._raw.reshape(num_experts, hidden, i_bytes)
                for e in ids:
                    _write_bytes(fd, rows[e])

            _write_bank(os.path.join(out_dir, gu_hot), len(ids) * 2 * inter * h_bytes, _write_hot_gate_up)
            _write_bank(os.path.join(out_dir, dn_hot), len(ids) * hidden * i_bytes, _write_hot_down)
            written.update((gu_hot, dn_hot))
            for role, hot in (("gate_up", gu_hot), ("down", dn_hot)):
                index_banks[layer][role]["hot_file"] = hot
                index_banks[layer][role]["hot_ids"] = list(ids)
        say(f"expert store: layer {layer + 1}/{num_layers}")

    ple_entry = None
    if existing is not None:
        ple_entry = existing.ple
    elif ple_tensor is not None:
        if drop_ple:
            say("expert store: dropping per-layer token embedding (--drop-ple)")
        else:
            raw = ple_tensor._raw
            size = int(raw.nbytes)
            _write_bank(os.path.join(out_dir, "ple.bin"), size, lambda fd: _write_bytes(fd, raw))
            written.add("ple.bin")
            ple_entry = {
                "file": "ple.bin", "offset": 0, "nbytes": size,
                "ggml_type": int(ple_tensor.ggml_type), "shape": list(ple_tensor.shape),
            }

    doc = {
        "format": STORE_FORMAT,
        "version": STORE_VERSION,
        "align": ALIGN,
        "quant_format": quant_format,
        "num_layers": num_layers,
        "num_experts": num_experts,
        "hidden_size": hidden,
        "intermediate_size": inter,
        "roles": ["gate_up", "down"],
        "gate_up_type": int(gate_up_type),
        "down_type": int(down_type),
        "fingerprint": fingerprint,
        "source_path": os.path.abspath(model_path),
        "layers": [{"layer": layer, "banks": index_banks[layer]} for layer in range(num_layers)],
    }
    if ple_entry is not None:
        doc["ple"] = ple_entry
    tmp = os.path.join(out_dir, INDEX_NAME + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, os.path.join(out_dir, INDEX_NAME))
    # drop banks left over from an earlier, differently-shaped store so the index is the
    # single source of truth
    for name in os.listdir(out_dir):
        if not (name.endswith(".bin") or name.endswith(".bin.tmp")):
            continue
        if name.endswith(".bin") and name in written:
            continue
        try:
            os.unlink(os.path.join(out_dir, name))
        except OSError:
            pass
    say(
        f"expert store: wrote {out_dir} "
        f"({num_layers} layers x {num_experts} experts, {quant_format}"
        + (f", hot banks {len(next(iter(hot_ids_per_layer or [])))}/layer" if hot_ids_per_layer else "")
        + ")"
    )
    return ExpertStoreIndex.load(out_dir)
