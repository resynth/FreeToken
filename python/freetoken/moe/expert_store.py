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


def repack_gguf_experts(
    model_path: str,
    out_dir: str,
    *,
    drop_ple: bool = False,
    progress: Callable[[str], None] | None = None,
) -> ExpertStoreIndex:
    """Repack a GGUF's routed experts into ``out_dir`` verbatim (no dequant).

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

    index_banks: dict[int, dict[str, dict]] = {}
    written: set[str] = set()
    for layer in range(num_layers):
        slot = layers[layer]
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
        say(f"expert store: layer {layer + 1}/{num_layers}")

    ple_entry = None
    if ple_tensor is not None:
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
        "fingerprint": _fingerprint(table),
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
        f"({num_layers} layers x {num_experts} experts, {quant_format})"
    )
    return ExpertStoreIndex.load(out_dir)
