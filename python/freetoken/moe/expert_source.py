"""Expert-source abstraction for the offload MoE cache.

Two implementations sit behind one small protocol:

* :class:`PinnedExpertSource` -- today's behaviour: anonymous pinned ``HostBanks``, one
  ``[E, rows, row_bytes]`` tensor per ``(layer, role)``, directly DMA-able by the GPU.
* :class:`MmapExpertSource` -- a file-backed ``mmap`` of a repacked expert store
  (:mod:`freetoken.moe.expert_store`); the OS page cache is the warm tier and the SSD the
  cold tier. ``layer_views`` returns mmap-backed ``uint8`` tensors, so the cache's staged
  H2D path moves rows through a pinned ring instead of reading a device alias.

Both expose :meth:`warm_row`, used by the staged copy path: it returns the pinned subset
row when the expert is resident in the RAM-pinned warm set, else the mmap/pageable row.
"""

from __future__ import annotations

import mmap
import os
import warnings
from dataclasses import dataclass, field
from typing import Protocol, Sequence

import torch

from freetoken.moe.expert_store import ExpertStoreIndex
from freetoken.utils import init_logger

logger = init_logger(__name__)


def _page_aligned_range(offset: int, length: int, size: int) -> tuple[int, int]:
    """Page-aligned ``[lo, hi)`` covering ``[offset, offset+length)``, clamped to ``size``.

    ``madvise(2)`` rejects a non-page-aligned address, so callers must round out.
    """
    page = mmap.PAGESIZE
    lo = offset - (offset % page)
    hi = min((offset + length + page - 1) // page * page, size)
    return lo, max(lo, hi)


class ExpertSource(Protocol):
    """Read-only per-``(layer, role)`` expert row source."""

    roles: tuple[str, ...]
    resident: bool  # True when rows are pinned (GPU-addressable); False for mmap/pageable

    def layer_views(self, layer: int) -> dict[str, torch.Tensor]: ...

    def warm_row(self, layer: int, role: str, expert: int) -> torch.Tensor: ...

    def prefetch(self, layer: int, experts: Sequence[int]) -> None: ...

    def close(self) -> None: ...

    @property
    def expert_bytes(self) -> int: ...


@dataclass
class PinnedExpertSource:
    """Wraps already-materialized pinned ``{role: [per-layer [E, rows, row_bytes]]}`` banks."""

    sources: dict[str, list[torch.Tensor]]
    roles: tuple[str, ...] = ()
    resident: bool = True
    _expert_bytes: int = 0

    def __post_init__(self) -> None:
        if not self.roles:
            self.roles = tuple(self.sources)
        self._expert_bytes = sum(
            int(t[0].numel() * t[0].element_size()) for t in self.sources.values()
        )

    @property
    def expert_bytes(self) -> int:
        return self._expert_bytes

    def layer_views(self, layer: int) -> dict[str, torch.Tensor]:
        return {role: per[layer] for role, per in self.sources.items()}

    def warm_row(self, layer: int, role: str, expert: int) -> torch.Tensor:
        return self.sources[role][layer][expert]

    def prefetch(self, layer: int, experts: Sequence[int]) -> None:
        return

    def close(self) -> None:
        return


@dataclass
class _PinnedRows:
    """A layer/role's pinned warm subset: ``tensor[k]`` holds expert ``experts[k]``."""

    tensor: torch.Tensor
    slot_of: dict[int, int] = field(default_factory=dict)

    def has(self, expert: int) -> bool:
        return expert in self.slot_of


class MmapExpertSource:
    """File-backed mmap expert source with an optional pinned warm subset.

    ``pin_plan`` maps ``layer -> expert ids`` to keep in pinned host RAM; those rows are
    copied out of the mmap once at open, so a warm hit is a plain CPU read with no page
    fault. ``None`` (or an empty plan) leaves every row on the page cache.
    """

    resident = False

    def __init__(self, store_dir: str, index: ExpertStoreIndex, *, pin_plan: dict[int, Sequence[int]] | None = None):
        self.store_dir = store_dir
        self.index = index
        self.roles = tuple(index.roles)
        self._maps: dict[tuple[int, str], mmap.mmap] = {}
        self._views: dict[tuple[int, str], torch.Tensor] = {}
        self._pins: dict[tuple[int, str], _PinnedRows] = {}
        self._closed = False
        self._expert_bytes = index.expert_bytes()
        self.pinned_bytes = 0
        self._build_pins(pin_plan)

    @classmethod
    def open(cls, store_dir: str, *, pin_plan: dict[int, Sequence[int]] | None = None) -> "MmapExpertSource":
        return cls(store_dir, ExpertStoreIndex.load(store_dir), pin_plan=pin_plan)

    @property
    def expert_bytes(self) -> int:
        return self._expert_bytes

    def _mmap(self, layer: int, role: str) -> mmap.mmap:
        key = (layer, role)
        mm = self._maps.get(key)
        if mm is not None:
            return mm
        loc = self.index.location(layer, role)
        path = self.index.resolve_file(self.store_dir, loc)
        need = loc.offset + self.index.num_experts * loc.stride
        size = os.path.getsize(path)
        if size < need:
            raise ValueError(
                f"{path}: expert store file is {size} bytes, need {need} "
                f"(layer {layer}, role {role!r})"
            )
        fd = os.open(path, os.O_RDONLY)
        try:
            mm = mmap.mmap(fd, 0, prot=mmap.PROT_READ)
        finally:
            os.close(fd)
        if hasattr(mm, "madvise"):
            # access within a layer is random; default readahead would pull adjacent
            # experts into the page cache that no routing asked for
            mm.madvise(mmap.MADV_RANDOM)
        self._maps[key] = mm
        return mm

    def _view(self, layer: int, role: str) -> torch.Tensor:
        key = (layer, role)
        t = self._views.get(key)
        if t is not None:
            return t
        mm = self._mmap(layer, role)
        loc = self.index.location(layer, role)
        nbytes = self.index.num_experts * loc.stride
        with warnings.catch_warnings():
            # torch.frombuffer warns that a read-only buffer yields a (de facto
            # unwritable) tensor; the source is read-only by contract, so silence it
            warnings.simplefilter("ignore", UserWarning)
            t = torch.frombuffer(mm, dtype=torch.uint8, count=loc.offset + nbytes)
        t = t[loc.offset :].view(self.index.num_experts, loc.rows, loc.row_bytes)
        self._views[key] = t
        return t

    def layer_views(self, layer: int) -> dict[str, torch.Tensor]:
        return {role: self._view(layer, role) for role in self.roles}

    # Back-compat alias used by ExpertBanks construction.
    def all_layer_views(self) -> dict[str, list[torch.Tensor]]:
        return {
            role: [self._view(layer, role) for layer in range(self.index.num_layers)]
            for role in self.roles
        }

    def warm_row(self, layer: int, role: str, expert: int) -> torch.Tensor:
        pin = self._pins.get((layer, role))
        if pin is not None:
            slot = pin.slot_of.get(expert)
            if slot is not None:
                return pin.tensor[slot]
        return self._view(layer, role)[expert]

    def is_pinned_row(self, layer: int, role: str, expert: int) -> bool:
        pin = self._pins.get((layer, role))
        return pin is not None and pin.has(expert)

    def prefetch(self, layer: int, experts: Sequence[int]) -> None:
        """``MADV_WILLNEED`` the byte ranges of ``experts`` (merged runs)."""
        if not experts or self._closed:
            return
        ordered = sorted({int(e) for e in experts if 0 <= int(e) < self.index.num_experts})
        for role in self.roles:
            loc = self.index.location(layer, role)
            mm = self._mmap(layer, role)
            if not hasattr(mm, "madvise"):
                return
            run_start = ordered[0]
            prev = ordered[0]
            for e in ordered[1:]:
                if e != prev + 1:
                    self._willneed(mm, loc, run_start, prev)
                    run_start = e
                prev = e
            self._willneed(mm, loc, run_start, prev)

    @staticmethod
    def _willneed(mm: mmap.mmap, loc, start: int, end: int) -> None:
        # madvise(2) requires a page-aligned address, and expert strides are essentially
        # never page multiples, so round the range out or the call is a silent EINVAL
        off = loc.expert_offset(start)
        length = loc.expert_offset(end) + loc.stride - off
        lo, hi = _page_aligned_range(off, length, len(mm))
        if hi <= lo:
            return
        try:
            mm.madvise(mmap.MADV_WILLNEED, lo, hi - lo)
        except (OSError, ValueError):
            pass

    def _build_pins(self, pin_plan: dict[int, Sequence[int]] | None) -> None:
        if not pin_plan:
            return
        try:
            from freetoken.kernel.pinned import alloc_pinned_tensor
        except Exception:  # noqa: BLE001 - CPU-only tooling has no pinned allocator
            alloc_pinned_tensor = None

        for layer, experts in pin_plan.items():
            ids = [int(e) for e in experts]
            if not ids:
                continue
            for role in self.roles:
                loc = self.index.location(layer, role)
                view = self._view(layer, role)
                pinned = None
                if alloc_pinned_tensor is not None and torch.cuda.is_available():
                    try:
                        pinned = alloc_pinned_tensor(len(ids), loc.rows, loc.row_bytes, dtype=torch.uint8)
                    except Exception as exc:  # noqa: BLE001 - no pin quota -> page cache only
                        logger.warning(f"expert store: could not pin warm subset ({exc}); leaving it pageable")
                if pinned is None:
                    pinned = torch.empty((len(ids), loc.rows, loc.row_bytes), dtype=torch.uint8)
                rows = _PinnedRows(tensor=pinned)
                for slot, expert in enumerate(ids):
                    pinned[slot].copy_(view[expert])
                    rows.slot_of[expert] = slot
                self._pins[(layer, role)] = rows
                self.pinned_bytes += int(pinned.numel())

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._views.clear()
        self._pins.clear()
        for mm in self._maps.values():
            try:
                mm.close()
            except BufferError:
                pass
        self._maps.clear()
