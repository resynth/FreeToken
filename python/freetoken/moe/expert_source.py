"""Expert-source abstraction for the offload MoE cache.

Two implementations sit behind one small protocol:

* :class:`PinnedExpertSource` -- today's behaviour: anonymous pinned ``HostBanks``, one
  ``[E, rows, row_bytes]`` tensor per ``(layer, role)``, directly DMA-able by the GPU.
* :class:`MmapExpertSource` -- a file-backed ``mmap`` of a repacked expert store
  (:mod:`freetoken.moe.expert_store`); the OS page cache is the warm tier and the SSD the
  cold tier. ``layer_views`` returns mmap-backed ``uint8`` tensors, so the cache's staged
  H2D path moves rows through a pinned ring instead of reading a device alias.

Both expose :meth:`warm_row` (single row) and :meth:`read_rows_into` (bulk copy into a
pinned destination). The staged H2D copy uses ``read_rows_into``: for the mmap source a
whole expert is read with one buffered ``preadv`` (large sequential I/O through the page
cache) instead of a per-page fault on the mapping, which is ~20x faster cold.
"""

from __future__ import annotations

import mmap
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
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


def _preadv_full(fd: int, dst: memoryview, offset: int) -> int:
    """``preadv`` ``len(dst)`` bytes at ``offset``; buffered reads can still return short."""
    done = 0
    need = len(dst)
    while done < need:
        got = os.preadv(fd, [dst[done:]], offset + done)
        if got <= 0:
            raise OSError(f"expert store short read: {done} of {need} bytes at {offset}")
        done += got
    return done


class ExpertSource(Protocol):
    """Read-only per-``(layer, role)`` expert row source."""

    roles: tuple[str, ...]
    resident: bool  # True when rows are pinned (GPU-addressable); False for mmap/pageable

    def layer_views(self, layer: int) -> dict[str, torch.Tensor]: ...

    def warm_row(self, layer: int, role: str, expert: int) -> torch.Tensor: ...

    def read_rows_into(
        self, layer: int, role: str, experts: Sequence[int], dst: torch.Tensor
    ) -> int:
        """Copy ``experts[i]``'s row into ``dst[i]``; return the count served from pins."""
        ...

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

    def read_rows_into(
        self, layer: int, role: str, experts: Sequence[int], dst: torch.Tensor
    ) -> int:
        rows = self.sources[role][layer]
        ids = [int(e) for e in experts]
        for i, expert in enumerate(ids):
            dst[i].copy_(rows[expert])
        return len(ids)

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
    fault. ``None`` (or an empty plan) leaves every row on the page cache. ``warm=True``
    sequentially reads the whole store once first, so the page cache (the warm tier) is
    populated before the first token rather than faulted in one 4 KiB page at a time.
    """

    resident = False

    def __init__(
        self,
        store_dir: str,
        index: ExpertStoreIndex,
        *,
        pin_plan: dict[int, Sequence[int]] | None = None,
        warm: bool = False,
    ):
        self.store_dir = store_dir
        self.index = index
        self.roles = tuple(index.roles)
        self._maps: dict[tuple[int, str], mmap.mmap] = {}
        self._views: dict[tuple[int, str], torch.Tensor] = {}
        self._fds: dict[tuple[int, str], int] = {}
        self._pins: dict[tuple[int, str], _PinnedRows] = {}
        self._closed = False
        self._expert_bytes = index.expert_bytes()
        self.pinned_bytes = 0
        if warm:
            # Warm before pinning so the pinned subset is copied from resident pages
            # (a 51 GiB subset faulting one page at a time is the same stall as decode).
            self.warm_cache()
        self._build_pins(pin_plan)

    @classmethod
    def open(
        cls,
        store_dir: str,
        *,
        pin_plan: dict[int, Sequence[int]] | None = None,
        warm: bool = False,
    ) -> "MmapExpertSource":
        return cls(store_dir, ExpertStoreIndex.load(store_dir), pin_plan=pin_plan, warm=warm)

    @property
    def expert_bytes(self) -> int:
        return self._expert_bytes

    def _path(self, layer: int, role: str) -> str:
        return self.index.resolve_file(self.store_dir, self.index.location(layer, role))

    def _fd(self, layer: int, role: str) -> int:
        """A cached read-only fd per bank, for buffered ``preadv`` (populates the page cache)."""
        key = (layer, role)
        fd = self._fds.get(key)
        if fd is None:
            fd = os.open(self._path(layer, role), os.O_RDONLY)
            self._fds[key] = fd
        return fd

    def _mmap(self, layer: int, role: str) -> mmap.mmap:
        key = (layer, role)
        mm = self._maps.get(key)
        if mm is not None:
            return mm
        loc = self.index.location(layer, role)
        path = self._path(layer, role)
        need = loc.offset + self.index.num_experts * loc.stride
        size = os.path.getsize(path)
        if size < need:
            raise ValueError(
                f"{path}: expert store file is {size} bytes, need {need} "
                f"(layer {layer}, role {role!r})"
            )
        mm = mmap.mmap(self._fd(layer, role), 0, prot=mmap.PROT_READ)
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

    def read_rows_into(
        self, layer: int, role: str, experts: Sequence[int], dst: torch.Tensor
    ) -> int:
        """Bulk-copy ``experts[i]``'s row from the store into ``dst[i]``.

        Pinned experts are copied from their warm buffer; the rest come from one buffered
        ``preadv`` per whole expert. That issues stride-sized sequential I/O and populates
        the page cache, instead of faulting the mapping one 4 KiB page at a time under
        ``MADV_RANDOM``. Returns the number of rows served from the pinned subset.
        """
        if self._closed:
            raise RuntimeError("expert source is closed")
        ids = [int(e) for e in experts]
        if dst.shape[0] != len(ids):
            raise ValueError(f"dst holds {dst.shape[0]} rows but {len(ids)} experts were requested")
        if not dst.is_contiguous():
            raise ValueError("read_rows_into destination must be contiguous")
        loc = self.index.location(layer, role)
        if tuple(dst.shape[1:]) != (loc.rows, loc.row_bytes):
            raise ValueError(
                f"dst rows are {tuple(dst.shape[1:])} but the store's are "
                f"{(loc.rows, loc.row_bytes)}"
            )
        row_nbytes = loc.rows * loc.row_bytes
        pin = self._pins.get((layer, role))
        fd = -1
        mv = None
        pinned = 0
        for i, expert in enumerate(ids):
            slot = pin.slot_of.get(expert) if pin is not None else None
            if slot is not None:
                dst[i].copy_(pin.tensor[slot])
                pinned += 1
                continue
            if fd < 0:
                fd = self._fd(layer, role)
                flat = dst.reshape(-1)
                if flat.dtype != torch.uint8:
                    flat = flat.view(torch.uint8)
                mv = memoryview(flat.numpy())
            _preadv_full(fd, mv[i * row_nbytes : (i + 1) * row_nbytes], loc.expert_offset(expert))
        return pinned

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
                pinned = None
                if alloc_pinned_tensor is not None and torch.cuda.is_available():
                    try:
                        pinned = alloc_pinned_tensor(len(ids), loc.rows, loc.row_bytes, dtype=torch.uint8)
                    except Exception as exc:  # noqa: BLE001 - no pin quota -> page cache only
                        logger.warning(f"expert store: could not pin warm subset ({exc}); leaving it pageable")
                if pinned is None:
                    pinned = torch.empty((len(ids), loc.rows, loc.row_bytes), dtype=torch.uint8)
                # large buffered reads, not per-page faults on the mapping
                self.read_rows_into(layer, role, ids, pinned)
                rows = _PinnedRows(tensor=pinned)
                for slot, expert in enumerate(ids):
                    rows.slot_of[expert] = slot
                self._pins[(layer, role)] = rows
                self.pinned_bytes += int(pinned.numel())

    def _bank_ranges(self) -> list[tuple[str, int, int]]:
        """Unique ``(path, offset, nbytes)`` bank ranges this source reads from."""
        ranges: dict[tuple[str, int, int], None] = {}
        for layer in range(self.index.num_layers):
            for role in self.roles:
                loc = self.index.location(layer, role)
                key = (self._path(layer, role), loc.offset, self.index.num_experts * loc.stride)
                ranges[key] = None
        return list(ranges)

    def warm_cache(self, *, workers: int = 4, chunk: int = 8 << 20) -> int:
        """Sequentially read every bank range once so the page cache is warm before serving.

        Bounded multi-threaded buffered reads (deliberately not O_DIRECT: the point is to
        leave the pages resident). Returns the bytes read.
        """
        ranges = self._bank_ranges()
        if not ranges:
            return 0
        total = sum(length for _, _, length in ranges)

        def read(path: str, offset: int, length: int) -> int:
            fd = os.open(path, os.O_RDONLY)
            try:
                mv = memoryview(bytearray(min(chunk, length) or 1))
                done = 0
                while done < length:
                    want = min(len(mv), length - done)
                    done += _preadv_full(fd, mv[:want], offset + done)
                return done
            finally:
                os.close(fd)

        started = time.monotonic()
        logger.info_rank0(
            f"expert store: warming page cache ({total / 2**30:.1f} GiB, {len(ranges)} files)"
        )
        if workers > 1 and len(ranges) > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                read_bytes = sum(pool.map(lambda spec: read(*spec), ranges))
        else:
            read_bytes = sum(read(*spec) for spec in ranges)
        elapsed = max(1e-6, time.monotonic() - started)
        logger.info_rank0(
            f"expert store: page cache warm ({read_bytes / 2**30:.1f} GiB in {elapsed:.1f}s, "
            f"{read_bytes / 2**20 / elapsed:.0f} MiB/s)"
        )
        return read_bytes

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
        for fd in self._fds.values():
            try:
                os.close(fd)
            except OSError:
                pass
        self._fds.clear()
