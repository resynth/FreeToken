"""Repacked expert store + mmap expert source (moe.expert_store / moe.expert_source)."""

from __future__ import annotations

import json
import mmap
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from freetoken.models.gguf.dequant import GGML_IQ4_NL, row_bytes
from freetoken.moe.expert_store import (
    ExpertStoreIndex,
    default_store_dir,
    is_expert_store,
    repack_gguf_experts,
)


@dataclass
class _FakeTensor:
    name: str
    ggml_type: int
    shape: tuple[int, ...]
    _raw: np.ndarray


def _bytes(rows: int, row_bytes: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, size=(rows, row_bytes), dtype=np.uint8)


def _fused_tensors(layers: int, E: int, H: int, I: int, ggml_type: int = GGML_IQ4_NL):
    out = []
    for layer in range(layers):
        out.append(_FakeTensor(
            f"blk.{layer}.ffn_gate_up_exps.weight", ggml_type, (E, 2 * I, H),
            _bytes(E * 2 * I, row_bytes(H, ggml_type), 10 + layer),
        ))
        out.append(_FakeTensor(
            f"blk.{layer}.ffn_down_exps.weight", ggml_type, (E, H, I),
            _bytes(E * H, row_bytes(I, ggml_type), 100 + layer),
        ))
    return out


def _patch(monkeypatch, tensors):
    from freetoken.models.gguf import reader

    monkeypatch.setattr(reader, "iter_gguf_tensors", lambda path: iter(tensors))


def _repack(monkeypatch, tensors, tmp_path, **kwargs):
    _patch(monkeypatch, tensors)
    return repack_gguf_experts("model.gguf", str(tmp_path), **kwargs)


def test_repack_and_mmap_round_trip(monkeypatch, tmp_path):
    from freetoken.moe.expert_source import MmapExpertSource

    E, H, I, L = 2, 64, 32, 3
    tensors = _fused_tensors(L, E, H, I)
    index = _repack(monkeypatch, tensors, tmp_path)
    assert (index.num_layers, index.num_experts, index.hidden_size, index.intermediate_size) == (L, E, H, I)
    assert index.quant_format == "iq4_nl"
    assert index.fingerprint

    source = MmapExpertSource.open(str(tmp_path))
    for layer in range(L):
        for role, suffix in (("gate_up", "ffn_gate_up_exps.weight"), ("down", "ffn_down_exps.weight")):
            loc = index.location(layer, role)
            t = next(t for t in tensors if t.name == f"blk.{layer}.{suffix}")
            want = t._raw.reshape(E, loc.rows, loc.row_bytes)
            assert np.array_equal(source.layer_views(layer)[role].numpy(), want)
            for expert in range(E):
                assert np.array_equal(source.warm_row(layer, role, expert).numpy(), want[expert])
                assert bytes(source._mmap(layer, role)[loc.expert_offset(expert):loc.expert_offset(expert) + loc.stride]) == want[expert].tobytes()
    source.close()


def test_repack_separate_gate_up_fuses_gate_then_up(monkeypatch, tmp_path):
    from freetoken.moe.expert_source import MmapExpertSource

    E, H, I = 2, 64, 32
    rb = row_bytes(H, GGML_IQ4_NL)
    gate = _bytes(E * I, rb, 1)
    up = _bytes(E * I, rb, 2)
    tensors = [
        _FakeTensor("blk.0.ffn_gate_exps.weight", GGML_IQ4_NL, (E, I, H), gate),
        _FakeTensor("blk.0.ffn_up_exps.weight", GGML_IQ4_NL, (E, I, H), up),
        _FakeTensor("blk.0.ffn_down_exps.weight", GGML_IQ4_NL, (E, H, I), _bytes(E * H, row_bytes(I, GGML_IQ4_NL), 3)),
    ]
    index = _repack(monkeypatch, tensors, tmp_path)
    source = MmapExpertSource.open(str(tmp_path))
    view = source.layer_views(0)["gate_up"].reshape(E, 2, I, rb)
    assert np.array_equal(view[:, 0].numpy(), gate.reshape(E, I, rb))
    assert np.array_equal(view[:, 1].numpy(), up.reshape(E, I, rb))


def test_store_rejects_a_foreign_index(tmp_path):
    (tmp_path / "index.json").write_text('{"format": "nope", "version": 1}')
    with pytest.raises(ValueError, match="not an expert store"):
        ExpertStoreIndex.load(str(tmp_path))


def test_store_rejects_a_bank_file_outside_the_store(tmp_path):
    doc = {
        "format": "freetoken_experts",
        "version": 1,
        "quant_format": "iq4_nl",
        "num_layers": 1,
        "num_experts": 2,
        "hidden_size": 64,
        "intermediate_size": 32,
        "roles": ["gate_up", "down"],
        "layers": [
            {"layer": 0, "banks": {
                "gate_up": {"file": "../evil.bin", "offset": 0, "stride": 100, "rows": 2, "row_bytes": 50},
                "down": {"file": "x.bin", "offset": 0, "stride": 100, "rows": 2, "row_bytes": 50},
            }},
        ],
    }
    (tmp_path / "index.json").write_text(json.dumps(doc))
    index = ExpertStoreIndex.load(str(tmp_path))
    with pytest.raises(ValueError, match="escapes the store dir"):
        index.resolve_file(str(tmp_path), index.location(0, "gate_up"))


def test_repack_removes_stale_bank_files(monkeypatch, tmp_path):
    _repack(monkeypatch, _fused_tensors(1, 2, 64, 32), tmp_path)
    stale = tmp_path / "layer-999.gate_up.bin"
    stale.write_bytes(b"old")
    stray_tmp = tmp_path / "layer-000.down.bin.tmp"
    stray_tmp.write_bytes(b"old")
    _repack(monkeypatch, _fused_tensors(1, 2, 64, 32), tmp_path)
    assert not stale.exists()
    assert not stray_tmp.exists()


def test_repack_rejects_mismatched_tensor_shape(monkeypatch, tmp_path):
    E, H, I = 2, 64, 32
    tensors = _fused_tensors(1, E, H, I)
    down = next(t for t in tensors if t.name.endswith("ffn_down_exps.weight"))
    down._raw = np.zeros((E * H, row_bytes(I, GGML_IQ4_NL) + 1), dtype=np.uint8)
    with pytest.raises(ValueError, match="inconsistent"):
        _repack(monkeypatch, tensors, tmp_path)


def test_store_loader_rejects_mismatched_geometry(monkeypatch, tmp_path):
    from freetoken.moe.expert_banks import _store_expert_banks

    _repack(monkeypatch, _fused_tensors(1, 2, 64, 32), tmp_path)
    cfg = SimpleNamespace(
        num_moe_layers=1, num_experts=2, expert_quant="iq4_nl",
        hidden_size=128, moe_intermediate_size=32,
    )
    with pytest.raises(ValueError, match="hidden/intermediate"):
        _store_expert_banks(
            "m.gguf", cfg, store_dir=str(tmp_path), usage_file=None, pin_budget_bytes=0
        )


def test_prefetch_range_is_page_aligned():
    from freetoken.moe.expert_source import _page_aligned_range

    lo, hi = _page_aligned_range(100, 5000, 4096 * 10)
    assert lo % mmap.PAGESIZE == 0 and (hi - lo) % mmap.PAGESIZE == 0
    assert lo <= 100 and hi >= 5100
    # clamped to the mapping size
    lo2, hi2 = _page_aligned_range(4096 * 9, 4096 * 4, 4096 * 10)
    assert lo2 == 4096 * 9 and hi2 == 4096 * 10


def test_prefetch_skips_pinned_rows(monkeypatch, tmp_path):
    from freetoken.moe.expert_source import MmapExpertSource

    E, H, I = 4, 64, 32
    _repack(monkeypatch, _fused_tensors(1, E, H, I), tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    source = MmapExpertSource.open(str(tmp_path), pin_plan={0: [1]})
    seen = []
    monkeypatch.setattr(
        MmapExpertSource,
        "_willneed",
        staticmethod(lambda mm, loc, start, end: seen.append((start, end))),
    )
    source.prefetch(0, [0, 1, 2])
    # the whole-layer prefill copy reads the mmap views, so the pinned expert stays
    # in: one merged run per role
    assert seen == [(0, 2), (0, 2)]
    seen.clear()
    source.prefetch(0, [0, 1, 2], include_pinned=False)
    # decode paths serve the pinned row from RAM: its range is skipped, and the
    # remaining experts break into one run per gap, per role
    assert seen == [(0, 0), (2, 2), (0, 0), (2, 2)]
    source.close()


def test_default_store_dir_maps_a_gguf_file(tmp_path):
    model = tmp_path / "m.gguf"
    model.write_bytes(b"")
    assert default_store_dir(str(model)) == str(model) + ".experts"
    assert is_expert_store(str(model) + ".experts") is False


def test_pin_subset_uses_the_usage_ranking(monkeypatch, tmp_path):
    from freetoken.moe.expert_source import MmapExpertSource

    E, H, I = 2, 64, 32
    tensors = _fused_tensors(1, E, H, I)
    index = _repack(monkeypatch, tensors, tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    # pin only expert 1; warm_row(1) must come from the pinned copy, warm_row(0) from mmap
    source = MmapExpertSource.open(str(tmp_path), pin_plan={0: [1]})
    assert source.is_pinned_row(0, "gate_up", 1)
    assert not source.is_pinned_row(0, "gate_up", 0)
    for role in ("gate_up", "down"):
        loc = index.location(0, role)
        assert torch.equal(source.warm_row(0, role, 1), source._view(0, role)[1])
        assert source.warm_row(0, role, 0).data_ptr() == source._view(0, role)[0].data_ptr()
    assert source.pinned_bytes > 0


def test_prefill_overlap_madvises_staged_layers(monkeypatch):
    """The overlap prefill copy pageable-reads the mmap views; WILLNEED the bank first."""
    from freetoken.distributed import set_tp_info, try_get_tp_info

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    from freetoken.moe.offload_cache import OffloadMoeCache

    class _Source:
        resident = False

        def __init__(self):
            self.calls = []

        def prefetch(self, layer, experts):
            self.calls.append((layer, list(experts)))

    gu = torch.arange(4 * 3 * 5, dtype=torch.float32).reshape(4, 3, 5)
    dn = torch.arange(4 * 2 * 7, dtype=torch.float32).reshape(4, 2, 7) + 1000
    source = _Source()
    cache = OffloadMoeCache(
        num_layers=1, num_experts=4, cache_size=8, device=torch.device("cpu"),
        quant_format="bf16", prefill_overlap=True,
    )
    cache.set_bank_sources(
        {"gate_up": [gu], "down": [dn]}, layer_residency=["mmap"], expert_source=source
    )
    cache.prefetch_prefill_layer(0)
    assert source.calls == [(0, list(range(4)))]
    assert torch.equal(cache.prefill_bank_buffers[0][0], gu)


def test_mmap_residency_stages_decode_and_allows_prefill_overlap(monkeypatch):
    from freetoken.distributed import set_tp_info, try_get_tp_info

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    from freetoken.moe.offload_cache import OffloadMoeCache

    gu = torch.arange(4 * 3 * 5, dtype=torch.float32).reshape(4, 3, 5)
    dn = torch.arange(4 * 2 * 7, dtype=torch.float32).reshape(4, 2, 7) + 1000
    cache = OffloadMoeCache(
        num_layers=1, num_experts=4, cache_size=8, device=torch.device("cpu"),
        quant_format="bf16", prefill_overlap=True,
    )
    cache.set_bank_sources({"gate_up": [gu], "down": [dn]}, layer_residency=["mmap"])
    assert cache.is_staged_layer(0)
    assert not cache.is_unpinned_layer(0)

    cache.num_indices.fill_(2)
    cache.evict_slots[:2] = torch.tensor([5, 4], dtype=torch.int32)
    cache.src_indices[:2] = torch.tensor([3, 1], dtype=torch.int32)
    cache._pending_src_layer = 0
    cache._pending_whole_layer = False
    cache.copy_missing()
    assert torch.equal(cache.bank_caches["gate_up"][5], gu[3])
    assert torch.equal(cache.bank_caches["down"][4], dn[1])

    cache._pending_whole_layer = True
    cache.copy_missing()
    assert torch.equal(cache.bank_caches["gate_up"][:4], gu)
    assert torch.equal(cache.bank_caches["down"][:4], dn)


def test_staged_copy_prefetches_the_current_layer_misses():
    """I: the staged copy knows the misses before the fill, so it WILLNEEDs them first."""
    from freetoken.distributed import set_tp_info, try_get_tp_info

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    from freetoken.moe.offload_cache import OffloadMoeCache

    class _Source:
        def __init__(self, resident, views):
            self.resident = resident
            self.views = views
            self.prefetches = []

        def warm_row(self, layer, role, expert):
            return self.views[role][layer][expert]

        def read_rows_into(self, layer, role, experts, dst):
            for i, e in enumerate(experts):
                dst[i].copy_(self.views[role][layer][int(e)])
            return 0

        def prefetch(self, layer, experts, *, include_pinned=True):
            self.prefetches.append((layer, list(experts), include_pinned))

    gu = torch.arange(4 * 3 * 5, dtype=torch.float32).reshape(4, 3, 5)
    dn = torch.arange(4 * 2 * 7, dtype=torch.float32).reshape(4, 2, 7) + 1000
    views = {"gate_up": [gu], "down": [dn]}

    def _misses(source):
        cache = OffloadMoeCache(
            num_layers=1, num_experts=4, cache_size=8, device=torch.device("cpu"),
            quant_format="bf16", prefill_overlap=True,
        )
        cache.set_bank_sources(views, layer_residency=["mmap"], expert_source=source)
        cache.num_indices.fill_(2)
        cache.evict_slots[:2] = torch.tensor([5, 4], dtype=torch.int32)
        cache.src_indices[:2] = torch.tensor([3, 1], dtype=torch.int32)
        cache._pending_src_layer = 0
        cache._pending_whole_layer = False
        return cache

    pageable = _Source(False, views)
    _misses(pageable).copy_missing()
    # the misses are WILLNEEDed up front, skipping rows the pin buffer serves
    assert pageable.prefetches == [(0, [3, 1], False)]

    pinned = _Source(True, views)
    _misses(pinned).copy_missing()
    # a resident source never prefetches
    assert pinned.prefetches == []

    off = _Source(False, views)
    cache = _misses(off)
    cache.staged_miss_prefetch = False
    cache.copy_missing()
    assert off.prefetches == []
    assert torch.equal(cache.bank_caches["gate_up"][5], gu[3])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_gpu_staged_copy_matches_the_store(monkeypatch, tmp_path):
    from freetoken.moe.expert_source import MmapExpertSource
    from freetoken.moe.offload_cache import OffloadMoeCache
    from freetoken.distributed import set_tp_info, try_get_tp_info

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    E, H, I = 4, 64, 32
    tensors = _fused_tensors(1, E, H, I)
    _repack(monkeypatch, tensors, tmp_path)
    source = MmapExpertSource.open(str(tmp_path))
    device = torch.device("cuda")
    cache = OffloadMoeCache(
        num_layers=1, num_experts=E, cache_size=8, device=device, quant_format="iq4_nl",
    )
    cache.set_bank_sources(source.all_layer_views(), layer_residency=["mmap"], expert_source=source)
    cache.num_indices.fill_(2)
    cache.evict_slots[:2] = torch.tensor([5, 4], dtype=torch.int32, device=device)
    cache.src_indices[:2] = torch.tensor([3, 1], dtype=torch.int32, device=device)
    cache._pending_src_layer = 0
    cache._pending_whole_layer = False
    cache.copy_missing()
    torch.cuda.synchronize()
    for role in ("gate_up", "down"):
        view = source.layer_views(0)[role]
        assert torch.equal(cache.bank_caches[role][5].cpu(), view[3])
        assert torch.equal(cache.bank_caches[role][4].cpu(), view[1])

    cache._pending_whole_layer = True
    cache.copy_missing()
    torch.cuda.synchronize()
    for role in ("gate_up", "down"):
        assert torch.equal(cache.bank_caches[role][:E].cpu(), source.layer_views(0)[role])


def test_staged_copy_chunks_across_the_ring(monkeypatch, tmp_path):
    """More misses than one ring buffer must span chunks and both double buffers."""
    from freetoken.distributed import set_tp_info, try_get_tp_info

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    from freetoken.moe.expert_source import MmapExpertSource
    from freetoken.moe.offload_cache import OffloadMoeCache

    monkeypatch.setenv("FREETOKEN_EXPERT_RING_ROWS", "2")
    E, H, I = 8, 64, 32
    _repack(monkeypatch, _fused_tensors(1, E, H, I), tmp_path)
    source = MmapExpertSource.open(str(tmp_path))
    cache = OffloadMoeCache(
        num_layers=1, num_experts=E, cache_size=E, device=torch.device("cpu"),
        quant_format="iq4_nl",
    )
    cache.set_bank_sources(source.all_layer_views(), layer_residency=["mmap"], expert_source=source)
    assert cache._staging_rows == 2

    ids = [7, 0, 3, 5, 1]
    slots = [4, 0, 6, 2, 5]
    cache.num_indices.fill_(len(ids))
    cache.evict_slots[: len(ids)] = torch.tensor(slots, dtype=torch.int32)
    cache.src_indices[: len(ids)] = torch.tensor(ids, dtype=torch.int32)
    cache._pending_src_layer = 0
    cache._pending_whole_layer = False
    cache.copy_missing()
    for role in ("gate_up", "down"):
        view = source.layer_views(0)[role]
        for expert, slot in zip(ids, slots):
            assert torch.equal(cache.bank_caches[role][slot], view[expert])
    assert cache.staged_page_layer[0] == 2 * len(ids)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_gpu_staged_copy_spans_chunks_and_buffers(monkeypatch, tmp_path):
    """GPU staged copy across ring chunks: both double buffers + events stay correct."""
    from freetoken.moe.expert_source import MmapExpertSource
    from freetoken.moe.offload_cache import OffloadMoeCache
    from freetoken.distributed import set_tp_info, try_get_tp_info

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    monkeypatch.setenv("FREETOKEN_EXPERT_RING_ROWS", "2")
    E, H, I = 8, 64, 32
    _repack(monkeypatch, _fused_tensors(1, E, H, I), tmp_path)
    source = MmapExpertSource.open(str(tmp_path))
    device = torch.device("cuda")
    cache = OffloadMoeCache(
        num_layers=1, num_experts=E, cache_size=E, device=device, quant_format="iq4_nl",
    )
    cache.set_bank_sources(source.all_layer_views(), layer_residency=["mmap"], expert_source=source)
    assert cache._staging_rows == 2

    ids = [7, 0, 3, 5, 1]
    slots = [4, 0, 6, 2, 5]
    cache.num_indices.fill_(len(ids))
    cache.evict_slots[: len(ids)] = torch.tensor(slots, dtype=torch.int32, device=device)
    cache.src_indices[: len(ids)] = torch.tensor(ids, dtype=torch.int32, device=device)
    cache._pending_src_layer = 0
    cache._pending_whole_layer = False
    cache.copy_missing()
    torch.cuda.synchronize(device)
    for role in ("gate_up", "down"):
        view = source.layer_views(0)[role]
        for expert, slot in zip(ids, slots):
            assert torch.equal(cache.bank_caches[role][slot].cpu(), view[expert])
    assert cache.staged_page_layer[0] == 2 * len(ids)


def test_staged_tier_counters_split_pinned_and_page(monkeypatch, tmp_path):
    from freetoken.distributed import set_tp_info, try_get_tp_info

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    from freetoken.moe.expert_source import MmapExpertSource
    from freetoken.moe.offload_cache import OffloadMoeCache

    E, H, I = 4, 64, 32
    _repack(monkeypatch, _fused_tensors(1, E, H, I), tmp_path)
    source = MmapExpertSource.open(str(tmp_path), pin_plan={0: [3]})
    cache = OffloadMoeCache(
        num_layers=1, num_experts=E, cache_size=8, device=torch.device("cpu"), quant_format="iq4_nl",
    )
    cache.set_bank_sources(source.all_layer_views(), layer_residency=["mmap"], expert_source=source)
    cache.num_indices.fill_(2)
    cache.evict_slots[:2] = torch.tensor([5, 4], dtype=torch.int32)
    cache.src_indices[:2] = torch.tensor([3, 1], dtype=torch.int32)
    cache._pending_src_layer = 0
    cache._pending_whole_layer = False
    cache.copy_missing()
    # one pinned expert (3) and one page-cache expert (1), each across both banks
    assert cache.staged_pinned_layer[0] == 2
    assert cache.staged_page_layer[0] == 2
    assert cache.staged_tier_stats()["per_layer"][0]["pinned_rate"] == 0.5


def test_finish_mmap_source_keeps_graphs_for_cpu_decode():
    """D: pure-CPU decode over mmap is graph-capturable; staged GPU decode is not."""
    from freetoken.engine.engine import _finish_mmap_source

    cfg = SimpleNamespace(
        cuda_graph_max_bs=8, cuda_graph_bs=[1, 2, 4, 8], expert_usage_file=None,
        expert_prefetch=None,
    )
    _finish_mmap_source(cfg, SimpleNamespace(decode_target="cpu"))
    assert cfg.cuda_graph_max_bs == 8
    assert cfg.cuda_graph_bs == [1, 2, 4, 8]

    _finish_mmap_source(cfg, SimpleNamespace(decode_target="gpu"))
    assert cfg.cuda_graph_max_bs == 0
    assert cfg.cuda_graph_bs == []


def test_finish_mmap_source_resolves_the_prefetch_depth(tmp_path, monkeypatch):
    """I/N: --expert-prefetch > FREETOKEN_EXPERT_PREFETCH > 4; 0 disables the plan."""
    from freetoken.engine.engine import _finish_mmap_source
    from freetoken.moe.usage import UsageData

    usage = UsageData(counts=[[3, 0, 2, 1]])  # rank: 0, 2, 3, 1
    path = tmp_path / "usage.json"
    usage.save(str(path))

    class _Cache:
        decode_target = "gpu"
        usage_prefetch = None
        staged_miss_prefetch = True

    def _cfg(prefetch=None):
        return SimpleNamespace(
            cuda_graph_max_bs=0, cuda_graph_bs=[], expert_usage_file=str(path),
            expert_prefetch=prefetch,
        )

    monkeypatch.setenv("FREETOKEN_EXPERT_PREFETCH", "2")
    cache = _Cache()
    _finish_mmap_source(_cfg(), cache)
    assert cache.usage_prefetch == [[0, 2]]
    assert cache.staged_miss_prefetch is True

    cache = _Cache()
    _finish_mmap_source(_cfg(1), cache)
    assert cache.usage_prefetch == [[0]]

    cache = _Cache()
    _finish_mmap_source(_cfg(0), cache)
    assert cache.usage_prefetch is None
    assert cache.staged_miss_prefetch is False

    monkeypatch.delenv("FREETOKEN_EXPERT_PREFETCH")
    cache = _Cache()
    _finish_mmap_source(_cfg(), cache)
    assert cache.usage_prefetch == [[0, 2, 3, 1]]


def test_select_expert_source_auto_prefers_mmap_over_budget(monkeypatch, tmp_path):
    from freetoken.engine.engine import _select_expert_source

    store = tmp_path / "store"
    store.mkdir()
    (store / "index.json").write_text('{"format": "freetoken_experts", "version": 1}')
    model = tmp_path / "x.gguf"
    model.write_bytes(b"")
    cfg = SimpleNamespace(
        expert_source="auto", expert_store=str(store), model_path=str(model),
        moe_cpu_layers=None, moe_strategy="offload",
        model_config=SimpleNamespace(
            num_moe_layers=1, num_experts=2, expert_quant="q4_0",
            moe_weight_format=None, hidden_size=64, moe_intermediate_size=32,
        ),
    )
    monkeypatch.setenv("FREETOKEN_PIN_BUDGET_GB", "0.000001")
    assert _select_expert_source(cfg, reserved=0) == "mmap"

    cfg.expert_store = None
    with pytest.raises(ValueError, match="ft experts repack"):
        _select_expert_source(cfg, reserved=0)

    # a non-repackable checkpoint gets the pin/residency remedy, not the GGUF-only repack
    cfg.model_path = "/models/hf-checkpoint"
    with pytest.raises(ValueError, match="pinned host RAM|--moe-cpu-layers") as exc:
        _select_expert_source(cfg, reserved=0)
    assert "ft experts repack" not in str(exc.value)


def test_parse_args_exposes_the_expert_flags(monkeypatch):
    from unittest.mock import patch

    from freetoken.server.args import parse_args

    class _Config:
        def to_dict(self):
            return {"architectures": ["LlamaForCausalLM"], "torch_dtype": "bfloat16"}

    with patch("freetoken.utils.cached_load_hf_config", lambda _path: _Config()):
        args, _run_shell = parse_args([
            "--model", "/models/anon",
            "--expert-source", "mmap",
            "--expert-store", "/tmp/store",
            "--expert-usage-file", "/tmp/u.json",
            "--expert-warm-file", "/tmp/w.json",
            "--expert-pin-budget", "4",
            "--expert-warm",
            "--expert-prefetch", "16",
            "--moe-collect-stats",
        ])
    assert args.expert_source == "mmap"
    assert args.expert_store == "/tmp/store"
    assert args.expert_usage_file == "/tmp/u.json"
    assert args.expert_warm_file == "/tmp/w.json"
    assert args.expert_pin_budget == 4.0
    assert args.expert_pin_fraction is None
    assert args.expert_warm is True
    assert args.expert_prefetch == 16
    assert args.moe_collect_stats is True


def test_experts_stats_forwards_the_store_flags(monkeypatch, tmp_path):
    import freetoken.experts.__main__ as experts_cli

    seen: dict = {}

    class _Cache:
        collect_decode_freq = False
        decode_freq = torch.zeros((1, 2), dtype=torch.int64)

    class _LLM:
        def __init__(self, model_path, **kwargs):
            seen["model_path"] = model_path
            seen.update(kwargs)
            self.engine = SimpleNamespace(moe_offload_cache=_Cache())

        def generate(self, prompts, sampling_params):
            return [{"text": ""}]

    monkeypatch.setattr("freetoken.llm.LLM", _LLM)
    monkeypatch.setattr("freetoken.gpu_select.assign_gpu", lambda gpu: None)
    monkeypatch.setattr("freetoken.gpu_select.bind_assigned_gpu", lambda: None)
    calib = tmp_path / "calib.txt"
    calib.write_text("hello")
    out = tmp_path / "usage.json"
    rc = experts_cli.main([
        "stats", "--model", "m.gguf", "--calib", str(calib), "--out", str(out),
        "--expert-source", "mmap", "--expert-store", "/tmp/store", "--expert-warm",
        "--ple-source", "/tmp/ple",
    ])
    assert rc == 0
    assert seen["expert_source"] == "mmap"
    assert seen["expert_store"] == "/tmp/store"
    assert seen["expert_warm"] is True
    assert seen["ple_source"] == "/tmp/ple"
    assert seen["moe_collect_stats"] is True
    assert out.exists()


def test_select_expert_source_pinned_matches_the_fit(monkeypatch, tmp_path):
    from freetoken.engine.engine import _select_expert_source

    cfg = SimpleNamespace(
        expert_source="auto", expert_store=None, model_path="x.gguf",
        moe_cpu_layers=None, moe_strategy="offload",
        model_config=SimpleNamespace(
            num_moe_layers=1, num_experts=2, expert_quant="q4_0",
            moe_weight_format=None, hidden_size=64, moe_intermediate_size=32,
        ),
    )
    monkeypatch.setenv("FREETOKEN_PIN_BUDGET_GB", "8")
    assert _select_expert_source(cfg, reserved=0) == "pinned"


def test_select_expert_source_allows_mmap_for_cpu_decode(monkeypatch, tmp_path):
    """D: --moe-strategy cpu over the mmap store (no pins, CPU executor reads the pages)."""
    from freetoken.engine.engine import _select_expert_source

    store = tmp_path / "store"
    store.mkdir()
    (store / "index.json").write_text('{"format": "freetoken_experts", "version": 1}')
    model = tmp_path / "x.gguf"
    model.write_bytes(b"")
    cfg = SimpleNamespace(
        expert_source="auto", expert_store=str(store), model_path=str(model),
        moe_cpu_layers=None, moe_strategy="cpu",
        model_config=SimpleNamespace(
            num_moe_layers=1, num_experts=2, expert_quant="iq4_xs+iq4_nl",
            moe_weight_format=None, hidden_size=64, moe_intermediate_size=32,
        ),
    )
    monkeypatch.setenv("FREETOKEN_PIN_BUDGET_GB", "0.000001")
    assert _select_expert_source(cfg, reserved=0) == "mmap"

    # explicit mmap is honored with a CPU strategy too
    cfg.expert_source = "mmap"
    assert _select_expert_source(cfg, reserved=0) == "mmap"

    # no store: CPU decode falls back to the OS-locked (pinned-source) path instead of erroring
    cfg.expert_source = "auto"
    cfg.expert_store = None
    assert _select_expert_source(cfg, reserved=0) == "pinned"


def test_select_expert_source_cpu_layers_keeps_pinned(monkeypatch, tmp_path):
    """Split residency needs mixed per-layer residency; auto keeps the pinned source there."""
    from freetoken.engine.engine import _select_expert_source

    store = tmp_path / "store"
    store.mkdir()
    (store / "index.json").write_text('{"format": "freetoken_experts", "version": 1}')
    cfg = SimpleNamespace(
        expert_source="auto", expert_store=str(store), model_path="x.gguf",
        moe_cpu_layers="4", moe_strategy="offload",
        model_config=SimpleNamespace(
            num_moe_layers=8, num_experts=2, expert_quant="iq4_xs+iq4_nl",
            moe_weight_format=None, hidden_size=64, moe_intermediate_size=32,
        ),
    )
    assert _select_expert_source(cfg, reserved=0) == "pinned"


def test_store_warm_file_sets_the_pin_plan(monkeypatch, tmp_path):
    from freetoken.moe.expert_banks import _store_expert_banks

    E, H, I, L = 4, 64, 32, 2
    _repack(monkeypatch, _fused_tensors(L, E, H, I), tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    warm_file = tmp_path / "reap.json"
    warm_file.write_text(json.dumps({"0": [3, 1], "1": [2]}))
    cfg = SimpleNamespace(
        num_moe_layers=L, num_experts=E, expert_quant="iq4_nl",
        hidden_size=H, moe_intermediate_size=I,
    )
    banks = _store_expert_banks(
        "m.gguf", cfg, store_dir=str(tmp_path), usage_file=None,
        warm_file=str(warm_file), pin_budget_bytes=10**9,
    )
    source = banks.source
    assert source.is_pinned_row(0, "gate_up", 3)
    assert source.is_pinned_row(0, "down", 1)
    assert source.is_pinned_row(1, "gate_up", 2)
    assert not source.is_pinned_row(0, "gate_up", 0)
    source.close()


def test_store_warm_file_rejects_a_layer_out_of_range(monkeypatch, tmp_path):
    from freetoken.moe.expert_banks import _store_expert_banks

    _repack(monkeypatch, _fused_tensors(1, 2, 64, 32), tmp_path)
    warm_file = tmp_path / "reap.json"
    warm_file.write_text(json.dumps({"3": [0]}))
    cfg = SimpleNamespace(
        num_moe_layers=1, num_experts=2, expert_quant="iq4_nl",
        hidden_size=64, moe_intermediate_size=32,
    )
    with pytest.raises(ValueError, match="MoE layers"):
        _store_expert_banks(
            "m.gguf", cfg, store_dir=str(tmp_path), usage_file=None,
            warm_file=str(warm_file), pin_budget_bytes=10**9,
        )


def test_read_rows_into_matches_the_store(monkeypatch, tmp_path):
    from freetoken.moe.expert_source import MmapExpertSource

    E, H, I, L = 5, 64, 32, 2
    index = _repack(monkeypatch, _fused_tensors(L, E, H, I), tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    source = MmapExpertSource.open(str(tmp_path), pin_plan={0: [4, 1]})
    ids = [4, 0, 2, 1, 3]
    for layer in range(L):
        for role in ("gate_up", "down"):
            loc = index.location(layer, role)
            view = source.layer_views(layer)[role]
            dst = torch.empty((len(ids), loc.rows, loc.row_bytes), dtype=torch.uint8)
            pinned = source.read_rows_into(layer, role, ids, dst)
            for i, expert in enumerate(ids):
                assert torch.equal(dst[i], view[expert])
            assert pinned == (2 if layer == 0 else 0)
    source.close()


def test_read_rows_into_rejects_a_wrong_sized_destination(monkeypatch, tmp_path):
    from freetoken.moe.expert_source import MmapExpertSource

    index = _repack(monkeypatch, _fused_tensors(1, 2, 64, 32), tmp_path)
    source = MmapExpertSource.open(str(tmp_path))
    loc = index.location(0, "gate_up")
    with pytest.raises(ValueError, match="rows"):
        source.read_rows_into(0, "gate_up", [0, 1], torch.empty((1, loc.rows, loc.row_bytes), dtype=torch.uint8))
    source.close()


def test_warm_cache_reads_every_bank(monkeypatch, tmp_path):
    from freetoken.moe.expert_source import MmapExpertSource

    E, H, I, L = 3, 64, 32, 2
    index = _repack(monkeypatch, _fused_tensors(L, E, H, I), tmp_path)
    want = sum(
        index.num_experts * index.location(layer, role).stride
        for layer in range(L)
        for role in ("gate_up", "down")
    )
    source = MmapExpertSource.open(str(tmp_path))
    assert source.warm_cache(workers=2) == want
    # warming is also reachable through open(warm=True)
    warmed = MmapExpertSource.open(str(tmp_path), warm=True)
    assert warmed.warm_cache(workers=1) == want
    source.close()
    warmed.close()

