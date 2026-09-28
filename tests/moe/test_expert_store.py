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
            "--expert-pin-budget", "4",
            "--moe-collect-stats",
        ])
    assert args.expert_source == "mmap"
    assert args.expert_store == "/tmp/store"
    assert args.expert_usage_file == "/tmp/u.json"
    assert args.expert_pin_budget == 4.0
    assert args.expert_pin_fraction is None
    assert args.moe_collect_stats is True


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
