"""CPU MoE executor -- native GGUF IQ4_NL / IQ4_XS / Q5_K experts (incl. composites).

The CPU W4A8 GEMV reads the *same* packed banks the GPU offload path streams and
dequantizes inside the K-loop (``iq4_nl_dot`` / ``iq4_xs_dot`` / ``q5_k_dot`` in
csrc/cpu_moe/cpu_moe_ext.cpp). We check it against the GPU ggml MoE kernel
(``ggml_moe_a8_vec``, also W4A8) on byte-identical banks; both quantize activations to
int8, so the only spread is the activation-quant grid + reduction order.

A composite tag (``iq4_xs+iq4_nl``) packs gate_up and down with different block codecs;
the CPU executor now picks a per-role dot kernel while sharing the single Q8_0/32
activation quantization.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from freetoken.models.gguf.dequant import (
    BLOCK_SHAPE,
    GGML_IQ4_NL,
    GGML_IQ4_XS,
    GGML_Q5_K,
    row_bytes,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

# (format tag, gate_up type, down type); the last two are composites: the two roles use
# different block codecs and the CPU executor selects a dot kernel per role.
_CASES = [
    ("iq4_nl", GGML_IQ4_NL, GGML_IQ4_NL),
    ("iq4_xs", GGML_IQ4_XS, GGML_IQ4_XS),
    ("q5_K", GGML_Q5_K, GGML_Q5_K),
    ("iq4_xs+iq4_nl", GGML_IQ4_XS, GGML_IQ4_NL),
    ("iq4_nl+iq4_xs", GGML_IQ4_NL, GGML_IQ4_XS),
]


def _random_bank(quant_type: int, rows: int, k: int, seed: int) -> torch.Tensor:
    """[rows, row_bytes] random packed blocks, fp16 scale fields forced finite/positive."""
    block, size = BLOCK_SHAPE[quant_type]
    rng = np.random.default_rng(seed)
    n = rows * (k // block)
    blocks = rng.integers(0, 256, size=(n, size), dtype=np.uint8)
    scales = (0.001 + 0.01 * rng.random(n)).astype(np.float16)
    raw = np.frombuffer(scales.tobytes(), np.uint8).reshape(n, 2)
    blocks[:, 0:2] = raw
    if quant_type in (GGML_Q5_K, GGML_IQ4_XS):
        blocks[:, 2:4] = raw  # dmin / scales_h
    return torch.from_numpy(blocks.reshape(rows, k // block * size)).contiguous()


def _make_cache(
    fmt: str, gu_quant_type: int, dn_quant_type: int, L: int, E: int, H: int, I: int, seed: int
):
    from freetoken.kernel.pinned import alloc_pinned_tensor

    def bank_layers(out_f: int, k: int, quant_type: int, off: int) -> list[torch.Tensor]:
        # per-layer banks [E, out_f, row_bytes(k)], matching load_gguf_expert_sources
        packed = _random_bank(quant_type, L * E * out_f, k, seed + off).reshape(L, E, out_f, -1)
        layers = []
        for i in range(L):
            pinned = alloc_pinned_tensor(*packed[i].shape, dtype=torch.uint8)
            pinned.copy_(packed[i])
            layers.append(pinned)
        return layers

    return SimpleNamespace(
        quant_format=fmt,
        bank_sources={
            "gate_up": bank_layers(2 * I, H, gu_quant_type, 1),
            "down": bank_layers(H, I, dn_quant_type, 2),
        },
        num_layers=L,
        num_experts=E,
        decode_target="cpu",
        cpu_executor=None,
    )


@pytest.mark.parametrize("fmt,gu_quant_type,dn_quant_type", _CASES)
@pytest.mark.parametrize("bs", [1, 3])
def test_cpu_decode_gguf_quant_matches_gpu(
    fmt: str, gu_quant_type: int, dn_quant_type: int, bs: int
):
    from freetoken.moe.cpu_executor import CpuMoeExecutor
    from freetoken.moe.gguf_experts import fused_experts_gguf

    torch.manual_seed(400 + bs)
    L, E, H, I, top_k = 2, 8, 512, 256, 2  # H, I multiples of the 256-elem K-quant block
    layer = 1
    dev = torch.device("cuda")
    cache = _make_cache(fmt, gu_quant_type, dn_quant_type, L, E, H, I, seed=gu_quant_type)

    ex = CpuMoeExecutor(
        cache,
        top_k=top_k,
        activation="silu",
        apply_router_weight_on_input=False,
        num_threads=0,
        max_tokens=bs,
        device=dev,
    )

    hidden = torch.randn(bs, H, device=dev, dtype=torch.bfloat16) * 0.5
    ids = torch.stack([torch.randperm(E, device=dev)[:top_k] for _ in range(bs)]).to(torch.int32)
    w = torch.rand(bs, top_k, device=dev, dtype=torch.float32)

    cpu_out = ex.decode(layer, hidden, w, ids).float()
    torch.cuda.synchronize()

    banks = cache.bank_sources
    gpu_out = fused_experts_gguf(
        hidden, banks["gate_up"][layer].to(dev), banks["down"][layer].to(dev),
        w, ids.clone(), "silu", gu_quant_type, dn_quant_type,
    ).float()

    rel = (cpu_out - gpu_out).abs().max() / (gpu_out.abs().max() + 1e-6)
    assert rel < 6e-2, f"{fmt} bs={bs} cpu-vs-gpu rel err {rel.item()}"


def _write_store(tmp_path, fmt: str, gu_type: int, dn_type: int, L: int, E: int, H: int, I: int, seed: int):
    """Write a fixed-stride store mirroring `ft experts repack`'s on-disk layout."""
    gu_rb, dn_rb = row_bytes(H, gu_type), row_bytes(I, dn_type)
    gu_rows, dn_rows = 2 * I, H
    layers = []
    for layer in range(L):
        gu = _random_bank(gu_type, E * gu_rows, H, seed + layer).numpy()
        dn = _random_bank(dn_type, E * dn_rows, I, seed + 100 + layer).numpy()
        (tmp_path / f"layer-{layer:03d}.gate_up.bin").write_bytes(gu.tobytes())
        (tmp_path / f"layer-{layer:03d}.down.bin").write_bytes(dn.tobytes())
        layers.append({
            "layer": layer,
            "banks": {
                "gate_up": {
                    "file": f"layer-{layer:03d}.gate_up.bin", "offset": 0,
                    "stride": gu_rows * gu_rb, "rows": gu_rows, "row_bytes": gu_rb,
                },
                "down": {
                    "file": f"layer-{layer:03d}.down.bin", "offset": 0,
                    "stride": dn_rows * dn_rb, "rows": dn_rows, "row_bytes": dn_rb,
                },
            },
        })
    (tmp_path / "index.json").write_text(json.dumps({
        "format": "freetoken_experts", "version": 1, "quant_format": fmt,
        "num_layers": L, "num_experts": E, "hidden_size": H, "intermediate_size": I,
        "roles": ["gate_up", "down"], "layers": layers,
    }))
    return tmp_path


@pytest.mark.parametrize("fmt,gu_quant_type,dn_quant_type", _CASES)
def test_cpu_decode_over_mmap_store_matches_gpu(
    tmp_path, fmt: str, gu_quant_type: int, dn_quant_type: int
):
    """D: the CPU executor reads a file-backed mmap store directly (no pinned banks)."""
    from freetoken.moe.cpu_executor import CpuMoeExecutor
    from freetoken.moe.expert_source import MmapExpertSource
    from freetoken.moe.gguf_experts import fused_experts_gguf

    torch.manual_seed(700)
    L, E, H, I, top_k = 2, 8, 512, 256, 2
    layer, bs = 1, 1
    dev = torch.device("cuda")
    store = _write_store(tmp_path, fmt, gu_quant_type, dn_quant_type, L, E, H, I, seed=gu_quant_type)
    source = MmapExpertSource.open(str(store))
    assert not source.resident
    cache = SimpleNamespace(
        quant_format=fmt,
        bank_sources=source.all_layer_views(),
        num_layers=L,
        num_experts=E,
        decode_target="cpu",
        cpu_executor=None,
        expert_source=source,
    )
    ex = CpuMoeExecutor(
        cache, top_k=top_k, activation="silu", apply_router_weight_on_input=False,
        num_threads=0, max_tokens=bs, device=dev,
    )
    assert ex.mmap_prefetch  # the executor enabled the C++ per-expert MADV_WILLNEED

    hidden = torch.randn(bs, H, device=dev, dtype=torch.bfloat16) * 0.5
    ids = torch.stack([torch.randperm(E, device=dev)[:top_k] for _ in range(bs)]).to(torch.int32)
    w = torch.rand(bs, top_k, device=dev, dtype=torch.float32)

    cpu_out = ex.decode(layer, hidden, w, ids).float()
    torch.cuda.synchronize()

    banks = cache.bank_sources
    gpu_out = fused_experts_gguf(
        hidden, banks["gate_up"][layer].to(dev), banks["down"][layer].to(dev),
        w, ids.clone(), "silu", gu_quant_type, dn_quant_type,
    ).float()
    rel = (cpu_out - gpu_out).abs().max() / (gpu_out.abs().max() + 1e-6)
    assert rel < 6e-2, f"{fmt} mmap cpu-vs-gpu rel err {rel.item()}"
    source.close()


if __name__ == "__main__":
    for fmt, gu_quant_type, dn_quant_type in _CASES:
        for bs in (1, 3):
            test_cpu_decode_gguf_quant_matches_gpu(fmt, gu_quant_type, dn_quant_type, bs)
            print(f"{fmt} bs={bs} OK")
