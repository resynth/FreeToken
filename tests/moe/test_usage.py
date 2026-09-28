"""Usage ranking and the derived pin/prefetch policy (moe.usage)."""

from __future__ import annotations

import pytest

from freetoken.moe.usage import UsageData, prefetch_plan, select_pins


def _usage(rows: list[list[int]]) -> UsageData:
    return UsageData(counts=rows)


def test_usage_round_trip(tmp_path):
    rows = [[1, 5, 3], [0, 2, 9]]
    path = tmp_path / "u.json"
    _usage(rows).save(str(path))
    loaded = UsageData.load(str(path))
    assert loaded.counts == rows
    assert loaded.num_layers == 2 and loaded.num_experts == 3


def test_usage_rejects_a_foreign_format(tmp_path):
    path = tmp_path / "x.json"
    path.write_text('{"format": "nope", "version": 1, "counts": [[1]]}')
    with pytest.raises(ValueError, match="not an expert usage file"):
        UsageData.load(str(path))


def test_usage_rejects_ragged_counts(tmp_path):
    path = tmp_path / "r.json"
    path.write_text(
        '{"format": "freetoken_expert_usage", "version": 1, "counts": [[1, 2], [3]]}'
    )
    with pytest.raises(ValueError, match="ragged"):
        UsageData.load(str(path))


def test_rank_is_activity_descending_then_id():
    usage = _usage([[1, 9, 9, 0]])
    assert usage.rank(0) == [1, 2, 0, 3]


def test_select_pins_respects_the_per_layer_floor():
    # Layer 0 is peaky, layer 2 is uniform and would lose a pure global ranking.
    counts = [[0] * 8 for _ in range(4)]
    counts[0] = [1000, 900, 0, 0, 0, 0, 0, 0]
    counts[1] = [10, 9, 8, 7, 6, 5, 4, 3]
    counts[2] = [5, 5, 5, 5, 5, 5, 5, 5]
    counts[3] = [1, 2, 3, 4, 5, 6, 7, 8]
    pins = select_pins(
        _usage(counts), num_experts=8, expert_bytes=100, budget_bytes=1000
    )
    assert all(len(pins[layer]) >= 2 for layer in range(4)), pins
    assert 0 in pins[0] and 1 in pins[0]
    assert sum(len(v) for v in pins.values()) == 10  # exactly the budget
    assert pins[0][:2] == [0, 1]


def test_select_pins_caps_at_num_experts():
    usage = _usage([[1, 2, 3, 4]])
    pins = select_pins(usage, num_experts=4, expert_bytes=1, budget_bytes=10**9)
    assert pins[0] == [3, 2, 1, 0]


def test_select_pins_without_budget_is_empty():
    usage = _usage([[1, 2]])
    assert select_pins(usage, num_experts=2, expert_bytes=10, budget_bytes=0) == {}


def test_prefetch_plan_takes_the_top_n():
    usage = _usage([[1, 9, 9, 0], [4, 4, 4, 4]])
    assert prefetch_plan(usage, 2) == [[1, 2], [0, 1]]
    assert prefetch_plan(usage, 0) == [[], []]
