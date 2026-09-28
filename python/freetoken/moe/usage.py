"""Per-``(layer, expert)`` routing usage and the derived pin/prefetch policy.

The offload cache can count how often each expert is routed (:meth:`decode_routing_stats`
already reads the histogram; ``--moe-collect-stats`` turns collection on). That histogram
is saved as a usage file and later consumed by ``--expert-usage-file``:

* :func:`select_pins` picks the pinned warm subset -- a per-layer floor first, then a
  global ranking of the remaining budget, so a few peaky layers cannot starve the uniform
  layers that are hit every token.
* :func:`prefetch_plan` picks the per-layer experts worth an ``MADV_WILLNEED`` before the
  layer's GEMM.

Absent a usage file the policy is empty: no explicit pinning/prefetch, and the LRU slot
cache plus the OS page cache approximate the same working set.

A second, calibration-free source of pinning is a *retained-expert plan* loaded from
``--expert-warm-file`` (:func:`load_warm_plan`, e.g. a REAP top-K JSON mapping
``layer -> [expert ids]``). It carries no frequency information, so :func:`select_warm_pins`
only seeds residency: it pins the listed experts under the pin budget and leaves the rest to
the page cache. ``--expert-usage-file`` still supplies a ranked prefetch when both are given.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass

from freetoken.utils import init_logger

logger = init_logger(__name__)

USAGE_FORMAT = "freetoken_expert_usage"
USAGE_VERSION = 1


@dataclass(frozen=True)
class UsageData:
    """Activation counts, indexed ``counts[layer][expert]``."""

    counts: list[list[int]]
    source: str | None = None

    @property
    def num_layers(self) -> int:
        return len(self.counts)

    @property
    def num_experts(self) -> int:
        return len(self.counts[0]) if self.counts else 0

    @classmethod
    def load(cls, path: str) -> "UsageData":
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
        if doc.get("format") != USAGE_FORMAT:
            raise ValueError(
                f"{path}: not an expert usage file (format={doc.get('format')!r}, "
                f"expected {USAGE_FORMAT!r})"
            )
        if int(doc.get("version", -1)) != USAGE_VERSION:
            raise ValueError(f"{path}: usage version {doc.get('version')} unsupported")
        counts = [[int(c) for c in row] for row in doc["counts"]]
        if counts and any(len(row) != len(counts[0]) for row in counts):
            raise ValueError(f"{path}: usage counts are ragged")
        return cls(counts=counts, source=doc.get("source"))

    def save(self, path: str, *, source: str | None = None) -> None:
        doc = {
            "format": USAGE_FORMAT,
            "version": USAGE_VERSION,
            "num_layers": self.num_layers,
            "num_experts": self.num_experts,
            "source": source if source is not None else self.source,
            "counts": self.counts,
        }
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(doc, f)
            f.write("\n")
        os.replace(tmp, path)

    @classmethod
    def from_freq(cls, freq, *, source: str | None = None) -> "UsageData":
        """Build from the cache's ``[num_layers, num_experts]`` decode-frequency tensor."""
        return cls(counts=[[int(v) for v in row] for row in freq.tolist()], source=source)

    def rank(self, layer: int) -> list[int]:
        """Expert ids of ``layer``, most active first (ties broken by id)."""
        row = self.counts[layer]
        return sorted(range(len(row)), key=lambda e: (-row[e], e))

    def totals(self) -> list[int]:
        return [sum(row) for row in self.counts]


def select_pins(
    usage: UsageData,
    *,
    num_experts: int,
    expert_bytes: int,
    budget_bytes: int,
) -> dict[int, list[int]]:
    """Choose the pinned warm subset under ``budget_bytes``.

    Each layer receives an even floor of the budget first; any remainder is then spent on
    a global descending ranking across layers. Returns ``{layer: [expert ids]}`` in rank
    order (empty when there is no budget or nothing to rank).
    """
    if budget_bytes <= 0 or expert_bytes <= 0 or not usage.counts:
        return {}
    L = usage.num_layers
    E = min(num_experts, usage.num_experts)
    per_layer_budget = budget_bytes // L
    floor_k = min(E, per_layer_budget // expert_bytes)

    pinned: dict[int, list[int]] = {}
    spent = 0
    for layer in range(L):
        k = min(floor_k, E)
        ranked = usage.rank(layer)[:k]
        pinned[layer] = ranked
        spent += len(ranked) * expert_bytes

    remaining = budget_bytes - spent
    if remaining <= 0:
        return pinned
    # Global remainder: every not-yet-pinned expert, most active first. A layer that was
    # already capped at E contributes nothing; a peaky layer can now take extra slots.
    candidates: list[tuple[int, int, int]] = []
    for layer in range(L):
        already = set(pinned[layer])
        row = usage.counts[layer]
        for expert in range(E):
            if expert not in already:
                candidates.append((row[expert], layer, expert))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    for _, layer, expert in candidates:
        if remaining < expert_bytes:
            break
        pinned[layer].append(expert)
        remaining -= expert_bytes
    for layer in range(L):
        if len(pinned[layer]) > 1:
            # keep pin order stable by rank (the append order above is already descending,
            # but the floor prefix + remainder suffix should read as one ranked list)
            pinned[layer].sort(key=lambda e: (-usage.counts[layer][e], e))
    return pinned


def load_warm_plan(path: str) -> dict[int, list[int]]:
    """Load a per-layer retained-expert plan: ``{"<layer>": [expert ids...], ...}``.

    This is the shape of a REAP top-K dump (the retained set of a REAPed sibling), not a
    frequency ranking. The plan only seeds the pinned warm subset (``--expert-warm-file``);
    pass a usage file with ``--expert-usage-file`` when a ranked prefetch is also wanted.
    """
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    if not isinstance(doc, dict):
        raise ValueError(f"{path}: warm plan must be a JSON object of layer -> [expert ids]")
    if doc.get("format") == USAGE_FORMAT:
        raise ValueError(
            f"{path}: this is an expert usage file; pass it with --expert-usage-file "
            f"(or convert it to a layer -> [expert ids] warm plan)"
        )
    plan: dict[int, list[int]] = {}
    for key, ids in doc.items():
        try:
            layer = int(key)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}: warm plan layer key must be an integer, got {key!r}") from exc
        if not isinstance(ids, list):
            raise ValueError(f"{path}: warm plan layer {layer} must be a list of expert ids")
        try:
            plan[layer] = [int(e) for e in ids]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}: warm plan layer {layer} holds a non-integer expert id") from exc
    return plan


def select_warm_pins(
    plan: dict[int, Sequence[int]],
    *,
    num_layers: int,
    num_experts: int,
    expert_bytes: int,
    budget_bytes: int,
) -> dict[int, list[int]]:
    """Pin the planned (retained) experts under ``budget_bytes``.

    Uses the same per-layer floor as :func:`select_pins` (an even share of the budget per
    layer, then any remainder spent across layers), but the plan has no ranking to spend the
    remainder on, so leftover budget is left to the page cache. Ids out of range are
    dropped. Returns ``{layer: [expert ids]}``.
    """
    if budget_bytes <= 0 or expert_bytes <= 0 or num_layers <= 0 or not plan:
        return {}
    per_layer_budget = budget_bytes // num_layers
    floor_k = per_layer_budget // expert_bytes

    planned: dict[int, list[int]] = {}
    for layer in range(num_layers):
        seen: set[int] = set()
        ids: list[int] = []
        for raw in plan.get(layer, ()):
            expert = int(raw)
            if 0 <= expert < num_experts and expert not in seen:
                seen.add(expert)
                ids.append(expert)
        planned[layer] = ids

    pinned: dict[int, list[int]] = {}
    spent = 0
    for layer in range(num_layers):
        k = min(floor_k, len(planned[layer]))
        pinned[layer] = planned[layer][:k]
        spent += k * expert_bytes

    remaining = budget_bytes - spent
    taken = {layer: len(pinned[layer]) for layer in range(num_layers)}
    progressed = True
    while remaining >= expert_bytes and progressed:
        progressed = False
        for layer in range(num_layers):
            if remaining < expert_bytes:
                break
            if taken[layer] < len(planned[layer]):
                pinned[layer].append(planned[layer][taken[layer]])
                taken[layer] += 1
                remaining -= expert_bytes
                progressed = True
    return pinned


def prefetch_plan(usage: UsageData, top_n: int) -> list[list[int]]:
    """Per-layer top-``top_n`` experts worth a prefetch (empty when ``top_n <= 0``)."""
    if top_n <= 0 or not usage.counts:
        return [[] for _ in range(usage.num_layers)]
    return [usage.rank(layer)[:top_n] for layer in range(usage.num_layers)]


def load_usage(path: str | None) -> UsageData | None:
    if not path:
        return None
    return UsageData.load(path)
