"""Select a small, diverse subset of patch IDs for debugging."""

from __future__ import annotations

from typing import Any

from src.utils.data_loader import group_by_patch


def _min_confidence(records: list[dict[str, Any]]) -> float | None:
    """Minimum non-null confidence among records; ``None`` if none present."""
    values: list[float] = []
    for r in records:
        c = r.get("confidence")
        if c is not None:
            values.append(float(c))
    if not values:
        return None
    return min(values)


def _patch_has_mask(records: list[dict[str, Any]]) -> bool:
    return any(r.get("mask_rle") is not None for r in records)


def select_sample_patches(records: list[dict[str, Any]]) -> list[str]:
    """Pick up to ~10 diverse patch IDs for inspection.

    Strategy:

    * Top 3 patches by annotation count
    * 3 patches with lowest minimum annotation confidence (among patches that have confidence)
    * 2 patches that include at least one mask (``mask_rle``)
    * 2 patches with no masks on any annotation

    Order is deterministic; duplicates are skipped. Result length is capped at 10.
    """
    by_patch = group_by_patch(records)
    if not by_patch:
        return []

    def add_unique(out: list[str], pid: str) -> None:
        if pid not in out:
            out.append(pid)

    selected: list[str] = []

    # Top 3 by count
    by_count = sorted(
        by_patch.items(),
        key=lambda x: (-len(x[1]), x[0]),
    )
    for pid, _ in by_count[:3]:
        add_unique(selected, pid)

    # 3 lowest min-confidence (patches with at least one confidence value)
    with_conf = [(pid, recs) for pid, recs in by_patch.items() if _min_confidence(recs) is not None]
    by_min_conf = sorted(
        with_conf,
        key=lambda x: (_min_confidence(x[1]) or 0.0, x[0]),
    )
    for pid, _ in by_min_conf[:3]:
        add_unique(selected, pid)

    # 2 with mask
    with_mask = sorted(pid for pid, recs in by_patch.items() if _patch_has_mask(recs))
    for pid in with_mask[:2]:
        add_unique(selected, pid)

    # 2 without mask
    without_mask = sorted(pid for pid, recs in by_patch.items() if not _patch_has_mask(recs))
    for pid in without_mask[:2]:
        add_unique(selected, pid)

    return selected[:10]
