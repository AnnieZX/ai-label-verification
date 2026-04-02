"""Placeholder overlay / visualization hooks (no image I/O or drawing)."""

from __future__ import annotations

from typing import Any


def build_overlay_input(record: dict[str, Any]) -> dict[str, Any]:
    """Prepare structured data for visualization or model input.

    Parameters are flat annotation records from :func:`src.utils.data_loader.load_annotations`.
    """
    return {
        "patch_id": record.get("patch_id"),
        "bbox": record.get("bbox"),
        "mask_rle": record.get("mask_rle"),
        "label": record.get("label"),
    }


def visualize_overlay_stub(record: dict[str, Any]) -> None:
    """Placeholder for future visualization."""
    patch_id = record.get("patch_id", "?")
    print(f"[STUB] Visualizing patch {patch_id}")
