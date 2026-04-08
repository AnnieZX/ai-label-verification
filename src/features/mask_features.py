"""Lightweight mask-related features without decoding RLE to a dense grid."""

from __future__ import annotations

from typing import Any


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _mask_area_from_record(record: dict[str, Any]) -> float | None:
    """Pull an optional scalar mask area from common record locations."""
    direct = record.get("mask_area")
    if direct is not None:
        try:
            return float(direct)
        except (TypeError, ValueError):
            pass

    props = record.get("properties")
    if isinstance(props, dict):
        for key in ("mask_area", "segmentation_area", "rle_area"):
            val = props.get(key)
            if val is not None:
                try:
                    return float(val)
                except (TypeError, ValueError):
                    continue
    return None


def _bbox_area(record: dict[str, Any]) -> float | None:
    raw = record.get("bbox")
    if raw is None or not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]))
    except (TypeError, ValueError):
        return None
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    a = w * h
    return a if a > 0 else None


def extract_mask_features(record: dict[str, Any]) -> dict[str, Any]:
    """Summarize mask metadata: presence, RLE shape hints, and optional area ratios.

    Does not decode ``mask_rle`` into pixels. Expects ``mask_rle`` dict with optional
    ``size`` (``[h, w]``) and ``counts`` (list).
    """
    mask_rle = record.get("mask_rle")
    if mask_rle is None or not isinstance(mask_rle, dict):
        return {
            "has_mask": False,
            "mask_size": None,
            "mask_counts_length": None,
            "mask_area_from_properties": _mask_area_from_record(record),
            "mask_area_ratio_estimate": None,
            "mask_bbox_area_ratio": None,
        }

    size_raw = mask_rle.get("size")
    mask_size: list[int] | None = None
    if isinstance(size_raw, (list, tuple)) and len(size_raw) == 2:
        try:
            mask_size = [int(size_raw[0]), int(size_raw[1])]
        except (TypeError, ValueError):
            mask_size = None

    counts = mask_rle.get("counts")
    counts_len: int | None = None
    if isinstance(counts, list):
        counts_len = len(counts)

    mask_area = _mask_area_from_record(record)

    mask_area_ratio: float | None = None
    if mask_size is not None and mask_area is not None:
        denom = float(mask_size[0] * mask_size[1])
        if denom > 0:
            mask_area_ratio = mask_area / denom

    bbox_area = _bbox_area(record)
    mask_bbox_ratio: float | None = None
    if mask_area is not None and bbox_area is not None and bbox_area > 0:
        mask_bbox_ratio = mask_area / bbox_area

    return {
        "has_mask": True,
        "mask_size": mask_size,
        "mask_counts_length": counts_len,
        "mask_area_from_properties": mask_area,
        "mask_area_ratio_estimate": mask_area_ratio,
        "mask_bbox_area_ratio": mask_bbox_ratio,
    }


def mask_score(features: dict[str, Any]) -> float:
    """Interpretable mask consistency score in ``[0, 1]``."""
    if not features.get("has_mask"):
        return 0.5

    score = 1.0

    mask_area = features.get("mask_area_from_properties")
    if mask_area is None:
        score -= 0.25

    ratio_mb = features.get("mask_bbox_area_ratio")
    if ratio_mb is not None:
        if ratio_mb < 0.05:
            score -= 0.3
        if ratio_mb > 1.5:
            score -= 0.25

    ratio_est = features.get("mask_area_ratio_estimate")
    if ratio_est is not None:
        if ratio_est < 1e-4:
            score -= 0.15
        if ratio_est > 1.0:
            score -= 0.2

    return _clamp01(score)
