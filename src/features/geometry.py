"""Bounding-box geometry features and a simple interpretable quality score."""

from __future__ import annotations

from typing import Any


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _parse_bbox(record: dict[str, Any]) -> tuple[float, float, float, float] | None:
    raw = record.get("bbox")
    if raw is None or not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        return float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3])
    except (TypeError, ValueError):
        return None


def extract_geometry_features(
    record: dict[str, Any],
    image_size: tuple[int, int] = (800, 800),
) -> dict[str, Any]:
    """Compute bbox geometry features relative to patch / image size.

    ``bbox`` format: ``[x1, y1, x2, y2]``. Missing or invalid bbox yields zeros
    and conservative flags.
    """
    image_w, image_h = int(image_size[0]), int(image_size[1])
    image_plane = float(max(1, image_w * image_h))

    parsed = _parse_bbox(record)
    if parsed is None:
        return {
            "bbox_width": 0.0,
            "bbox_height": 0.0,
            "bbox_area": 0.0,
            "bbox_aspect_ratio": None,
            "bbox_area_ratio": 0.0,
            "bbox_out_of_bounds": False,
            "bbox_has_negative_coords": False,
        }

    x1, y1, x2, y2 = parsed
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    area = width * height
    area_ratio = area / image_plane
    aspect: float | None
    if height > 0:
        aspect = width / height
    else:
        aspect = None

    out_of_bounds = (
        x1 < 0
        or y1 < 0
        or x2 > float(image_w)
        or y2 > float(image_h)
    )
    negative_coords = x1 < 0 or y1 < 0

    return {
        "bbox_width": width,
        "bbox_height": height,
        "bbox_area": area,
        "bbox_aspect_ratio": aspect,
        "bbox_area_ratio": area_ratio,
        "bbox_out_of_bounds": out_of_bounds,
        "bbox_has_negative_coords": negative_coords,
    }


def geometry_score(features: dict[str, Any]) -> float:
    """Baseline geometry plausibility score in ``[0, 1]`` with simple penalties."""
    score = 1.0

    area = float(features.get("bbox_area") or 0.0)
    if area <= 0.0:
        score -= 0.5

    if features.get("bbox_out_of_bounds"):
        score -= 0.35

    if features.get("bbox_has_negative_coords"):
        score -= 0.25

    ar = float(features.get("bbox_area_ratio") or 0.0)
    if ar < 1e-4:
        score -= 0.2
    if ar > 0.95:
        score -= 0.2

    return _clamp01(score)
