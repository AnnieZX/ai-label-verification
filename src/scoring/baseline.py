"""Baseline verification score from confidence, geometry, and mask heuristics."""

from __future__ import annotations

from typing import Any

from src.features.geometry import extract_geometry_features, geometry_score
from src.features.mask_features import extract_mask_features, mask_score


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _decision_from_final(final_score: float) -> str:
    if final_score >= 0.75:
        return "likely_good"
    if final_score >= 0.45:
        return "review"
    return "suspicious"


def _collect_reasons(
    record: dict[str, Any],
    geom: dict[str, Any],
    mask: dict[str, Any],
    confidence: float | None,
) -> list[str]:
    reasons: list[str] = []

    if geom.get("bbox_out_of_bounds"):
        reasons.append("bbox goes out of image bounds")

    if geom.get("bbox_has_negative_coords"):
        reasons.append("bbox has negative coordinates")

    ar = float(geom.get("bbox_area_ratio") or 0.0)
    if ar < 1e-4 and float(geom.get("bbox_area") or 0) > 0:
        reasons.append("bbox area is very small")

    if ar > 0.95:
        reasons.append("bbox covers most of the image")

    if not mask.get("has_mask"):
        reasons.append("mask is missing")

    ratio_mb = mask.get("mask_bbox_area_ratio")
    if mask.get("has_mask") and ratio_mb is not None:
        if ratio_mb < 0.05 or ratio_mb > 1.5:
            reasons.append("mask-to-bbox ratio looks suspicious")

    if mask.get("has_mask") and mask.get("mask_area_from_properties") is None:
        reasons.append("mask has no reported area in metadata")

    if confidence is not None and confidence < 0.35:
        reasons.append("low model confidence")

    return reasons


def baseline_verification_score(
    record: dict[str, Any],
    image_size: tuple[int, int] = (800, 800),
) -> dict[str, Any]:
    """Compute a first-pass verification score without LVM or rasterized masks.

    Parameters
    ----------
    record
        Flat annotation dict (e.g. from :func:`src.utils.data_loader.load_annotations`).
    image_size
        ``(width, height)`` of the patch coordinate system for geometry checks.
    """
    geom = extract_geometry_features(record, image_size=image_size)
    mfeat = extract_mask_features(record)
    g = geometry_score(geom)
    m = mask_score(mfeat)

    raw_conf = record.get("confidence")
    confidence: float | None
    try:
        confidence = float(raw_conf) if raw_conf is not None else None
    except (TypeError, ValueError):
        confidence = None

    if confidence is not None:
        final_score = 0.5 * confidence + 0.3 * g + 0.2 * m
    else:
        final_score = 0.6 * g + 0.4 * m

    final_score = _clamp01(final_score)
    decision = _decision_from_final(final_score)
    reasons = _collect_reasons(record, geom, mfeat, confidence)

    return {
        "confidence": confidence,
        "geometry_score": round(g, 4),
        "mask_score": round(m, 4),
        "final_score": round(final_score, 4),
        "decision": decision,
        "reasons": reasons,
    }
