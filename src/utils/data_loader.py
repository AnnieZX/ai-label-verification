"""Load patch-annotation JSON into flat records for the verification pipeline."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping


def _as_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_list(value: Any) -> list[Any] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return list(value)
    return None


def _as_mask_rle(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return dict(value)
    return None


def _metadata_mapping(patch_obj: Mapping[str, Any]) -> Mapping[str, Any]:
    raw = patch_obj.get("metadata")
    if isinstance(raw, Mapping):
        return raw
    return {}


def _annotation_confidence(ann: Mapping[str, Any]) -> float | None:
    props = ann.get("properties")
    if not isinstance(props, Mapping):
        return None
    return _as_float(props.get("confidence"))


def load_annotations(json_path: str) -> list[dict[str, Any]]:
    """Parse annotation JSON into one flat dict per (patch, annotation).

    Expects top-level ``patches``: patch_id -> ``{ "metadata": ..., "annotations": [...] }``.

    Missing fields are returned as ``None``. No image or URL resolution is performed.
    """
    path = Path(json_path)
    text = path.read_text(encoding="utf-8")
    data = json.loads(text)

    if not isinstance(data, dict):
        msg = "Root JSON value must be an object"
        raise ValueError(msg)

    patches = data.get("patches")
    if not isinstance(patches, dict):
        msg = "Expected key 'patches' with an object value"
        raise ValueError(msg)

    records: list[dict[str, Any]] = []

    for patch_id, patch_obj in patches.items():
        if not isinstance(patch_obj, Mapping):
            continue

        meta = _metadata_mapping(patch_obj)
        item_id = _as_str(meta.get("itemId"))
        grid_index = _as_int(meta.get("gridIndex"))
        tiff_url = _as_str(meta.get("tiffUrl"))
        bounds = _as_list(meta.get("bounds"))

        raw_anns = patch_obj.get("annotations", [])
        if not isinstance(raw_anns, list):
            continue

        for ann in raw_anns:
            if not isinstance(ann, Mapping):
                continue

            records.append(
                {
                    "patch_id": str(patch_id),
                    "item_id": item_id,
                    "grid_index": grid_index,
                    "bbox": _as_list(ann.get("pixelBbox")),
                    "mask_rle": _as_mask_rle(ann.get("segmentationRLE")),
                    "label": _as_str(ann.get("classLabel")),
                    "confidence": _annotation_confidence(ann),
                    "tiff_url": tiff_url,
                    "bounds": bounds,
                }
            )

    return records


def group_by_patch(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group flat records by ``patch_id``."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        pid = rec.get("patch_id")
        if isinstance(pid, str):
            grouped[pid].append(rec)
    return dict(grouped)
