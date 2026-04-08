#!/usr/bin/env python3
"""End-to-end smoke test: load annotations, sample patches, baseline verification."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Allow `python scripts/test_pipeline.py` from any working directory
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.features.geometry import extract_geometry_features  # noqa: E402
from src.features.mask_features import extract_mask_features  # noqa: E402
from src.preprocessing.overlay_stub import build_overlay_input  # noqa: E402
from src.scoring.baseline import baseline_verification_score  # noqa: E402
from src.utils.data_loader import group_by_patch, load_annotations  # noqa: E402
from src.utils.sample_selector import select_sample_patches  # noqa: E402

_PATCH_SIZE: tuple[int, int] = (800, 800)


def _preview_overlay(overlay: dict[str, object]) -> dict[str, object]:
    """Shrink RLE for terminal output only."""
    out = dict(overlay)
    rle = out.get("mask_rle")
    if isinstance(rle, dict):
        counts = rle.get("counts")
        n = len(counts) if isinstance(counts, list) else None
        out["mask_rle"] = {
            "size": rle.get("size"),
            "counts_values": n,
        }
    return out


def _geom_summary(features: dict[str, Any]) -> dict[str, Any]:
    return {
        "bbox_area": round(float(features["bbox_area"]), 2),
        "bbox_area_ratio": round(float(features["bbox_area_ratio"]), 6),
        "bbox_out_of_bounds": features["bbox_out_of_bounds"],
        "bbox_has_negative_coords": features["bbox_has_negative_coords"],
        "bbox_aspect_ratio": (
            None
            if features["bbox_aspect_ratio"] is None
            else round(float(features["bbox_aspect_ratio"]), 4)
        ),
    }


def _mask_summary(features: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "has_mask": features["has_mask"],
        "mask_size": features["mask_size"],
        "mask_counts_length": features["mask_counts_length"],
        "mask_area_from_properties": features["mask_area_from_properties"],
    }
    mar = features.get("mask_area_ratio_estimate")
    mbr = features.get("mask_bbox_area_ratio")
    out["mask_area_ratio_estimate"] = None if mar is None else round(float(mar), 6)
    out["mask_bbox_area_ratio"] = None if mbr is None else round(float(mbr), 6)
    return out


def main() -> None:
    """Load data, print sample patch summaries, baseline scores, overlay stubs."""
    json_path = _REPO_ROOT / "data" / "annotations.json"
    print(f"Loading: {json_path}")
    records = load_annotations(str(json_path))
    print(f"Total flat records: {len(records)}")

    sample_patch_ids = select_sample_patches(records)
    by_patch = group_by_patch(records)

    print(f"\nSelected {len(sample_patch_ids)} patch id(s) for preview:\n")

    for patch_id in sample_patch_ids:
        patch_records = by_patch.get(patch_id, [])
        n = len(patch_records)
        print("-" * 60)
        print(f"patch_id: {patch_id}")
        print(f"annotations: {n}")

        for i, rec in enumerate(patch_records[:2]):
            geom = extract_geometry_features(rec, image_size=_PATCH_SIZE)
            mfeat = extract_mask_features(rec)
            baseline = baseline_verification_score(rec, image_size=_PATCH_SIZE)

            line = json.dumps(rec, default=str)
            if len(line) > 220:
                line = line[:220] + "..."
            print(f"  example[{i}] record: {line}")
            print(f"    geometry: {json.dumps(_geom_summary(geom), default=str)}")
            print(f"    mask: {json.dumps(_mask_summary(mfeat), default=str)}")
            print(
                "    baseline: "
                f"final={baseline['final_score']} decision={baseline['decision']!r} "
                f"g={baseline['geometry_score']} m={baseline['mask_score']} "
                f"conf={baseline['confidence']}"
            )
            rsn = baseline["reasons"]
            if rsn:
                print(f"    reasons: {rsn[:5]}{' ...' if len(rsn) > 5 else ''}")

        if patch_records:
            example = patch_records[0]
        else:
            example = {"patch_id": patch_id, "bbox": None, "mask_rle": None, "label": None}

        overlay = build_overlay_input(example)
        print(f"  build_overlay_input: {json.dumps(_preview_overlay(overlay), default=str)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
