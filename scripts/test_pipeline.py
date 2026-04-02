#!/usr/bin/env python3
"""End-to-end smoke test: load annotations, sample patches, overlay stubs (no images)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow `python scripts/test_pipeline.py` from any working directory
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.preprocessing.overlay_stub import build_overlay_input  # noqa: E402
from src.utils.data_loader import group_by_patch, load_annotations  # noqa: E402
from src.utils.sample_selector import select_sample_patches  # noqa: E402


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


def main() -> None:
    """Load data, print sample patch summaries, build stub overlay payloads."""
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
            line = json.dumps(rec, default=str)
            if len(line) > 240:
                line = line[:240] + "..."
            print(f"  example[{i}]: {line}")

        if patch_records:
            example = patch_records[0]
        else:
            example = {"patch_id": patch_id, "bbox": None, "mask_rle": None, "label": None}

        overlay = build_overlay_input(example)
        print(f"  build_overlay_input: {json.dumps(_preview_overlay(overlay), default=str)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
