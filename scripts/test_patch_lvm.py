#!/usr/bin/env python3
"""Validate patch -> overlay -> open-source LVM flow on a real cropped patch."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow `python scripts/test_patch_lvm.py` from any working directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.lvm.open_source_verifier import verify_bbox_with_open_source_lvm  # noqa: E402
from src.preprocessing.demo_overlay import draw_single_bbox_overlay  # noqa: E402


def main() -> None:
    image_path = _REPO_ROOT / "outputs" / "test_patch.jpg"
    overlay_path = _REPO_ROOT / "outputs" / "test_patch_overlay.jpg"
    result_path = _REPO_ROOT / "outputs" / "lvm_results" / "test_patch_result.json"

    bbox = [100, 100, 500, 500]
    label = "item"

    if not image_path.exists():
        msg = (
            f"Input patch not found: {image_path}\n"
            "Run `python scripts/test_png_crop.py` first."
        )
        raise FileNotFoundError(msg)

    draw_single_bbox_overlay(
        image_path=str(image_path),
        bbox=bbox,
        output_path=str(overlay_path),
        label=label,
    )

    result = verify_bbox_with_open_source_lvm(
        image_path=str(image_path),
        bbox=bbox,
        label=label,
    )

    print(f"decision: {result.get('decision')}")
    print(f"score: {result.get('score')}")
    print(f"explanation: {result.get('explanation')}")

    result_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "image_path": str(image_path),
        "overlay_path": str(overlay_path),
        "bbox": bbox,
        **result,
    }
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved result json: {result_path}")


if __name__ == "__main__":
    main()
