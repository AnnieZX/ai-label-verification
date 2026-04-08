#!/usr/bin/env python3
"""Run a local open-source LVM demo on one downloaded image and fake bbox."""

from __future__ import annotations

import sys
from pathlib import Path

import requests

# Allow `python scripts/demo_open_source_lvm.py` from any working directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.lvm.open_source_verifier import verify_bbox_with_open_source_lvm  # noqa: E402
from src.preprocessing.demo_overlay import draw_single_bbox_overlay  # noqa: E402


def _download_demo_image(target_path: Path) -> Path:
    """Download a lightweight public demo image if not already present."""
    if target_path.exists():
        return target_path

    target_path.parent.mkdir(parents=True, exist_ok=True)
    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"

    response = requests.get(url, timeout=20)
    response.raise_for_status()
    target_path.write_bytes(response.content)
    return target_path


def main() -> None:
    image_path = _download_demo_image(_REPO_ROOT / "data" / "sample" / "demo_image.jpg")
    overlay_path = _REPO_ROOT / "outputs" / "demo_overlay.jpg"

    fake_bbox = [120, 80, 320, 260]
    label = "car"
    model_name = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"

    draw_single_bbox_overlay(
        image_path=str(image_path),
        bbox=fake_bbox,
        output_path=str(overlay_path),
        label=label,
    )

    try:
        result = verify_bbox_with_open_source_lvm(
            image_path=str(image_path),
            bbox=fake_bbox,
            label=label,
            model_name=model_name,
        )
    except RuntimeError as exc:
        print("Open-source LVM demo failed.")
        print(str(exc))
        print("Suggestion: try a smaller/different open-source VLM and re-run.")
        return

    print("=== Open-source LVM demo result ===")
    print(f"image path: {image_path}")
    print(f"bbox: {fake_bbox}")
    print(f"model name: {result.get('model')}")
    print(f"decision: {result.get('decision')}")
    print(f"score: {result.get('score')}")
    print(f"explanation: {result.get('explanation')}")
    print(f"raw_text: {result.get('raw_text')}")
    print(f"overlay image: {overlay_path}")


if __name__ == "__main__":
    main()
