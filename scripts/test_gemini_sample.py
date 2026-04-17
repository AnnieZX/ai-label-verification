#!/usr/bin/env python3
"""End-to-end Gemini multimodal verifier demo using a local sample image."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.lvm.gemini_verifier import verify_label_with_gemini  # noqa: E402


def _pick_sample_image(repo_root: Path) -> Path:
    """Use the first existing sample among known paths."""
    candidates = [
        repo_root / "outputs" / "test_patch.jpg",
        repo_root / "outputs" / "demo_overlay.jpg",
        repo_root / "outputs" / "debug_bbox.png",
    ]
    for p in candidates:
        if p.is_file():
            return p
    names = ", ".join(str(p.relative_to(repo_root)) for p in candidates)
    msg = f"No sample image found. Expected one of: {names}"
    raise FileNotFoundError(msg)


def main() -> None:
    image_path = _pick_sample_image(_REPO_ROOT)
    out_path = _REPO_ROOT / "outputs" / "lvm_results" / "gemini_sample_result.json"
    label = "item"

    result = verify_label_with_gemini(str(image_path), label=label)

    print(f"decision: {result.get('decision')}")
    print(f"score: {result.get('score')}")
    print(f"explanation: {result.get('explanation')}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "image_path": str(image_path),
        "label": label,
        **result,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
