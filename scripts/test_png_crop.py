#!/usr/bin/env python3
"""Simple SSH-side test for center-cropping an orthomosaic PNG."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from any working directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.preprocessing.png_cropper import crop_center_patch  # noqa: E402


def main() -> None:
    input_image = "/deac/csc/yangGrp/cuij/GoldMining/Data/Orthomosaic/Anel.png"
    output_patch = _REPO_ROOT / "outputs" / "test_patch.jpg"
    patch_size = 800

    output_patch.parent.mkdir(parents=True, exist_ok=True)
    out_path = crop_center_patch(
        image_path=input_image,
        output_path=str(output_patch),
        patch_size=patch_size,
    )

    print(f"input image path: {input_image}")
    print(f"output patch path: {out_path}")
    print(f"patch size: {patch_size}")


if __name__ == "__main__":
    main()
