"""Minimal PNG cropper utilities for orthomosaic patch extraction."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

# Orthomosaic PNGs are intentionally very large in this project; disable
# Pillow's decompression bomb safeguard for this controlled research use case.
Image.MAX_IMAGE_PIXELS = None


def crop_center_patch(image_path: str, output_path: str, patch_size: int = 800) -> str:
    """
    Open a large orthomosaic PNG, crop a centered square patch, save it, and return output_path.
    """
    if patch_size <= 0:
        msg = f"patch_size must be > 0, got {patch_size}"
        raise ValueError(msg)

    with Image.open(image_path) as img:
        width, height = img.size
        if width < patch_size or height < patch_size:
            msg = (
                f"Image is too small for {patch_size}x{patch_size} crop: "
                f"got {width}x{height} at {image_path}"
            )
            raise ValueError(msg)

        left = (width - patch_size) // 2
        top = (height - patch_size) // 2
        right = left + patch_size
        bottom = top + patch_size

        patch = img.crop((left, top, right, bottom))

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    patch.convert("RGB").save(out)
    return str(out)
