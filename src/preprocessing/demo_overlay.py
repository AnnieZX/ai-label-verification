"""Simple Pillow overlay utility for demo bbox visualization."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def draw_single_bbox_overlay(
    image_path: str,
    bbox: list[float],
    output_path: str,
    label: str | None = None,
) -> str:
    """Draw one visible bbox (and optional label) onto an image."""
    if len(bbox) != 4:
        msg = "bbox must have 4 values: [x1, y1, x2, y2]"
        raise ValueError(msg)

    x1, y1, x2, y2 = (float(v) for v in bbox)
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    color = (255, 40, 40)
    width = 4
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

    if label:
        text = str(label)
        font = ImageFont.load_default()
        text_bbox = draw.textbbox((x1, y1), text, font=font)
        tw = text_bbox[2] - text_bbox[0]
        th = text_bbox[3] - text_bbox[1]
        tx = max(0.0, x1)
        ty = max(0.0, y1 - th - 6)
        draw.rectangle([tx, ty, tx + tw + 6, ty + th + 4], fill=color)
        draw.text((tx + 3, ty + 2), text, fill=(255, 255, 255), font=font)

    image.save(out_file)
    return str(out_file)
