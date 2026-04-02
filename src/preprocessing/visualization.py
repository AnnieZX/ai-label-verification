"""Draw bounding boxes on images using NumPy + Matplotlib."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from typing import Any


def create_dummy_image(size: tuple[int, int] = (800, 800)) -> np.ndarray:
    """Return a blank RGB image (white background), shape ``(H, W, 3)``, ``uint8``."""
    height, width = int(size[0]), int(size[1])
    return np.full((height, width, 3), 255, dtype=np.uint8)


def _parse_bbox(record: dict[str, Any]) -> tuple[float, float, float, float] | None:
    raw = record.get("bbox")
    if raw is None or not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]))
    except (TypeError, ValueError):
        return None
    return x1, y1, x2, y2


def _random_color(rng: np.random.Generator) -> tuple[float, float, float]:
    """RGB in [0, 1], biased away from very dark colors."""
    return tuple(float(v) for v in rng.uniform(0.15, 0.95, size=3))


def draw_bboxes(image: np.ndarray, records: list[dict[str, Any]]) -> np.ndarray:
    """Draw axis-aligned boxes and optional labels on a copy of ``image``.

    Parameters
    ----------
    image:
        RGB image, shape ``(H, W, 3)``, typically ``uint8``.
    records:
        Annotation dicts; uses ``bbox`` as ``[x1, y1, x2, y2]`` and optional ``label``.

    Returns
    -------
    np.ndarray
        New RGB ``uint8`` image of the same shape as ``image``.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        msg = f"image must have shape (H, W, 3); got {image.shape}"
        raise ValueError(msg)

    h, w = image.shape[0], image.shape[1]
    dpi = 100
    fig, ax = plt.subplots(
        figsize=(w / dpi, h / dpi),
        dpi=dpi,
    )
    ax.imshow(image)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    plt.subplots_adjust(0, 0, 1, 1)

    rng = np.random.default_rng()

    for rec in records:
        parsed = _parse_bbox(rec)
        if parsed is None:
            continue
        x1, y1, x2, y2 = parsed
        color = _random_color(rng)
        width_px = x2 - x1
        height_px = y2 - y1
        rect = Rectangle(
            (x1, y1),
            width_px,
            height_px,
            linewidth=2,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)

        label = rec.get("label")
        if label is not None and str(label).strip():
            ax.text(
                x1,
                y1 - 4,
                str(label),
                color=color,
                fontsize=9,
                verticalalignment="bottom",
                clip_on=True,
            )

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)

    out = rgba[:, :, :3].copy()
    if out.dtype != np.uint8:
        out = np.clip(out, 0, 255).astype(np.uint8)
    return out


if __name__ == "__main__":
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    out_dir = root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    img = create_dummy_image((800, 800))
    demo_records: list[dict[str, Any]] = [
        {"bbox": [50, 50, 200, 180], "label": "a"},
        {"bbox": [320, 100, 520, 260], "label": "b"},
        {"bbox": [600, 400, 750, 700], "label": "c"},
        {"bbox": [100, 500, 400, 650], "label": "d"},
        {"bbox": [450, 30, 700, 200], "label": "e"},
    ]
    result = draw_bboxes(img, demo_records)
    out_path = out_dir / "debug_bbox.png"
    plt.imsave(out_path, result)
    print(f"Wrote {out_path}")
