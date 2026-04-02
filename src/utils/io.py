"""I/O utilities for loading typed pipeline metadata."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PredictionInput:
    """Structured prediction payload for the pipeline."""

    label: str
    bbox: tuple[float, float, float, float]
    mask: str


@dataclass(frozen=True)
class InputMetadata:
    """Container for image location and normalized prediction data."""

    image_path: str
    prediction: PredictionInput


def parse_prediction(prediction: dict[str, Any]) -> PredictionInput:
    """Convert a raw prediction dictionary into a typed object."""
    raw_bbox = prediction.get("bbox", (0.0, 0.0, 0.0, 0.0))
    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
        raw_bbox = (0.0, 0.0, 0.0, 0.0)

    bbox = tuple(float(value) for value in raw_bbox)
    return PredictionInput(
        label=str(prediction.get("label", "unknown")),
        bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
        mask=str(prediction.get("mask", "")),
    )


def load_input_metadata(image_path: str, prediction: dict[str, Any]) -> InputMetadata:
    """Return normalized metadata used by downstream pipeline stages."""
    return InputMetadata(image_path=image_path, prediction=parse_prediction(prediction))
