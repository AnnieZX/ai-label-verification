"""Geometry feature extraction from bounding-box predictions."""

from dataclasses import dataclass

from src.utils.io import PredictionInput


@dataclass(frozen=True)
class GeometryFeatures:
    """Simple geometric properties derived from a bbox."""

    width: float
    height: float
    area: float
    score: float

def extract_geometry_features(prediction: PredictionInput) -> GeometryFeatures:
    """Compute basic bbox features and a lightweight geometry confidence score."""
    x_min, y_min, x_max, y_max = prediction.bbox
    width = max(0.0, x_max - x_min)
    height = max(0.0, y_max - y_min)
    area = width * height
    # Placeholder normalization: maps bbox area to [0, 1] with a soft cap.
    score = min(area / 10000.0, 1.0)
    return GeometryFeatures(width=width, height=height, area=area, score=score)
