"""Construct a placeholder overlay representation from input metadata."""

from dataclasses import dataclass

from src.utils.io import InputMetadata, PredictionInput


@dataclass(frozen=True)
class OverlayRepresentation:
    """Serializable overlay payload used by verification and feature stages."""

    image_path: str
    prediction: PredictionInput
    overlay: str
    crop: str


def build_overlay(metadata: InputMetadata) -> OverlayRepresentation:
    """Create placeholder overlay artifacts from input metadata."""
    return OverlayRepresentation(
        image_path=metadata.image_path,
        prediction=metadata.prediction,
        overlay="dummy_overlay",
        crop="dummy_crop",
    )
