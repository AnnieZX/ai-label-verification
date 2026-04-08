"""Minimal end-to-end pipeline runner for AI label verification."""

import json
from dataclasses import dataclass, asdict
from typing import Any

from src.features.geometry import extract_geometry_features, geometry_score
from src.features.stability import compute_stability_score
from src.lvm.verifier import verify_with_lvm
from src.preprocessing.overlay_builder import build_overlay
from src.scoring.fusion import fuse_scores
from src.utils.io import load_input_metadata


@dataclass(frozen=True)
class PipelineResult:
    """Final structured output for the minimal verification pipeline."""

    image_path: str
    label: str
    lvm_score: float
    geometry_score: float
    stability_score: float
    final_score: float
    decision: str
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable pipeline result."""
        return asdict(self)


def run_pipeline(image_path: str, prediction: dict[str, Any]) -> PipelineResult:
    """Run the placeholder pipeline stages in order."""
    metadata = load_input_metadata(image_path=image_path, prediction=prediction)
    overlay = build_overlay(metadata)
    verification = verify_with_lvm(overlay)
    flat_prediction: dict[str, Any] = {
        "bbox": list(metadata.prediction.bbox),
        "label": metadata.prediction.label,
        "confidence": None,
        "mask_rle": None,
    }
    geometry_feats = extract_geometry_features(flat_prediction, image_size=(800, 800))
    geom_scr = geometry_score(geometry_feats)
    stability = compute_stability_score()
    fusion = fuse_scores(
        lvm_score=verification.score,
        geometry_score=geom_scr,
        stability_score=stability.score,
    )

    explanation = (
        f"{verification.explanation} "
        f"Geometry area={geometry_feats['bbox_area']:.2f}, stability={stability.score:.2f}."
    )

    return PipelineResult(
        image_path=metadata.image_path,
        label=metadata.prediction.label,
        lvm_score=round(verification.score, 4),
        geometry_score=round(geom_scr, 4),
        stability_score=round(stability.score, 4),
        final_score=round(fusion.final_score, 4),
        decision=fusion.decision,
        explanation=explanation,
    )


def main() -> None:
    """Create dummy input and run the minimal placeholder pipeline."""
    dummy_image_path = "data/dummy_image.jpg"
    dummy_prediction = {
        "label": "cat",
        "bbox": [10, 20, 110, 140],
        "mask": "dummy_mask_placeholder",
    }

    result = run_pipeline(image_path=dummy_image_path, prediction=dummy_prediction)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
