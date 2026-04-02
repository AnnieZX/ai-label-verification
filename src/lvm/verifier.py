"""Placeholder vision-language verification logic."""

from dataclasses import dataclass

from src.preprocessing.overlay_builder import OverlayRepresentation


@dataclass(frozen=True)
class VerificationResult:
    """Result of a single label verification pass."""

    decision: str
    score: float
    explanation: str

def verify_with_lvm(overlay_payload: OverlayRepresentation) -> VerificationResult:
    """Return a deterministic placeholder verification result."""
    _ = overlay_payload
    return VerificationResult(
        decision="suspicious",
        score=0.55,
        explanation="Placeholder verifier flagged weak semantic consistency.",
    )
