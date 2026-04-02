"""Placeholder temporal or perturbation stability scoring."""


from dataclasses import dataclass


@dataclass(frozen=True)
class StabilityResult:
    """Placeholder output for stability estimation."""

    score: float


def compute_stability_score() -> StabilityResult:
    """Return a fixed placeholder stability score."""
    return StabilityResult(score=0.65)
