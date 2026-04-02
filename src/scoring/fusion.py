"""Score fusion logic for combining multiple evidence signals."""


from dataclasses import dataclass


@dataclass(frozen=True)
class FusionResult:
    """Final fused score and high-level decision."""

    final_score: float
    decision: str


def fuse_scores(
    lvm_score: float,
    geometry_score: float,
    stability_score: float,
    lvm_weight: float = 0.5,
    geometry_weight: float = 0.2,
    stability_weight: float = 0.3,
) -> FusionResult:
    """Combine scores with a weighted average and basic safety checks."""
    total_weight = lvm_weight + geometry_weight + stability_weight
    if total_weight <= 0:
        return FusionResult(final_score=0.0, decision="suspicious")

    final_score = (
        lvm_score * lvm_weight
        + geometry_score * geometry_weight
        + stability_score * stability_weight
    ) / total_weight
    decision = "verified" if final_score >= 0.7 else "suspicious"
    return FusionResult(final_score=final_score, decision=decision)
