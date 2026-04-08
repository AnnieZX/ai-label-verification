"""Construct prompts and response schemas for vision-language bbox verification."""

from __future__ import annotations

import json


def build_bbox_verification_prompt(label: str | None, bbox: list[float]) -> str:
    """Assemble a strict bbox-verification prompt for multimodal models."""
    if len(bbox) != 4:
        msg = "bbox must have length 4: [x1, y1, x2, y2]"
        raise ValueError(msg)
    normalized_label = str(label) if label is not None else "unknown"
    normalized_bbox = [float(bbox[i]) for i in range(4)]

    _ = json.dumps(build_expected_output_schema(), indent=2)

    return f"""You are a strict evaluator of object detection bounding boxes.

The predicted object label is: {normalized_label}
The bounding box is: {normalized_bbox}

Carefully examine the image and the bounding box.

Your task is to determine whether the bounding box accurately localizes the object.

Strict evaluation criteria:

* The box must tightly enclose the object.
* If the box includes large background regions -> penalize.
* If the box misses parts of the object -> penalize.
* If the box is shifted or not centered on the object -> penalize.

Important:
Do NOT be lenient. If there is any significant issue, choose 'review' or 'suspicious'.

Return ONLY a JSON object with:
{{
"decision": one of ["likely_good", "review", "suspicious"],
"score": float between 0 and 1,
"explanation": short explanation grounded in visual reasoning
}}

Do NOT include the options in the answer.
Do NOT output anything except JSON."""


def build_expected_output_schema() -> dict[str, str]:
    """Describe the JSON object the LVM must return (for prompts and downstream parsing)."""
    return {
        "decision": "likely_good | review | suspicious",
        "score": "float between 0 and 1",
        "explanation": "short natural language explanation",
    }
