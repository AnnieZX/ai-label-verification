"""Gemini-based multimodal label verification (API-only, structured JSON)."""

from __future__ import annotations

import json
import re
from typing import Any

from src.lvm.gemini_client import generate_with_image_and_prompt


def _build_verification_prompt(label: str) -> str:
    """Assemble the user prompt for claimed-label verification."""
    return f"""You are verifying whether the highlighted or visually salient region in this image matches the claimed object label.

Claimed label: {label}

Return JSON only:

{{
"decision": "likely_good" | "review" | "suspicious",
"score": 0.0,
"explanation": "short explanation"
}}

Guidelines:

* likely_good = strong visual match
* review = ambiguous
* suspicious = likely incorrect
* score must be between 0 and 1
* keep explanation short"""


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Try parsing a JSON object from model output."""
    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _coerce_decision(value: Any) -> str | None:
    if value is None:
        return None
    v = str(value).strip().lower()
    if v in {"likely_good", "review", "suspicious"}:
        return v
    return None


def _coerce_score(value: Any) -> float | None:
    if value is None:
        return None
    try:
        s = float(value)
    except (TypeError, ValueError):
        return None
    if 0.0 <= s <= 1.0:
        return s
    return None


def verify_label_with_gemini(
    image_path: str,
    label: str,
    *,
    model: str | None = None,
) -> dict[str, Any]:
    """
    Send image + structured prompt to Gemini; return normalized JSON with ``raw_response``.

    On JSON parse failure, returns a safe fallback with ``decision=review`` and
    ``explanation`` / ``raw_response`` set to the raw model text.
    """
    prompt = _build_verification_prompt(label=label)
    raw_text = generate_with_image_and_prompt(image_path, prompt, model=model)

    parsed = _extract_json_object(raw_text)
    if parsed is None:
        return {
            "decision": "review",
            "score": 0.5,
            "explanation": raw_text,
            "raw_response": raw_text,
        }

    decision = _coerce_decision(parsed.get("decision")) or "review"
    score = _coerce_score(parsed.get("score"))
    if score is None:
        score = 0.5
    explanation = str(parsed.get("explanation") or "").strip() or raw_text

    return {
        "decision": decision,
        "score": score,
        "explanation": explanation,
        "raw_response": raw_text,
    }
