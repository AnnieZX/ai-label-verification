"""Local open-source LVM verifier using Hugging Face Transformers."""

from __future__ import annotations

import json
import re
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from src.lvm.prompt_builder import build_bbox_verification_prompt


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Try parsing a JSON object from full model text."""
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
        return float(value)
    except (TypeError, ValueError):
        return None


def _fallback_parse(raw_text: str) -> tuple[str, float | None, str]:
    """Fallback parser when model output is not valid JSON."""
    lowered = raw_text.lower()
    decision = "review"
    if "suspicious" in lowered:
        decision = "suspicious"
    elif "likely_good" in lowered:
        decision = "likely_good"
    elif "review" in lowered:
        decision = "review"

    score_match = re.search(r"\b(?:score|confidence)\s*[:=]?\s*([01](?:\.\d+)?)\b", lowered)
    score = float(score_match.group(1)) if score_match else None
    return decision, score, raw_text.strip()


def verify_bbox_with_open_source_lvm(
    image_path: str,
    bbox: list[float],
    label: str | None = None,
    model_name: str = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
) -> dict[str, Any]:
    """Run local open-source LVM bbox verification on one image and bbox.

    Returns normalized fields even when generation/parsing is imperfect.
    Raises RuntimeError with a clear message when the model cannot load.
    """
    prompt = build_bbox_verification_prompt(label=label, bbox=bbox)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    try:
        image = Image.open(image_path).convert("RGB")
    except OSError as exc:
        msg = f"Could not open image: {image_path}"
        raise RuntimeError(msg) from exc

    try:
        # Model + processor loading happens here (first run will download weights).
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=dtype,
        )
        model.to(device)
        model.eval()
    except Exception as exc:  # noqa: BLE001
        msg = (
            f"Failed to load open-source VLM '{model_name}'. "
            "Try a smaller or different model, and ensure transformers/torch are installed."
        )
        raise RuntimeError(msg) from exc

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    try:
        prompt_text = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = processor(
            images=image,
            text=prompt_text,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Keep generation short and deterministic for demo stability.
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=180,
                do_sample=False,
                temperature=0.0,
            )

        raw_text = processor.decode(output_ids[0], skip_special_tokens=True).strip()
    except Exception as exc:  # noqa: BLE001
        msg = "Model inference failed while generating a response."
        raise RuntimeError(msg) from exc

    parsed_json = _extract_json_object(raw_text)
    if parsed_json is not None:
        decision = _coerce_decision(parsed_json.get("decision")) or "review"
        score = _coerce_score(parsed_json.get("score"))
        explanation = str(parsed_json.get("explanation") or raw_text).strip()
    else:
        decision, score, explanation = _fallback_parse(raw_text)

    return {
        "model": model_name,
        "decision": decision,
        "score": score,
        "explanation": explanation,
        "raw_text": raw_text,
    }
