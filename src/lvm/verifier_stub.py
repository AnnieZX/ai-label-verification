"""Stub LVM verifier: builds prompts and returns placeholder results (no network I/O)."""

from __future__ import annotations

from typing import Any

from src.lvm.prompt_builder import build_bbox_verification_prompt


def verify_bbox_with_lvm(
    image_path: str,
    bbox: list[float],
    label: str | None,
    provider: str = "gemini",
    model: str | None = None,
) -> dict[str, Any]:
    """Run the bbox verification flow against a vision-language model.

    Today this returns a fixed **stub** payload so callers can integrate the interface
    before any real API keys or SDKs are wired in.

    Parameters
    ----------
    image_path:
        Path or URI to the image (patch or full frame). Not read in the stub.
    bbox:
        ``[x1, y1, x2, y2]`` in pixel coordinates.
    label:
        Predicted class label, if any.
    provider:
        Logical backend id, e.g. ``"gemini"``, ``"openai"``.
    model:
        Model id when using a real provider; unused in the stub.

    Returns
    -------
    dict
        Normalized result including prompt text and a placeholder decision/score.
    """
    prompt = build_bbox_verification_prompt(label=label, bbox=bbox)

    # -------------------------------------------------------------------------
    # Future: real LVM call sites (no imports / keys here yet)
    #
    # * Gemini (generateContent + inline image bytes or Files API):
    #     # from google import genai
    #     # client = genai.Client(api_key=...)
    #     # response = client.models.generate_content(
    #     #     model=model or "gemini-2.0-flash",
    #     #     contents=[prompt, image_part],
    #     # )
    #     # raw = response.text or response.candidates[0].content.parts[0].text
    #
    # * OpenAI Responses API (multimodal input):
    #     # import openai
    #     # client = openai.OpenAI(api_key=...)
    #     # response = client.responses.create(
    #     #     model=resolved_model,
    #     #     input=[{"role": "user", "content": [{"type": "input_text", "text": prompt},
    #     #                                          {"type": "input_image", ...}}]}],
    #     # )
    #     # raw = response.output_text
    #
    # Then: parse JSON from `raw`, validate keys (decision/score/explanation),
    # map errors to a safe fallback dict.
    # -------------------------------------------------------------------------

    _ = image_path  # image will be attached in real implementations

    return {
        "provider": provider,
        "model": model,
        "prompt": prompt,
        "decision": "review",
        "score": 0.55,
        "explanation": "Stub response. Replace with real LVM API call later.",
        "raw_response": None,
    }
