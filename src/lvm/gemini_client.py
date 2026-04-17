"""Gemini API client: remote multimodal calls only (no local model weights)."""

from __future__ import annotations

import os
from pathlib import Path

from google import genai
from google.genai import types


def _mime_type_for_path(image_path: str) -> str:
    """Return a MIME type for common image extensions."""
    suffix = Path(image_path).suffix.lower()
    mapping = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }
    return mapping.get(suffix, "image/jpeg")


def generate_with_image_and_prompt(
    image_path: str,
    prompt: str,
    *,
    api_key: str | None = None,
    model: str | None = None,
) -> str:
    """
    Send image bytes + text prompt to the Gemini API and return raw text.

    Uses ``GEMINI_API_KEY`` when ``api_key`` is omitted, and ``GEMINI_MODEL``
    when ``model`` is omitted (default ``gemini-2.5-pro``).
    """
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key or not str(key).strip():
        msg = "GEMINI_API_KEY is not set or is empty."
        raise ValueError(msg)

    model_id = model or os.environ.get("GEMINI_MODEL", "gemini-2.5-pro")

    path = Path(image_path)
    if not path.is_file():
        msg = f"Image file not found: {image_path}"
        raise FileNotFoundError(msg)

    image_bytes = path.read_bytes()
    mime = _mime_type_for_path(str(path))

    # Remote API only: ``genai.Client`` talks to Google's servers; no local model download.
    client = genai.Client(api_key=key)

    try:
        response = client.models.generate_content(
            model=model_id,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime),
                prompt,
            ],
        )
    except Exception as exc:  # noqa: BLE001
        msg = f"Gemini API request failed: {exc}"
        raise RuntimeError(msg) from exc

    text = getattr(response, "text", None)
    if text is None and response is not None:
        try:
            cands = getattr(response, "candidates", None) or []
            if cands:
                parts = getattr(cands[0].content, "parts", None) or []
                text = "".join(getattr(p, "text", "") or "" for p in parts)
        except (AttributeError, IndexError, TypeError):
            text = None

    if not text or not str(text).strip():
        msg = "Gemini API returned empty or invalid response text."
        raise RuntimeError(msg)

    return str(text).strip()
