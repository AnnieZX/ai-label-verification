#!/usr/bin/env python3
"""Lightweight stats and examples for patch-based prediction JSON (COCO-like / custom).

Usage:
    python analyze_annotations.py
    python analyze_annotations.py data/annotations.json
    python analyze_annotations.py /absolute/path/to/file.json

If omitted, the path defaults to ``data/annotations.json`` next to this script
(i.e. this repo's bundled export).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

# Default export location for this repository (next to repo root = parent of this file).
_DEFAULT_ANNOTATIONS = Path(__file__).resolve().parent / "data" / "annotations.json"



def load_annotations_file(path: Path) -> dict[str, Any]:
    """Load and parse the JSON file; report readable errors to stderr and exit."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"Error: could not read file: {path}\n{exc}", file=sys.stderr)
        sys.exit(1)

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        print(f"Error: invalid JSON in {path}\n{exc}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(data, dict):
        print("Error: root JSON value must be an object.", file=sys.stderr)
        sys.exit(1)

    return data


def get_patches(data: Mapping[str, Any]) -> dict[str, Any]:
    """Return the ``patches`` mapping or exit with an error."""
    patches = data.get("patches")
    if not isinstance(patches, dict):
        print("Error: expected top-level key 'patches' with a dict value.", file=sys.stderr)
        sys.exit(1)
    return patches


def annotation_confidence(annotation: Mapping[str, Any]) -> float | None:
    """Return ``properties.confidence`` if present and numeric; otherwise ``None``."""
    props = annotation.get("properties")
    if not isinstance(props, Mapping):
        return None
    raw = props.get("confidence")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def has_pixel_bbox(annotation: Mapping[str, Any]) -> bool:
    """Return True if ``pixelBbox`` exists and is non-empty."""
    bbox = annotation.get("pixelBbox")
    if bbox is None:
        return False
    if isinstance(bbox, (list, tuple)):
        return len(bbox) > 0
    if isinstance(bbox, dict):
        return bool(bbox)
    return True


def has_segmentation_rle(annotation: Mapping[str, Any]) -> bool:
    """Return True if ``segmentationRLE`` exists and looks present."""
    rle = annotation.get("segmentationRLE")
    if rle is None:
        return False
    if isinstance(rle, str):
        return len(rle.strip()) > 0
    if isinstance(rle, (list, dict)):
        return len(rle) > 0
    return True


def collect_metadata_keys(patches: Mapping[str, Any]) -> set[str]:
    """Union of all metadata keys observed across patches."""
    keys: set[str] = set()
    for patch_obj in patches.values():
        if not isinstance(patch_obj, Mapping):
            continue
        meta = patch_obj.get("metadata")
        if isinstance(meta, Mapping):
            keys.update(str(k) for k in meta.keys())
    return keys


def print_section(title: str) -> None:
    """Print a section header."""
    line = "=" * len(title)
    print(f"\n{line}\n{title}\n{line}")


def suggest_next_steps(
    num_with_mask: int,
    num_with_bbox: int,
    num_annotations: int,
) -> None:
    """Print short, data-driven suggestions."""
    print_section("Next step suggestions")
    if num_annotations == 0:
        print("- No annotations found. Check export path or patch filtering.")
        return

    mask_ratio = num_with_mask / num_annotations
    bbox_ratio = num_with_bbox / num_annotations

    if mask_ratio >= 0.3:
        print(
            f"- Many annotations have masks (~{mask_ratio:.0%}). Consider mask-based "
            "or mask+bbox verification in the pipeline."
        )
    elif bbox_ratio >= 0.5:
        print(
            f"- Mostly bbox coverage (~{bbox_ratio:.0%}). A bbox-first verification "
            "pipeline is a good default; add masks where export allows."
        )
    else:
        print(
            "- Sparse bbox/mask signals. Inspect a few patches manually; ensure "
            "`pixelBbox` / `segmentationRLE` are populated as expected."
        )

    print("- Spot-check low-confidence or high-count patches against images (TIFF URLs in metadata).")


def main() -> None:
    """Parse CLI, analyze patches/annotations, print summary and examples."""
    parser = argparse.ArgumentParser(
        description="Summarize patch-based prediction JSON for debugging."
    )
    parser.add_argument(
        "json_path",
        nargs="?",
        type=Path,
        default=_DEFAULT_ANNOTATIONS,
        help=(
            "Path to annotations JSON (top-level 'patches' dict). "
            f"Default: {_DEFAULT_ANNOTATIONS}"
        ),
    )
    args = parser.parse_args()
    path: Path = args.json_path

    if not path.is_file():
        print(f"Error: not a file: {path}", file=sys.stderr)
        sys.exit(1)

    data = load_annotations_file(path)
    patches = get_patches(data)

    patch_ids = sorted(patches.keys(), key=str)
    total_patches = len(patch_ids)

    total_annotations = 0
    with_bbox = 0
    with_mask = 0
    class_counter: Counter[str] = Counter()
    confidences: list[float] = []
    zero_annotation_patches = 0
    counts_per_patch: list[tuple[str, int]] = []

    for pid in patch_ids:
        patch_obj = patches[pid]
        if not isinstance(patch_obj, Mapping):
            continue

        anns = patch_obj.get("annotations", [])
        if not isinstance(anns, list):
            anns = []

        n = len(anns)
        total_annotations += n
        counts_per_patch.append((str(pid), n))
        if n == 0:
            zero_annotation_patches += 1

        for ann in anns:
            if not isinstance(ann, Mapping):
                continue
            label = ann.get("classLabel")
            if label is not None:
                class_counter[str(label)] += 1

            if has_pixel_bbox(ann):
                with_bbox += 1
            if has_segmentation_rle(ann):
                with_mask += 1

            c = annotation_confidence(ann)
            if c is not None:
                confidences.append(c)

    counts_per_patch.sort(key=lambda x: x[1], reverse=True)
    top5 = counts_per_patch[:5]

    meta_keys_sample = collect_metadata_keys(patches)
    example_patch_ids = patch_ids[: min(5, len(patch_ids))]

    # --- Summary ---
    print_section("Annotation file summary")
    print(f"File: {path.resolve()}")
    print(f"Total patches: {total_patches}")
    print(f"Total annotations: {total_annotations}")
    print(f"Annotations with pixelBbox: {with_bbox}")
    print(f"Annotations with segmentationRLE: {with_mask}")
    print(f"Patches with zero annotations: {zero_annotation_patches}")

    print("\n--- Unique classLabel (top 20 by count) ---")
    if not class_counter:
        print("(no classLabel found)")
    else:
        print(f"  Unique labels: {len(class_counter)}")
        for label, cnt in class_counter.most_common(20):
            print(f"  {label!r}: {cnt}")

    print("\n--- Confidence (properties.confidence) ---")
    if not confidences:
        print("  count: 0 (no numeric confidences found)")
    else:
        print(f"  count: {len(confidences)}")
        print(f"  min:   {min(confidences):.6g}")
        print(f"  max:   {max(confidences):.6g}")
        print(f"  mean:  {statistics.mean(confidences):.6g}")

    print("\n--- Top 5 patches by annotation count ---")
    if not top5:
        print("(none)")
    else:
        for pid, cnt in top5:
            print(f"  {pid!r}: {cnt} annotations")

    print("\n--- Example patch IDs (up to 5) ---")
    if not example_patch_ids:
        print("(none)")
    else:
        for pid in example_patch_ids:
            print(f"  {pid!r}")

    print("\n--- Example metadata keys (union across patches) ---")
    if not meta_keys_sample:
        print("(no metadata keys found)")
    else:
        for key in sorted(meta_keys_sample)[:40]:
            print(f"  {key}")
        if len(meta_keys_sample) > 40:
            print(f"  ... and {len(meta_keys_sample) - 40} more keys")

    # --- Three example patches ---
    example_ids: list[str] = []
    seen: set[str] = set()
    for pid, _ in top5:
        if pid not in seen:
            example_ids.append(pid)
            seen.add(pid)
        if len(example_ids) >= 3:
            break
    for pid in patch_ids:
        if len(example_ids) >= 3:
            break
        if pid not in seen:
            example_ids.append(str(pid))
            seen.add(pid)

    print_section("Example patches (detail)")
    for pid in example_ids[:3]:
        raw = patches.get(pid)
        if not isinstance(raw, Mapping):
            print(f"\nPatch {pid!r}: (invalid patch object, skipped)")
            continue

        anns = raw.get("annotations", [])
        if not isinstance(anns, list):
            anns = []

        meta = raw.get("metadata")
        meta_keys: list[str] = []
        if isinstance(meta, Mapping):
            meta_keys = sorted(str(k) for k in meta.keys())

        print(f"\nPatch id: {pid!r}")
        print(f"  Number of annotations: {len(anns)}")
        print(f"  Metadata keys ({len(meta_keys)}): {', '.join(meta_keys[:12])}")
        if len(meta_keys) > 12:
            print(f"    ... (+{len(meta_keys) - 12} more)")

        for i, ann in enumerate(anns[:2]):
            if not isinstance(ann, Mapping):
                print(f"  Annotation [{i}]: (not an object, skipped)")
                continue
            cl = ann.get("classLabel", "(missing)")
            conf = annotation_confidence(ann)
            conf_str = f"{conf:.6g}" if conf is not None else "(none)"
            print(
                f"  Annotation [{i}]: classLabel={cl!r} "
                f"bbox={has_pixel_bbox(ann)} mask={has_segmentation_rle(ann)} "
                f"confidence={conf_str}"
            )

    suggest_next_steps(
        num_with_mask=with_mask,
        num_with_bbox=with_bbox,
        num_annotations=total_annotations,
    )


if __name__ == "__main__":
    main()


