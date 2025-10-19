"""Utility helpers for working with icon bounding boxes."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Sequence, Tuple


def load_icon_boxes(
    box_path: Path,
    image_size: Tuple[int, int],
) -> Dict[str, Tuple[float, float, float, float]]:
    """Return a mapping from overlay ID (e.g. ``ID1``) to absolute xyxy bbox."""
    with box_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict) and "boxes" in payload:
        entries: Sequence = payload["boxes"]
    elif isinstance(payload, list):
        entries = payload
    else:
        raise ValueError(f"Unsupported bounding box format in {box_path}")

    boxes: Dict[str, Tuple[float, float, float, float]] = {}
    fallback_id = 1
    for entry in entries:
        if isinstance(entry, dict):
            coords = entry.get("bbox") or entry.get("box") or entry.get("rect") or entry.get("coordinates")
            raw_id = entry.get("id") or entry.get("icon_id") or entry.get("index") or entry.get("label")
        else:
            coords = entry
            raw_id = None

        if coords is None:
            raise ValueError(f"Missing bbox coordinates in entry from {box_path}")

        coord_values = [float(value) for value in coords]
        bbox = resolve_bbox(coord_values, image_size)
        icon_id = parse_icon_id(raw_id, fallback_id)
        boxes[f"ID{icon_id}"] = bbox
        fallback_id += 1

    return boxes


def parse_icon_id(raw_id, fallback: int) -> int:
    if raw_id is None:
        return fallback
    try:
        return int(raw_id)
    except (TypeError, ValueError):
        return fallback


def resolve_bbox(coords: Sequence[float], image_size: Tuple[int, int]) -> Tuple[float, float, float, float]:
    if len(coords) != 4:
        raise ValueError("Bounding box coordinates must have exactly 4 values")

    width, height = image_size
    x1, y1, x2, y2 = coords

    if max(coords) <= 1.0:
        x1 *= width
        y1 *= height
        x2 *= width
        y2 *= height
    elif x2 <= x1 or y2 <= y1:
        w = coords[2]
        h = coords[3]
        x2 = x1 + w
        y2 = y1 + h

    return clamp_bbox((x1, y1, x2, y2), image_size)


def clamp_bbox(
    bbox: Tuple[float, float, float, float],
    image_size: Tuple[int, int],
) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    width, height = image_size
    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))
    return x1, y1, x2, y2


def bbox_to_yolo(
    bbox: Tuple[float, float, float, float],
    image_size: Tuple[int, int],
) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    width, height = image_size
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0
    return cx / width, cy / height, bw / width, bh / height


__all__ = [
    "bbox_to_yolo",
    "clamp_bbox",
    "load_icon_boxes",
    "parse_icon_id",
    "resolve_bbox",
]

