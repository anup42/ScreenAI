#!/usr/bin/env python3
"""Visualize ViP-LLaVA icon annotations by overlaying labels on screenshots."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Tuple

from PIL import Image, ImageDraw, ImageFont

from icon_box_utils import load_icon_boxes


PALETTE = [
    (239, 71, 111),
    (255, 209, 102),
    (6, 214, 160),
    (17, 138, 178),
    (7, 59, 76),
    (255, 133, 27),
    (111, 45, 168),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw icon names next to bounding boxes using ViP-LLaVA annotations.",
    )
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        required=True,
        help="Directory containing *_vip_llava.json annotation files.",
    )
    parser.add_argument(
        "--boxes-dir",
        type=Path,
        required=True,
        help="Directory containing bounding box JSON files.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        required=True,
        help="Directory containing the original screenshots.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where visualized images will be written.",
    )
    parser.add_argument(
        "--boxes-suffix",
        default=".json",
        help="File suffix for bounding box files (default: .json).",
    )
    parser.add_argument(
        "--font-path",
        type=Path,
        default=None,
        help="Optional TrueType font to use for labels.",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=16,
        help="Font size for label text.",
    )
    parser.add_argument(
        "--line-width",
        type=int,
        default=3,
        help="Line width for rectangle outlines.",
    )
    parser.add_argument(
        "--include-id",
        action="store_true",
        help="Include the icon ID alongside the predicted name.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce log verbosity.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    for directory in (args.annotations_dir, args.boxes_dir, args.images_dir):
        if not directory.exists():
            logging.error("Directory %s does not exist", directory)
            return 1

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    font = load_font(args.font_path, args.font_size)

    annotation_files = sorted(
        path for path in args.annotations_dir.iterdir() if path.suffix.lower() == ".json"
    )
    if not annotation_files:
        logging.error("No annotation JSON files found in %s", args.annotations_dir)
        return 1

    for annotation_path in annotation_files:
        payload = load_annotation(annotation_path)
        image_name = payload["image"]
        image_path = args.images_dir / image_name
        if not image_path.exists():
            logging.warning("Skipping %s because image %s is missing", annotation_path.name, image_name)
            continue

        with Image.open(image_path) as pil_image:
            image = pil_image.convert("RGBA")

        boxes_path = (args.boxes_dir / Path(image_name).stem).with_suffix(args.boxes_suffix)
        if not boxes_path.exists():
            logging.warning("Skipping %s because bounding boxes %s are missing", image_name, boxes_path.name)
            continue

        boxes = load_icon_boxes(boxes_path, image.size)
        draw_annotations(image, boxes, payload["annotations"], font, args.line_width, args.include_id)

        target = output_dir / image_path.name
        image.convert("RGB").save(target)

    logging.info("Visualization complete. Files saved to %s", output_dir)
    return 0


def load_annotation(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if "annotations" not in payload:
        raise ValueError(f"Annotation file {path} missing 'annotations' key")
    return payload


def load_font(font_path: Path | None, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if font_path:
        try:
            return ImageFont.truetype(str(font_path), size=size)
        except (OSError, IOError) as exc:
            logging.warning("Failed to load font %s: %s. Falling back to default font.", font_path, exc)
    return ImageFont.load_default()


def draw_annotations(
    image: Image.Image,
    boxes: Dict[str, Tuple[float, float, float, float]],
    annotations: list,
    font: ImageFont.ImageFont,
    line_width: int,
    include_id: bool,
) -> None:
    draw = ImageDraw.Draw(image)
    width, height = image.size

    for index, annotation in enumerate(annotations):
        icon_id = annotation["id"]
        name = annotation["name"]
        bbox = boxes.get(icon_id)
        if bbox is None:
            logging.warning("Skipping %s because no bounding box was found", icon_id)
            continue

        color = PALETTE[index % len(PALETTE)]
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        label = f"{icon_id}: {name}" if include_id else name
        text_w, text_h = measure_text(draw, label, font)
        padding = 3
        text_x = max(0, min(x1, width - text_w - 2 * padding))
        text_y = y1 - text_h - 2 * padding
        if text_y < 0:
            text_y = min(height - text_h - 2 * padding, y1 + 2)
        background = [text_x, text_y, text_x + text_w + 2 * padding, text_y + text_h + 2 * padding]
        draw.rectangle(background, fill=(*color, 160))
        draw.text((text_x + padding, text_y + padding), label, font=font, fill=(0, 0, 0))


def measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    if hasattr(draw, "textbbox"):
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        return right - left, bottom - top
    size = draw.textsize(text, font=font)
    return size[0], size[1]


if __name__ == "__main__":
    raise SystemExit(main())
