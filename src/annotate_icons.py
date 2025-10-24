#!/usr/bin/env python3
"""Annotate UI icons in screenshots using Qwen2.5-VL."""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForVision2Seq, AutoProcessor, PreTrainedModel

try:
    from transformers import BitsAndBytesConfig  # type: ignore
except ImportError:  # pragma: no cover
    BitsAndBytesConfig = None  # type: ignore


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Qwen/Qwen2.5-VL-72B-Instruct to assign names to icons "
            "identified by bounding boxes in screenshots."
        )
    )
    parser.add_argument(
        "--screenshots-dir",
        type=Path,
        required=True,
        help="Directory containing screenshot images to annotate",
    )
    parser.add_argument(
        "--boxes-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing bounding box JSON files. Defaults to the "
            "screenshots directory."
        ),
    )
    parser.add_argument(
        "--boxes-suffix",
        default=".json",
        help="File suffix for bounding box files (default: .json)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where annotation JSON files will be written",
    )
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen2.5-VL-72B-Instruct",
        help="Hugging Face model identifier to use",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Device map argument passed to transformers (default: auto)",
    )
    parser.add_argument(
        "--torch-dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Torch dtype for model weights when not using quantization",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load the model using 4-bit quantization (requires bitsandbytes)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=4,
        help="Maximum number of tokens to generate per icon",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. Use 0 for deterministic decoding",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling probability mass (only when temperature > 0)",
    )
    parser.add_argument(
        "--crop-padding",
        type=float,
        default=0.05,
        help="Extra padding ratio applied around each bounding box crop",
    )
    parser.add_argument(
        "--system-prompt",
        default=(
            "You are an expert UI analyst. Name each user interface icon "
            "concisely and accurately."
        ),
        help="System prompt injected before each request",
    )
    parser.add_argument(
        "--user-prompt",
        default=(
            "Icon {icon_id} is shown in the attached crop from screenshot "
            "'{screenshot_name}'. Provide a short lowercase name for the icon."
        ),
        help="User prompt template. Placeholders: {icon_id}, {screenshot_name}",
    )
    parser.add_argument(
        "--overlay-mode",
        action="store_true",
        help=(
            "Describe numbered icons directly from screenshots that already have "
            "bounding boxes drawn. Skips external bounding box JSON files."
        ),
    )
    parser.add_argument(
        "--overlay-user-prompt",
        default=(
            "The screenshot '{screenshot_name}' contains multiple icons highlighted "
            "with numbered boxes. For each visible number, provide a short lowercase "
            "name for the icon in the format 'ID: name'. Use one line per ID."
        ),
        help="User prompt template used when --overlay-mode is enabled.",
    )
    parser.add_argument(
        "--lowercase-output",
        dest="lowercase_output",
        action="store_true",
        default=True,
        help="Force model responses to lowercase (default)",
    )
    parser.add_argument(
        "--no-lowercase-output",
        dest="lowercase_output",
        action="store_false",
        help="Keep model responses without lowercasing",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce logging verbosity",
    )
    return parser.parse_args()


@dataclass
class IconRegion:
    icon_id: int
    bbox: Tuple[int, int, int, int]

    def padded(self, padding: float, image_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
        if padding <= 0:
            return self.bbox
        x1, y1, x2, y2 = self.bbox
        width, height = image_size
        pad_w = int(round((x2 - x1) * padding))
        pad_h = int(round((y2 - y1) * padding))
        return (
            max(0, x1 - pad_w),
            max(0, y1 - pad_h),
            min(width, x2 + pad_w),
            min(height, y2 + pad_h),
        )


def discover_images(directory: Path) -> List[Path]:
    return [path for path in sorted(directory.iterdir()) if path.suffix.lower() in IMAGE_EXTENSIONS]


def load_icon_regions(bbox_path: Path, image_size: Tuple[int, int]) -> List[IconRegion]:
    with bbox_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict) and "boxes" in payload:
        raw_boxes: Sequence = payload["boxes"]
    elif isinstance(payload, list):
        raw_boxes = payload
    else:
        raise ValueError(f"Unsupported bounding box format in {bbox_path}")

    regions: List[IconRegion] = []
    fallback_id = 1
    for entry in raw_boxes:
        if isinstance(entry, dict):
            coords = entry.get("bbox") or entry.get("box") or entry.get("rect") or entry.get("coordinates")
            raw_id = entry.get("id") or entry.get("icon_id") or entry.get("index") or entry.get("label")
        else:
            coords = entry
            raw_id = None

        if coords is None:
            raise ValueError(f"Missing bbox coordinates in entry from {bbox_path}")

        coord_values = [float(value) for value in coords]
        bbox = resolve_bbox(coord_values, image_size)
        icon_id = parse_icon_id(raw_id, fallback_id)
        regions.append(IconRegion(icon_id=icon_id, bbox=bbox))
        fallback_id += 1

    return regions


def parse_icon_id(raw_id, fallback: int) -> int:
    if raw_id is None:
        return fallback
    try:
        return int(raw_id)
    except (TypeError, ValueError):
        return fallback


def resolve_bbox(coords: Sequence[float], image_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
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


def clamp_bbox(bbox: Tuple[float, float, float, float], image_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
    width, height = image_size
    x1 = int(round(max(0.0, min(float(width), bbox[0]))))
    y1 = int(round(max(0.0, min(float(height), bbox[1]))))
    x2 = int(round(max(0.0, min(float(width), bbox[2]))))
    y2 = int(round(max(0.0, min(float(height), bbox[3]))))
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return (x1, y1, x2, y2)


def build_conversation(system_prompt: str, user_prompt: str, icon: IconRegion, screenshot: Path) -> List[Dict[str, object]]:
    user_message = user_prompt.format(icon_id=icon.icon_id, screenshot_name=screenshot.name).strip()
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_message},
            ],
        },
    ]


def build_overlay_conversation(system_prompt: str, user_prompt: str, screenshot: Path) -> List[Dict[str, object]]:
    message = user_prompt.format(screenshot_name=screenshot.name).strip()
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": message},
            ],
        },
    ]


def _string_to_device(device: object) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if isinstance(device, str):
        try:
            return torch.device(device)
        except (RuntimeError, TypeError):
            if device == "disk":
                return torch.device("cpu")
            raise
    if isinstance(device, int):
        return torch.device(f"cuda:{device}")
    raise TypeError(f"Unsupported device entry {device!r}")


def _resolve_modal_device_map(model: PreTrainedModel) -> Tuple[torch.device, Dict[str, torch.device]]:
    """Return the default device and per-modality overrides.

    When Accelerate shards the model across multiple GPUs the ``hf_device_map``
    attribute indicates where every module was placed.  The annotation pipeline
    has to move the text tokens and the pixel values to the matching device
    before calling ``generate``; otherwise PyTorch keeps everything on the
    first GPU.
    """
    
    print(getattr(model, "hf_device_map", None))

    device_map = getattr(model, "hf_device_map", None)
    if not isinstance(device_map, dict) or not device_map:
        default_device = resolve_model_device(model)
        return default_device, {"text": default_device, "vision": default_device}

    default_device = _string_to_device(next(iter(device_map.values())))
    text_device: torch.device | None = None
    vision_device: torch.device | None = None

    for module_name, raw_device in device_map.items():
        device = _string_to_device(raw_device)
        lowered = module_name.lower()
        if text_device is None and any(
            token in lowered for token in ("embed_tokens", "language", "text_model", "lm_head")
        ):
            text_device = device
        if vision_device is None and any(
            token in lowered for token in ("vision", "visual", "clip", "image")
        ):
            vision_device = device

    modal_devices = {
        "text": text_device or default_device,
        "vision": vision_device or default_device,
    }
    return default_device, modal_devices


def load_model_and_processor(args: argparse.Namespace):
    dtype_lookup = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    model_kwargs: Dict[str, object] = {
        "device_map": args.device_map,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }

    if args.load_in_4bit:
        if BitsAndBytesConfig is None:
            raise RuntimeError(
                "bitsandbytes is not available. Install it or remove --load-in-4bit."
            )
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        model_kwargs["torch_dtype"] = dtype_lookup[args.torch_dtype]

    model = AutoModelForVision2Seq.from_pretrained(args.model_name, **model_kwargs)
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    model.eval()
    return model, processor


def resolve_model_device(model: PreTrainedModel) -> torch.device:
    device_attr = getattr(model, "device", None)
    if isinstance(device_attr, torch.device):
        return device_attr
    if isinstance(device_attr, str):
        try:
            return torch.device(device_attr)
        except RuntimeError:
            pass
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict) and device_map:
        first_device = next(iter(device_map.values()))
        if isinstance(first_device, str):
            try:
                return torch.device(first_device)
            except (RuntimeError, TypeError):
                if first_device == "disk":
                    return torch.device("cpu")
                raise
        if isinstance(first_device, int):
            return torch.device(f"cuda:{first_device}")
    return torch.device("cpu")


def prepare_inputs(
    processor: AutoProcessor,
    conversation: List[Dict[str, object]],
    image: Image.Image,
    default_device: torch.device,
    modal_devices: Dict[str, torch.device],
) -> Dict[str, torch.Tensor]:
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[image], return_tensors="pt")
    tensor_inputs: Dict[str, torch.Tensor] = {}
    for key, value in inputs.items():
        if not isinstance(value, torch.Tensor):
            tensor_inputs[key] = value
            continue

        key_lower = key.lower()
        if any(token in key_lower for token in ("pixel", "image")):
            device = modal_devices.get("vision", default_device)
        else:
            device = modal_devices.get("text", default_device)
        tensor_inputs[key] = value.to(device)
    return tensor_inputs


def generate_label(
    model: PreTrainedModel,
    processor: AutoProcessor,
    inputs: Dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> str:
    eos_token_id = processor.tokenizer.eos_token_id if processor.tokenizer is not None else model.config.eos_token_id
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "temperature": args.temperature if args.temperature > 0 else None,
        "top_p": args.top_p,
        "pad_token_id": eos_token_id,
        "eos_token_id": eos_token_id,
    }
    generation_kwargs = {key: value for key, value in generation_kwargs.items() if value is not None}

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, **generation_kwargs)

    input_length = inputs["input_ids"].shape[-1]
    new_tokens = generated_ids[:, input_length:].to("cpu")
    decoded = processor.batch_decode(
        new_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0]
    return decoded.strip()


def normalise_label(text: str, lowercase: bool) -> str:
    cleaned = text.strip()
    if not cleaned:
        return cleaned
    cleaned = cleaned.splitlines()[0].strip()
    lower = cleaned.lower()
    prefixes = [
        "icon:",
        "icon is",
        "the icon is",
        "this icon is",
        "this is",
        "it is",
        "icon",
    ]
    for prefix in prefixes:
        if lower.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip(" .:-")
            lower = cleaned.lower()
            break
    if ":" in cleaned:
        cleaned = cleaned.split(":", 1)[1].strip()
    cleaned = cleaned.strip(" .,'\"`")
    cleaned = " ".join(cleaned.split())
    if lowercase:
        cleaned = cleaned.lower()
    return cleaned


def annotate_screenshot(
    image_path: Path,
    regions: List[IconRegion],
    model: PreTrainedModel,
    processor: AutoProcessor,
    args: argparse.Namespace,
    model_device: torch.device,
    modal_devices: Dict[str, torch.device],
) -> List[Dict[str, object]]:
    annotations: List[Dict[str, object]] = []
    with Image.open(image_path) as pil_image:
        image = pil_image.convert("RGB")
        for icon in regions:
            padded_bbox = icon.padded(args.crop_padding, image.size)
            crop = image.crop(padded_bbox)
            conversation = build_conversation(args.system_prompt, args.user_prompt, icon, image_path)
            inputs = prepare_inputs(
                processor,
                conversation,
                crop,
                model_device,
                modal_devices,
            )
            raw_label = generate_label(model, processor, inputs, args)
            label = normalise_label(raw_label, args.lowercase_output)
            annotations.append(
                {
                    "id": icon.icon_id,
                    "bbox": icon.bbox,
                    "padded_bbox": list(padded_bbox),
                    "raw_response": raw_label,
                    "label": label,
                }
            )
    return annotations


def annotate_overlay_screenshot(
    image_path: Path,
    model: PreTrainedModel,
    processor: AutoProcessor,
    args: argparse.Namespace,
    model_device: torch.device,
    modal_devices: Dict[str, torch.device],
) -> Dict[str, str]:
    with Image.open(image_path) as pil_image:
        image = pil_image.convert("RGB")

    conversation = build_overlay_conversation(args.system_prompt, args.overlay_user_prompt, image_path)
    inputs = prepare_inputs(
        processor,
        conversation,
        image,
        model_device,
        modal_devices,
    )
    raw_response = generate_label(model, processor, inputs, args)
    cleaned_response = raw_response.strip()
    if args.lowercase_output:
        cleaned_response = cleaned_response.lower()
    return {"raw_response": raw_response, "response": cleaned_response}


def save_annotations(output_dir: Path, image_path: Path, results: List[Dict[str, object]], model_name: str) -> None:
    payload = {
        "image": image_path.name,
        "model": model_name,
        "annotations": sorted(results, key=lambda item: item["id"]),
    }
    target = output_dir / f"{image_path.stem}_labels.json"
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def save_overlay_annotation(
    output_dir: Path, image_path: Path, result: Dict[str, str], model_name: str
) -> None:
    payload = {
        "image": image_path.name,
        "model": model_name,
        "mode": "overlay",
        "raw_response": result["raw_response"],
        "response": result["response"],
    }
    target = output_dir / f"{image_path.stem}_labels.json"
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if not args.screenshots_dir.exists():
        logging.error("Screenshots directory %s does not exist", args.screenshots_dir)
        return 1

    boxes_dir = args.boxes_dir or args.screenshots_dir
    if not args.overlay_mode and not boxes_dir.exists():
        logging.error("Bounding boxes directory %s does not exist", boxes_dir)
        return 1

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    images = discover_images(args.screenshots_dir)
    if not images:
        logging.error("No images found in %s", args.screenshots_dir)
        return 1

    logging.info("Loading model %s", args.model_name)
    model, processor = load_model_and_processor(args)
    model_device, modal_devices = _resolve_modal_device_map(model)
    if getattr(model, "hf_device_map", None):
        logging.info("Model sharded across devices: %s", model.hf_device_map)
    else:
        logging.info("Model loaded on device %s", model_device)

    for image_path in tqdm(images, desc="Annotating screenshots"):
        if args.overlay_mode:
            logging.info("Annotating %s using overlay mode", image_path.name)
            result = annotate_overlay_screenshot(
                image_path,
                model,
                processor,
                args,
                model_device,
                modal_devices,
            )
            save_overlay_annotation(output_dir, image_path, result, args.model_name)
            continue

        bbox_path = (boxes_dir / image_path.stem).with_suffix(args.boxes_suffix)
        if not bbox_path.exists():
            logging.warning("Skipping %s because %s is missing", image_path.name, bbox_path.name)
            continue
        try:
            with Image.open(image_path) as tmp_image:
                regions = load_icon_regions(bbox_path, tmp_image.size)
        except Exception as exc:  # pragma: no cover
            logging.error("Failed loading bounding boxes for %s: %s", image_path.name, exc)
            continue

        if not regions:
            logging.warning("No bounding boxes found for %s", image_path.name)
            continue

        logging.info("Annotating %s (%d icons)", image_path.name, len(regions))
        results = annotate_screenshot(
            image_path,
            regions,
            model,
            processor,
            args,
            model_device,
            modal_devices,
        )
        save_annotations(output_dir, image_path, results, args.model_name)

    logging.info("Annotation complete. Files saved to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
