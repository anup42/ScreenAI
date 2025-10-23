#!/usr/bin/env python3
"""Annotate overlay UI icons in screenshots using ViP-LLaVA models."""
from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, PreTrainedModel, VipLlavaForConditionalGeneration

try:
    from transformers import BitsAndBytesConfig  # type: ignore
except ImportError:  # pragma: no cover
    BitsAndBytesConfig = None  # type: ignore


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
MODEL_ALIASES = {
    "7b": "llava-hf/vip-llava-7b-hf",
    "13b": "llava-hf/vip-llava-13b-hf",
}
DEFAULT_PROMPT = """You are an assistant that identifies UI icons.
The image has icons labeled with IDs.
Please respond in the format:
ID1: [icon name]
ID2: [icon name]
ID3: [icon name]"""


@dataclass
class AnnotationResult:
    image: str
    model: str
    prompt: str
    raw_response: str
    annotations: Dict[str, str]
    unparsed_lines: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Describe overlay-labeled UI icons using ViP-LLaVA 7B or 13B."
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        required=True,
        help="Directory containing images with embedded bounding boxes and IDs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where JSON annotation files will be written.",
    )
    parser.add_argument(
        "--model-size",
        choices=sorted(MODEL_ALIASES),
        default="7b",
        help="Model size to load (maps to official ViP-LLaVA checkpoints).",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Optional explicit Hugging Face repository name. Overrides --model-size.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Optional local path to a ViP-LLaVA model. Overrides --model-id and --model-size.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Primary device to run inference on (default: cuda if available, else cpu).",
    )
    parser.add_argument(
        "--device-map",
        default=None,
        help="Optional accelerate device map string/dict accepted by transformers.",
    )
    parser.add_argument(
        "--torch-dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Torch dtype for model weights unless 4-bit quantization is used.",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load the model using 4-bit quantization (requires bitsandbytes).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=192,
        help="Maximum number of tokens to generate for each image.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. Use 0 for deterministic decoding.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling probability mass (used only when temperature > 0).",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt presented to the model when describing the icons.",
    )
    parser.add_argument(
        "--lowercase-output",
        action="store_true",
        help="Lowercase the parsed icon names before saving.",
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

    if not args.images_dir.exists():
        logging.error("Images directory %s does not exist", args.images_dir)
        return 1

    images = discover_images(args.images_dir)
    if not images:
        logging.error("No images found in %s", args.images_dir)
        return 1

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        model_id = resolve_model_id(args)
    except (FileNotFoundError, ValueError) as exc:
        logging.error("%s", exc)
        return 1
    logging.info("Loading ViP-LLaVA model %s", model_id)
    model, processor = load_model_and_processor(model_id, args)
    model_device, modal_devices = _resolve_modal_device_map(model)
    if getattr(model, "hf_device_map", None):
        logging.info("Model sharded across devices: %s", model.hf_device_map)
    else:
        logging.info("Model loaded on device %s", model_device)

    for image_path in tqdm(images, desc="Annotating overlay screenshots"):
        annotation = annotate_overlay_image(
            image_path,
            model,
            processor,
            args,
            model_device,
            modal_devices,
        )
        save_annotation(output_dir, annotation)

    logging.info("Annotation complete. Files saved to %s", output_dir)
    return 0


def discover_images(images_dir: Path) -> List[Path]:
    return sorted(
        path
        for path in images_dir.iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS and path.is_file()
    )


def resolve_model_id(args: argparse.Namespace) -> str:
    if getattr(args, "model_path", None):
        model_path = args.model_path.expanduser()
        if not model_path.exists():
            raise FileNotFoundError(f"Model path {model_path} does not exist")
        return str(model_path)
    if args.model_id:
        return args.model_id
    try:
        return MODEL_ALIASES[args.model_size]
    except KeyError as exc:  # pragma: no cover - guard against future edits
        raise ValueError(f"Unsupported model size {args.model_size!r}") from exc


def load_model_and_processor(
    model_id: str,
    args: argparse.Namespace,
) -> Tuple[PreTrainedModel, AutoProcessor]:
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    chosen_dtype = dtype_map[args.torch_dtype]
    if args.device == "cpu" and chosen_dtype != torch.float32 and not args.load_in_4bit:
        logging.warning(
            "Overriding torch dtype to float32 because the CPU backend does not support %s",
            args.torch_dtype,
        )
        chosen_dtype = torch.float32

    model_kwargs: Dict[str, object] = {"low_cpu_mem_usage": True, "trust_remote_code": True}
    if args.load_in_4bit:
        if BitsAndBytesConfig is None:
            raise RuntimeError("bitsandbytes is required for --load-in-4bit but is not installed.")
        if args.device_map is None:
            model_kwargs["device_map"] = "auto"
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    else:
        model_kwargs["torch_dtype"] = chosen_dtype
        if args.device_map:
            model_kwargs["device_map"] = args.device_map

    model = VipLlavaForConditionalGeneration.from_pretrained(model_id, **model_kwargs)
    if not args.load_in_4bit and not args.device_map:
        model.to(torch.device(args.device))
    model.eval()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return model, processor


def annotate_overlay_image(
    image_path: Path,
    model: PreTrainedModel,
    processor: AutoProcessor,
    args: argparse.Namespace,
    model_device: torch.device,
    modal_devices: Dict[str, torch.device],
) -> AnnotationResult:
    with Image.open(image_path) as pil_image:
        image = pil_image.convert("RGB")

    conversation = build_overlay_conversation(args.prompt)
    inputs = prepare_inputs(processor, conversation, image, model_device, modal_devices)
    raw_response = generate_response(model, processor, inputs, args)
    annotations, leftovers = parse_icon_descriptions(raw_response, args.lowercase_output)
    return AnnotationResult(
        image=image_path.name,
        model=model.name_or_path if hasattr(model, "name_or_path") else "ViP-LLaVA",
        prompt=args.prompt,
        raw_response=raw_response,
        annotations=annotations,
        unparsed_lines=leftovers,
    )


def build_overlay_conversation(prompt: str) -> List[Dict[str, object]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt.strip()},
                {"type": "image"},
            ],
        }
    ]


def _string_to_device(device: object) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if isinstance(device, str):
        if device == "disk":
            return torch.device("cpu")
        return torch.device(device)
    if isinstance(device, int):
        return torch.device(f"cuda:{device}")
    raise TypeError(f"Unsupported device entry {device!r}")


def _resolve_modal_device_map(model: PreTrainedModel) -> Tuple[torch.device, Dict[str, torch.device]]:
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


def generate_response(
    model: PreTrainedModel,
    processor: AutoProcessor,
    inputs: Dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> str:
    eos_token_id = processor.tokenizer.eos_token_id if processor.tokenizer else model.config.eos_token_id
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "top_p": args.top_p,
        "temperature": args.temperature if args.temperature > 0 else None,
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


def parse_icon_descriptions(response: str, lowercase: bool = False) -> Tuple[Dict[str, str], List[str]]:
    annotations: Dict[str, str] = {}
    leftovers: List[str] = []

    for raw_line in response.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        icon_id, icon_name = _split_icon_line(line)
        if icon_id is None or icon_name is None:
            leftovers.append(line)
            continue
        normalised_id = normalise_icon_id(icon_id)
        cleaned_name = icon_name.strip()
        if lowercase:
            cleaned_name = cleaned_name.lower()
        if normalised_id in annotations:
            logging.warning("Duplicate annotation for %s; keeping the first value", normalised_id)
            continue
        annotations[normalised_id] = cleaned_name

    return annotations, leftovers


def _split_icon_line(line: str) -> Tuple[str | None, str | None]:
    if ":" in line:
        key, value = line.split(":", 1)
        return key.strip(), value.strip()

    match = re.match(r"^(id\s*\d+|#?\d+)\s+(.+)$", line, flags=re.IGNORECASE)
    if match:
        return match.group(1), match.group(2)

    return None, None


def normalise_icon_id(raw_id: str) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z#]", "", raw_id).upper()
    cleaned = cleaned.replace("#", "")
    if cleaned.startswith("ID"):
        suffix = cleaned[2:]
    else:
        suffix = cleaned
    return f"ID{suffix}"


def save_annotation(output_dir: Path, annotation: AnnotationResult) -> None:
    payload = {
        "image": annotation.image,
        "model": annotation.model,
        "prompt": annotation.prompt,
        "raw_response": annotation.raw_response,
        "annotations": [
            {"id": icon_id, "name": annotation.annotations[icon_id]}
            for icon_id in sorted(annotation.annotations, key=_sort_icon_id)
        ],
    }
    if annotation.unparsed_lines:
        payload["unparsed_lines"] = annotation.unparsed_lines

    target = output_dir / f"{Path(annotation.image).stem}_vip_llava.json"
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _sort_icon_id(icon_id: str) -> Tuple[int, str]:
    match = re.match(r"ID(\d+)", icon_id.upper())
    if match:
        return int(match.group(1)), icon_id.upper()
    return (0, icon_id.upper())


if __name__ == "__main__":
    raise SystemExit(main())
