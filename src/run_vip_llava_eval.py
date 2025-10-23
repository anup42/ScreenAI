#!/usr/bin/env python3
"""Batch annotate overlay UI icons using ViP-LLaVA via the llava package.

This script mirrors the objective and prompt of annotate_icons_vip_llava.py,
but runs inference through llava's native model builder and utilities.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
from tqdm import tqdm

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
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
        description="Describe overlay-labeled UI icons using ViP-LLaVA (llava backend)."
    )
    parser.add_argument("--images-dir", type=Path, required=True, help="Directory of overlay-labeled images")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write JSON outputs")

    parser.add_argument("--model-path", required=True, help="Model path or HF repo id (e.g. mucai/vip-llava-7b)")
    parser.add_argument("--model-base", default=None, help="Optional base model for LoRA/MM-projector variants")
    parser.add_argument("--conv-mode", default=None, help="Conversation template override (auto if omitted)")
    parser.add_argument("--offline", action="store_true", help="Force offline mode; never attempt to reach HF.")

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=192)

    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt presented to the model")
    parser.add_argument("--lowercase-output", action="store_true", help="Lowercase parsed icon names")
    parser.add_argument("--quiet", action="store_true", help="Reduce log verbosity")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    # Enforce offline behavior if requested
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    if not args.images_dir.exists():
        logging.error("Images directory %s does not exist", args.images_dir)
        return 1
    images = discover_images(args.images_dir)
    if not images:
        logging.error("No images found in %s", args.images_dir)
        return 1
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load model via llava
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    # Validate local model directory if a filesystem path is provided
    _validate_local_model_dir(args.model_path)
    _verify_weight_shards(args.model_path)

    # Heuristic: if local dir has no safetensors files, force use_safetensors=False to
    # avoid weight-resolution codepaths that expect safetensors and can hit None checks.
    use_safetensors = _has_safetensors(args.model_path)

    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path,
        args.model_base,
        model_name,
        local_files_only=args.offline,
        use_safetensors=use_safetensors,
    )

    # Determine conversation mode (same heuristic as llava.eval.run_llava)
    if args.conv_mode is not None:
        conv_mode = args.conv_mode
    else:
        if "llama-2" in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"
    logging.info("Using conversation mode: %s", conv_mode)

    for image_path in tqdm(images, desc="Annotating overlay screenshots"):
        result = annotate_image_llava(
            image_path=image_path,
            tokenizer=tokenizer,
            model=model,
            image_processor=image_processor,
            conv_mode=conv_mode,
            prompt=args.prompt,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            lowercase=args.lowercase_output,
        )
        save_annotation(args.output_dir, result)

    logging.info("Annotation complete. Files saved to %s", args.output_dir)
    return 0


def annotate_image_llava(
    image_path: Path,
    tokenizer,
    model,
    image_processor,
    conv_mode: str,
    prompt: str,
    temperature: float,
    top_p: float | None,
    num_beams: int,
    max_new_tokens: int,
    lowercase: bool,
) -> AnnotationResult:
    # Prepare prompt with image token
    qs = prompt
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if getattr(model.config, "mm_use_im_start_end", False):
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt_text = conv.get_prompt()

    # Load and process image
    with Image.open(image_path) as pil_image:
        image = pil_image.convert("RGB")
    images_tensor = process_images([image], image_processor, model.config).to(
        model.device, dtype=torch.float16
    )

    input_ids = (
        tokenizer_image_token(prompt_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .to(model.device)
    )

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stop_tokens = [stop_str] if isinstance(stop_str, str) and len(stop_str) > 0 else []
    stopping_criteria = KeywordsStoppingCriteria(stop_tokens, tokenizer, input_ids) if stop_tokens else None

    with torch.inference_mode():
        gen_kwargs = dict(
            input_ids,
            images=images_tensor,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
        if stopping_criteria is not None:
            gen_kwargs["stopping_criteria"] = [stopping_criteria]
        output_ids = model.generate(**gen_kwargs)

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    outputs = outputs if isinstance(outputs, str) else ("" if outputs is None else str(outputs))
    if stop_tokens and outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.strip()

    annotations, leftovers = parse_icon_descriptions(outputs, lowercase)
    return AnnotationResult(
        image=image_path.name,
        model=getattr(model, "name_or_path", "ViP-LLaVA"),
        prompt=prompt,
        raw_response=outputs,
        annotations=annotations,
        unparsed_lines=leftovers,
    )


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


def discover_images(images_dir: Path) -> List[Path]:
    return sorted(
        p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def _validate_local_model_dir(model_path: str) -> None:
    """If model_path looks like a local folder, ensure key files exist.

    This prevents Transformers from attempting hub downloads when a file is missing.
    """
    path = Path(model_path)
    if not path.exists():
        # Likely a repo id; let llava/transformers handle error messaging.
        return
    if not path.is_dir():
        return

    required = [
        path / "config.json",
        path / "generation_config.json",
        path / "tokenizer_config.json",
        path / "special_tokens_map.json",
        path / "tokenizer.model",
    ]
    has_weights = any([
        (path / "pytorch_model.bin").exists(),
        (path / "pytorch_model.bin.index.json").exists(),
        any(path.glob("pytorch_model-*.bin")),
        (path / "model.safetensors").exists(),
        (path / "model.safetensors.index.json").exists(),
        any(path.glob("model-*.safetensors")),
    ])

    missing = [str(p.name) for p in required if not p.exists()]
    if not has_weights:
        missing.append("pytorch_model(.bin/.index.json + shards) or model(.safetensors/.index.json + shards)")
    if missing:
        raise FileNotFoundError(
            "Model directory is missing required files: " + ", ".join(missing)
        )


def _has_safetensors(model_path: str) -> bool:
    p = Path(model_path)
    if not p.exists() or not p.is_dir():
        # Probably a repo id; let downstream decide.
        return True
    if (p / "model.safetensors").exists() or (p / "model.safetensors.index.json").exists():
        return True
    if any(p.glob("model-*.safetensors")):
        return True
    return False


def _verify_weight_shards(model_path: str) -> None:
    """Ensure that all weight shards referenced by the index file exist locally.

    This prevents downstream loaders from producing None checkpoint entries that cause
    attribute errors like .endswith on None.
    """
    path = Path(model_path)
    if not path.exists() or not path.is_dir():
        return

    bin_index = path / "pytorch_model.bin.index.json"
    safetensors_index = path / "model.safetensors.index.json"

    index_file = None
    expected_ext = None
    if safetensors_index.exists():
        index_file = safetensors_index
        expected_ext = ".safetensors"
    elif bin_index.exists():
        index_file = bin_index
        expected_ext = ".bin"
    else:
        # If no index file, rely on single-file names if present
        single_candidates = [path / "pytorch_model.bin", path / "model.safetensors"]
        if any(p.exists() for p in single_candidates):
            return
        # Nothing we can verify here
        return

    import json
    try:
        with index_file.open("r", encoding="utf-8") as f:
            idx = json.load(f)
    except Exception as exc:
        raise RuntimeError(f"Failed to read weight index {index_file}: {exc}") from exc

    weight_map = idx.get("weight_map") or {}
    shard_names = sorted(set(weight_map.values()))
    missing = []
    wrong_ext = []
    for name in shard_names:
        if not isinstance(name, str):
            missing.append(str(name))
            continue
        if expected_ext and not name.endswith(expected_ext):
            wrong_ext.append(name)
        if not (path / name).exists():
            missing.append(name)

    if wrong_ext:
        raise RuntimeError(
            "Weight index references shards with wrong extension for this index: "
            + ", ".join(wrong_ext)
        )
    if missing:
        raise FileNotFoundError(
            "Missing weight shard files referenced by index: " + ", ".join(missing)
        )


if __name__ == "__main__":
    raise SystemExit(main())
