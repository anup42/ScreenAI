#!/usr/bin/env python3
"""Minimal ViP-LLaVA evaluation runner using the llava package.

This wraps llava.eval.run_llava.eval_model with a simple CLI, closely
matching the reference snippet from the ViP-LLaVA repository.
"""
from __future__ import annotations

import argparse
from types import SimpleNamespace

from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ViP-LLaVA eval (llava.eval.run_llava.eval_model)")
    parser.add_argument("--model-path", required=True, help="Model path or HF repo id (e.g. mucai/vip-llava-7b)")
    parser.add_argument("--image-file", required=True, help="Image path or URL (multiple allowed using --sep)")
    parser.add_argument("--query", required=True, help="User prompt/question")

    parser.add_argument("--model-base", default=None, help="Optional base model for LoRA/MM-projector variants")
    parser.add_argument("--conv-mode", default=None, help="Conversation template override (auto if omitted)")
    parser.add_argument("--sep", default=",", help="Separator for multiple images (default: ,)")

    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_name = get_model_name_from_path(args.model_path)

    ns = SimpleNamespace(
        model_path=args.model_path,
        model_name=model_name,
        query=args.query,
        image_file=args.image_file,
        conv_mode=args.conv_mode,
        model_base=args.model_base,
        temperature=args.temperature,
        top_p=args.top_p,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
        sep=args.sep,
    )

    eval_model(ns)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

