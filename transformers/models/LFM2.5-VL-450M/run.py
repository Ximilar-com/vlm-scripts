#!/usr/bin/env python3
"""Run inference with LiquidAI/LFM2.5-VL-450M.

Usage:
    python models/LFM2.5-VL-450M/run.py --model_path /path/to/model --images photo.jpg
    python models/LFM2.5-VL-450M/run.py --model_path /path/to/model --images img.jpg --user_prompt "Classify this." --debug
"""

import argparse
import logging
import os
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

MODEL_ID = "LiquidAI/LFM2.5-VL-450M"

# Liquid models use image tiling with token budget control
PROCESSOR_KWARGS = {
    "min_image_tokens": 64,
    "max_image_tokens": 256,
    "do_image_splitting": True,
}


def _require_lfm25_mps_fallback() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--device", default="auto")
    args, _ = parser.parse_known_args()

    uses_mps = args.device == "mps"
    auto_on_apple_silicon = (
            args.device == "auto"
            and platform.system() == "Darwin"
            and platform.machine() == "arm64"
    )
    if not uses_mps and not auto_on_apple_silicon:
        return

    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        return

    if uses_mps:
        print(
            "run.py: error: LiquidAI/LFM2.5-VL models on MPS require "
            "PYTORCH_ENABLE_MPS_FALLBACK=1 before Python starts. "
            "Re-run with `PYTORCH_ENABLE_MPS_FALLBACK=1 python ...` or choose "
            "`--device cpu`/`--device cuda`.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    print(
        "run.py: error: LiquidAI/LFM2.5-VL with `--device auto` may resolve to "
        "MPS on Apple Silicon and requires PYTORCH_ENABLE_MPS_FALLBACK=1 before "
        "Python starts. Set the env var and re-run, or pass `--device cpu` to avoid MPS.",
        file=sys.stderr,
    )
    raise SystemExit(2)


_require_lfm25_mps_fallback()

from base import (
    build_messages,
    get_arg_parser,
    load_images,
    load_model,
    print_config,
    resolve_device,
    resolve_dtype,
    run_inference,
)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = get_arg_parser(MODEL_ID).parse_args()

    device = resolve_device(args.device)
    dtype = resolve_dtype(device, args.dtype, model_id=MODEL_ID)

    model, processor = load_model(MODEL_ID, args.model_path, device, dtype, processor_kwargs=PROCESSOR_KWARGS)
    images = load_images(args.images, max_size=args.resize)
    messages = build_messages(images, args.user_prompt, args.system_prompt)

    print_config(MODEL_ID, args.model_path, processor, images, device, dtype, args.max_tokens, args.temperature, args.resize, args.user_prompt, args.system_prompt)

    result = run_inference(model, processor, messages, images, max_tokens=args.max_tokens, temperature=args.temperature, debug=args.debug)
    print("\033[31mOutput:\033[0m")
    print(result)


if __name__ == "__main__":
    main()
