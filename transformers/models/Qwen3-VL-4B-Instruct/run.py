#!/usr/bin/env python3
"""Run inference with Qwen/Qwen3-VL-4B-Instruct.

Usage:
    python models/Qwen3-VL-4B-Instruct/run.py --model_path /path/to/model --images photo.jpg
    python models/Qwen3-VL-4B-Instruct/run.py --model_path /path/to/model --images img.jpg --user_prompt "Classify this." --debug
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from base import (build_messages,
                  get_arg_parser,
                  load_images,
                  load_model,
                  print_config,
                  resolve_device,
                  resolve_dtype,
                  run_inference,
                  )

# HuggingFace model ID — used as base model for LoRA adapters
MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"

# Qwen3-VL uses dynamic resolution — no special processor kwargs needed
PROCESSOR_KWARGS = {}
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
