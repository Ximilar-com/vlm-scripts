#!/usr/bin/env python3
"""Run inference with Qwen/Qwen3-VL-4B-Instruct.

Edit the constants below to change the prompt, images, or generation
settings. The model is downloaded automatically on first run: set
MODEL_URL_HF_QWEN3_VL_4B_INSTRUCT to the download URL from your model's details
page at https://app.ximilar.com/platform/vlm/tasks/ and run:

    export MODEL_DOWNLOAD_DIRECTORY=./stored
    export MODEL_URL_HF_QWEN3_VL_4B_INSTRUCT='https://...'
    uv run models/Qwen3-VL-4B-Instruct/run.py
"""

import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]  # the transformers/ directory
sys.path.insert(0, str(ROOT_DIR))

from base import (
    build_messages,
    ensure_model,
    load_images,
    load_model,
    print_config,
    resolve_device,
    resolve_dtype,
    run_inference,
)

# --- Model -------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"  # HuggingFace ID -- used as base model for LoRA adapters
MODEL_URL_ENV = "MODEL_URL_HF_QWEN3_VL_4B_INSTRUCT"  # env var holding the model download URL

# Qwen3-VL uses dynamic resolution -- no special processor kwargs needed
PROCESSOR_KWARGS = {}

# --- Inference settings (edit these) ------------------------------------------
IMAGES = ["media/photo.jpg"]  # local image paths, relative to the transformers/ directory
USER_PROMPT = "Describe the product in the image."
SYSTEM_PROMPT = None
MAX_TOKENS = 256
TEMPERATURE = 0.7  # 0.0 = greedy
RESIZE = None  # max image dimension in px, None = keep original size
DEBUG = False


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    model_path = ensure_model(MODEL_URL_ENV, MODEL_ID)
    device = resolve_device("auto")
    dtype = resolve_dtype(device, "auto")

    model, processor = load_model(MODEL_ID, model_path, device, dtype, processor_kwargs=PROCESSOR_KWARGS)
    images = load_images([str(ROOT_DIR / p) for p in IMAGES], max_size=RESIZE)
    messages = build_messages(images, USER_PROMPT, SYSTEM_PROMPT)

    print_config(MODEL_ID, model_path, processor, images, device, dtype, MAX_TOKENS, TEMPERATURE, RESIZE, USER_PROMPT, SYSTEM_PROMPT)

    result = run_inference(model, processor, messages, images, max_tokens=MAX_TOKENS, temperature=TEMPERATURE, debug=DEBUG)
    print("\033[31mOutput:\033[0m")
    print(result)


if __name__ == "__main__":
    main()
