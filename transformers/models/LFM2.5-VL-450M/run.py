#!/usr/bin/env python3
"""Run inference with LiquidAI/LFM2.5-VL-450M (no tools).

The fine-tuned checkpoint may emit thinking (<think>...</think>) or
tool-call markers alongside its answer, so the output is parsed with the
same response schema run_agentic.py uses: markers become Thinking /
Tool calls / Answer sections when present; a plain response is just the
Answer.

For tool calling (with a simulated tool round), see run_agentic.py in
this directory.

Edit the constants below to change the prompt, images, or generation
settings. The model is downloaded automatically on first run:

    export MODEL_DOWNLOAD_DIRECTORY=./stored
    export MODEL_URL_HF_LFM2_5_450M='https://...'
    uv run models/LFM2.5-VL-450M/run.py

The download URL comes from your model's details page at
https://app.ximilar.com/platform/vlm/tasks/ (valid for 24 hours).
"""

import logging
import os
import platform
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]  # the transformers/ directory
sys.path.insert(0, str(ROOT_DIR))


def _require_lfm25_mps_fallback() -> None:
    """LFM2.5-VL uses ops MPS doesn't implement; the CPU-fallback env var must be set before torch loads."""
    if (
            platform.system() == "Darwin"
            and platform.machine() == "arm64"
            and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "1"
    ):
        print(
            "error: LiquidAI/LFM2.5-VL on Apple Silicon requires PYTORCH_ENABLE_MPS_FALLBACK=1 "
            "before Python starts. Re-run with `PYTORCH_ENABLE_MPS_FALLBACK=1 uv run ...`.",
            file=sys.stderr,
        )
        raise SystemExit(2)


_require_lfm25_mps_fallback()

from agentic import parse_agentic, print_parsed, run_agentic_inference
from base import (
    build_messages,
    ensure_model,
    load_images,
    load_model,
    print_config,
    resolve_device,
    resolve_dtype,
)

# --- Model -------------------------------------------------------------------
MODEL_ID = "LiquidAI/LFM2.5-VL-450M"  # HuggingFace ID -- used as base model for LoRA adapters
MODEL_URL_ENV = "MODEL_URL_HF_LFM2_5_450M"  # env var holding the model download URL
FAMILY = "lfm2.5"  # wire-format family: qwen3vl | lfm2.5 | gemma4

# Liquid models use image tiling with token budget control
PROCESSOR_KWARGS = {
    "min_image_tokens": 64,
    "max_image_tokens": 256,
    "do_image_splitting": True,
}

# --- Inference settings (edit these) ------------------------------------------
IMAGES = ["media/photo.jpg"]  # local image paths, relative to the transformers/ directory
USER_PROMPT = "Describe the product in the image."
SYSTEM_PROMPT = None
MAX_TOKENS = 256
TEMPERATURE = 0.0  # 0.0 = greedy
RESIZE = None  # max image dimension in px, None = keep original size
ENABLE_THINKING = True  # gemma-4 only; LFM2.5 always thinks
SHOW_RAW = False  # also print the undecoded raw generation
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

    raw = run_agentic_inference(
        model,
        processor,
        messages,
        images,
        FAMILY,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        enable_thinking=ENABLE_THINKING,
        debug=DEBUG,
    )
    parsed = parse_agentic(raw, processor)

    print("\033[31mOutput:\033[0m")
    print_parsed(parsed, show_raw=SHOW_RAW)


if __name__ == "__main__":
    main()
