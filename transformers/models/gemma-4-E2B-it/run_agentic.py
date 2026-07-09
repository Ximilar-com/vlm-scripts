#!/usr/bin/env python3
"""Run agentic inference with google/gemma-4-E2B-it.

This model emits thinking and tool calls alongside its answer
(<|channel>thought...<channel|> + <|tool_call>call:name{...}<tool_call|>). Output is parsed into separate
Thinking / Tool calls / Answer sections.

The model expects its tool calls to really be executed. The example can't
do that, so when the model calls a tool, a prepared simulated result (see
SIMULATED_TOOL_RESULTS) is fed back and the model runs once more to
produce its final answer.

For plain answer-only inference (no parsing), see run.py in this directory.

Edit the constants below to change the prompt, images, tools, or
generation settings. The model is downloaded automatically on first run:

    export MODEL_DOWNLOAD_DIRECTORY=./stored
    export MODEL_URL_HF_GEMMA_4_E2B_IT='https://...'
    uv run models/gemma-4-E2B-it/run_agentic.py

The download URL comes from your model's details page at
https://app.ximilar.com/platform/vlm/tasks/ (valid for 24 hours).
"""

import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]  # the transformers/ directory
sys.path.insert(0, str(ROOT_DIR))

from transformers import AutoModelForMultimodalLM

from agentic import (
    append_simulated_tool_results,
    parse_agentic,
    print_parsed,
    run_agentic_inference,
)
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
MODEL_ID = "google/gemma-4-E2B-it"  # HuggingFace ID -- used as base model for LoRA adapters
MODEL_URL_ENV = "MODEL_URL_HF_GEMMA_4_E2B_IT"  # env var holding the model download URL
FAMILY = "gemma4"  # wire-format family: qwen3vl | lfm2.5 | gemma4

# Gemma requires left-side padding for correct attention masking
PROCESSOR_KWARGS = {
    "padding_side": "left",
}

# --- Inference settings (edit these) ------------------------------------------
IMAGES = ["media/photo.jpg"]  # local image paths, relative to the transformers/ directory
USER_PROMPT = "Look at the product in the image, think through category, price and weight, then call submit_product_classification with your answer."
SYSTEM_PROMPT = "You are an expert product analyser."

# Tools exposed to the model: {"name", "description", "parameters"} objects
# (parameters is a JSON Schema). Replace with your own tool definitions.
TOOLS = [
    {
        "name": "submit_product_classification",
        "description": "Submit the final classification for the product in the image.",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "description": "Product category, e.g. shoes"},
                "price": {"type": "number", "description": "Estimated price in USD"},
                "weight": {"type": "number", "description": "Estimated weight in kg"},
            },
            "required": ["category", "price", "weight"],
        },
    },
]

# Prepared results returned to the model when it calls a tool (per tool
# name) -- the example can't execute real tools. Optionally add
# "images": ["local/path.jpg"] to attach media to a simulated result.
SIMULATED_TOOL_RESULTS = {
    "submit_product_classification": {
        "status": "accepted",
        "message": "Classification stored. Summarize the submitted values for the user.",
    },
}

MAX_TOKENS = 512  # thinking + tool calls need more room than a plain answer
TEMPERATURE = 0.0  # 0.0 = greedy
RESIZE = None  # max image dimension in px, None = keep original size
ENABLE_THINKING = True  # gemma-4 only; LFM2.5 and Qwen3-VL-Thinking always think
SHOW_RAW = False  # also print the undecoded raw generation
DEBUG = False


def generate(model, processor, messages, images):
    raw = run_agentic_inference(
        model,
        processor,
        messages,
        images,
        FAMILY,
        tools=TOOLS,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        enable_thinking=ENABLE_THINKING,
        debug=DEBUG,
    )
    return parse_agentic(raw, processor)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    model_path = ensure_model(MODEL_URL_ENV, MODEL_ID)
    device = resolve_device("auto")
    dtype = resolve_dtype(device, "auto")

    model, processor = load_model(MODEL_ID, model_path, device, dtype, processor_kwargs=PROCESSOR_KWARGS, auto_model_class=AutoModelForMultimodalLM)
    images = load_images([str(ROOT_DIR / p) for p in IMAGES], max_size=RESIZE)
    messages = build_messages(images, USER_PROMPT, SYSTEM_PROMPT)

    print_config(MODEL_ID, model_path, processor, images, device, dtype, MAX_TOKENS, TEMPERATURE, RESIZE, USER_PROMPT, SYSTEM_PROMPT)
    print(f"  Tools: {[t['name'] for t in TOOLS]}")
    print()

    parsed = generate(model, processor, messages, images)
    print("\033[31mOutput (turn 1):\033[0m")
    print_parsed(parsed, show_raw=SHOW_RAW)

    if parsed.finish_reason != "tool_call":
        return

    # The model called a tool; feed the prepared simulated result back and
    # run once more so the model produces its final answer.
    tool_images = append_simulated_tool_results(messages, parsed.tool_calls, SIMULATED_TOOL_RESULTS, root_dir=ROOT_DIR)
    images = images + tool_images

    parsed = generate(model, processor, messages, images)
    print()
    print("\033[31mOutput (turn 2 - final answer after simulated tool result):\033[0m")
    print_parsed(parsed, show_raw=SHOW_RAW)


if __name__ == "__main__":
    main()
