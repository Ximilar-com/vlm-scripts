#!/bin/bash
# Example: Run LiquidAI/LFM2.5-VL-1.6B with a LoRA adapter

uv sync
export MODEL_PATH="../stored/lfm2.5-1.6b-lora/model"
export RUN_SCRIPT="./models/LFM2.5-VL-1.6B/run.py"
[ -d "$MODEL_PATH" ] && [ -f "$RUN_SCRIPT" ] || echo "INVALID PATHS"
[ -d "$MODEL_PATH" ] && [ -f "$RUN_SCRIPT" ] && [[ "$(basename "$PWD")" == "transformers" ]] || echo "You have to be in the transformers directory to run this script"
echo "Using model path: $MODEL_PATH"
echo "Using run script: $RUN_SCRIPT"
uv run $RUN_SCRIPT \
    --model_path "$MODEL_PATH" \
    --images "https://m.media-amazon.com/images/I/71jGMgjyOOL._AC_SY300_SX300_QL70_ML2_.jpg" \
    --user_prompt "Assign a category, price and weight based on the provided image.

Only return category, price (USD) and weight (pounds).

Return output in this YAML format:
---
Category: string
Price: float
Weight: string

Generated output:" \
    --system_prompt "You are an expert product analyser for Amazon."
