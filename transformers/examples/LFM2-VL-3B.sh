#!/bin/bash
# Example: Run LiquidAI/LFM2-VL-3B with a LoRA adapter

uv sync
uv run "$(dirname "$0")/../models/LFM2-VL-3B/run.py" \
    --model_path "$(dirname "$0")/../stored/lf2-3b-lora/" \
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
