#!/bin/bash
# Example: Run LiquidAI/LFM2-VL-1.6B with a LoRA adapter

uv sync
uv run transformers/models/LFM2-VL-1.6B/run.py \
    --model_path tmp/923a29bb-cf33-4400-b06d-2948a90e5a59/model/ \
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
