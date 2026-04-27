#!/bin/bash
# Example: Run LiquidAI/LFM2.5-VL-450M with a LoRA adapter

uv sync
uv run transformers/models/LFM2.5-VL-450M/run.py \
    --model_path tmp/a22afc30-0a92-4d86-ac7a-ec31be04c5e1/model/ \
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
