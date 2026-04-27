#!/bin/bash
# Example: Run google/gemma-3-4b-it with a LoRA adapter

uv sync
uv run "$(dirname "$0")/../models/gemma-3-4b-it/run.py" \
    --model_path "$(dirname "$0")/../stored/gemma-lora/" \
    --images "https://m.media-amazon.com/images/I/71jGMgjyOOL._AC_SY300_SX300_QL70_ML2_.jpg" \
    --device cpu \
    --user_prompt "Assign a category, price and weight based on the provided image.

Only return category, price (USD) and weight (pounds).

Return output in this YAML format:
---
Category: string
Price: float
Weight: string

Generated output:" \
    --system_prompt "You are an expert product analyser for Amazon."
