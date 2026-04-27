#!/bin/bash
# Example: Run google/gemma-4-E2B-it with a LoRA adapter

uv sync
uv run transformers/models/gemma-4-E2B-it/run.py \
    --model_path tmp/ef34953d-6799-489a-89a8-39509d4db21e/model/ \
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
