#!/bin/bash
# Example: Run Qwen3-VL-2B-Instruct with a LoRA adapter

python models/Qwen3-VL-2B-Instruct/run.py \
    --model_path stored/qwen3-2b-lora/ \
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
