#!/bin/bash
# Product classification example with system prompt
#
# Usage:
#   bash examples/run_classify.sh product.jpg

IMAGE="${1:?Usage: $0 <image_path>}"
MODEL="${2:-stored/liquid_quantized_gguf/model.gguf}"

PROMPT="You are an expert product analyser for Amazon.

Assign a category, price and weight based on the provided image.

Only return category, price (USD) and weight (pounds).

Return output in this YAML format:
---
Category: string
Price: float
Weight: string

Generated output:"

llama-cli \
    -m "$MODEL" \
    --image "$IMAGE" \
    -p "$PROMPT" \
    -n 256 \
    --temp 0.1
