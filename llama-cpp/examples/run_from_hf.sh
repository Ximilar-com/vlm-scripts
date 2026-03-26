#!/bin/bash
# Run the official LFM2-VL-450M from HuggingFace (downloads automatically)
#
# Usage:
#   bash examples/run_from_hf.sh photo.jpg "Describe this image."
#   bash examples/run_from_hf.sh photo.jpg "Describe this image." Q8_0

IMAGE="${1:?Usage: $0 <image_path> [prompt] [quantization]}"
PROMPT="${2:-Describe this image in detail.}"
QUANT="${3:-Q4_0}"

llama-cli \
    -hf "LiquidAI/LFM2-VL-450M-GGUF:${QUANT}" \
    --image "$IMAGE" \
    -p "$PROMPT" \
    -n 256
