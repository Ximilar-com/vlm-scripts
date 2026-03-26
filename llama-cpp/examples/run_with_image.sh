#!/bin/bash
# Run your fine-tuned model with a local image
#
# Usage:
#   bash examples/run_with_image.sh photo.jpg "Describe this image."
#   bash examples/run_with_image.sh product.jpg "What is this product?"

IMAGE="${1:?Usage: $0 <image_path> [prompt]}"
PROMPT="${2:-Describe this image in detail.}"
MODEL="${3:-stored/liquid_quantized_gguf/model.gguf}"

llama-cli \
    -m "$MODEL" \
    --image "$IMAGE" \
    -p "$PROMPT" \
    -n 256
