#!/bin/bash
# Run your fine-tuned GGUF model with the base mmproj (vision projector)
#
# Usage:
#   bash examples/run_finetuned.sh photo.jpg "Describe this image."

IMAGE="${1:?Usage: $0 <image_path> [prompt]}"
PROMPT="${2:-Describe this image in detail.}"
MODEL="${3:-stored/liquid_quantized_gguf/model.gguf}"
MMPROJ="$HOME/.cache/huggingface/hub/models--LiquidAI--LFM2-VL-450M-GGUF/snapshots/fddce730245b2f3cb199aca481db72a33de8e323/mmproj-LFM2-VL-450M-Q8_0.gguf"

if [ ! -f "$MMPROJ" ]; then
    echo "mmproj not found. Downloading base model to get it..."
    llama-cli -hf LiquidAI/LFM2-VL-450M-GGUF:Q4_0 -p "test" -n 1 2>/dev/null || true
    echo "Retrying..."
fi

llama-cli -m "$MODEL" --mmproj "$MMPROJ" --image "$IMAGE" -p "$PROMPT" -n 256
