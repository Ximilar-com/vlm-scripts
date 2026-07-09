#!/usr/bin/env bash
# Run agentic inference with LiquidAI/LFM2.5-VL-450M (GGUF via llama.cpp): thinking + tool calls.
#
# The model is downloaded automatically on first run (Python helper), then
# llama-cli runs the prompt via `nix run nixpkgs#llama-cpp`:
#
#     export MODEL_DOWNLOAD_DIRECTORY=./stored
#     export MODEL_URL_GGUF_LFM2_5_450M='https://...'
#     bash models/LFM2.5-VL-450M/run_agentic.sh
#
# The download URL comes from your model's details page at
# https://app.ximilar.com/platform/vlm/tasks/ (valid for 24 hours). The
# artifact is an archive containing the model GGUF and the mmproj (vision
# projector) GGUF.
#
# To use a local llama.cpp install instead of nix
# (https://github.com/ggml-org/llama.cpp/blob/master/docs/install.md):
#     export LLAMA_CLI="llama-cli"
set -euo pipefail
cd "$(dirname "$0")/../.."  # the llama-cpp/ directory

MODEL_ID="LiquidAI/LFM2.5-VL-450M"
MODEL_URL_ENV="MODEL_URL_GGUF_LFM2_5_450M"

# --- Inference settings (edit these) ---
# The tool is described in the system prompt (llama-cli has no tools flag).
# The raw thinking / tool-call markers are printed as-is (-sp). For parsed
# output and a simulated tool-execution round, see the transformers examples.
IMAGE="media/photo.jpg"
SYSTEM_PROMPT="You are an expert product analyser. You can call this tool:
submit_product_classification(category: string, price: number, weight: number) — Submit the final classification for the product in the image."
PROMPT="Look at the product in the image, think through category, price and weight, then call submit_product_classification with your answer."
MAX_TOKENS=1024
TEMPERATURE=0.0

# Download the model on first run and get the local GGUF paths
MODEL_PATHS="$(uv run download_model.py "$MODEL_URL_ENV" "$MODEL_ID")"
eval "$MODEL_PATHS"

LLAMA_CLI="${LLAMA_CLI:-nix run nixpkgs#llama-cpp --}"
$LLAMA_CLI \
    -m "$MODEL_GGUF" \
    --mmproj "$MMPROJ_GGUF" \
    --image "$IMAGE" \
    -sys "$SYSTEM_PROMPT" \
    -p "$PROMPT" \
    -n "$MAX_TOKENS" \
    --temp "$TEMPERATURE" \
    -sp \
    --single-turn
