# Ximilar VLM — llama.cpp

Run your trained VLM models from [Ximilar Platform](https://www.ximilar.com) using [llama.cpp](https://github.com/ggml-org/llama.cpp).

llama.cpp runs GGUF-quantized models locally with full vision support — fast inference on CPU, Apple Silicon (Metal), and NVIDIA GPUs.

## Supported Models

| Model | Base Model |
|---|---|
| LiquidAI LFM2-VL-450M | [LiquidAI/LFM2-VL-450M](https://huggingface.co/LiquidAI/LFM2-VL-450M) |

## Requirements

- [llama.cpp](https://github.com/ggml-org/llama.cpp) (with `llama-cli`)
- Your downloaded GGUF model from Ximilar (placed in `stored/`)

## Quick Start

### 1. Install llama.cpp

```bash
# macOS (Homebrew)
brew install llama.cpp

# Linux — build from source
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build
cmake --build build --config Release
# Binary is at build/bin/llama-cli
```

### 2. Place your model

Download your GGUF model from Ximilar and place it in `stored/`. For example:

```
stored/liquid_quantized_gguf/model.gguf
```

### 3. Download the base model (needed once)

Your fine-tuned GGUF only contains text weights. The vision projector (`mmproj`) comes from the base model. Run this once to download it:

```bash
llama-cli -hf LiquidAI/LFM2-VL-450M-GGUF:Q4_0 -p "test" -n 1
```

This caches the base model and mmproj to `~/.cache/huggingface/hub/`.

### 4. Run inference

```bash
# Fine-tuned model with image (uses base mmproj for vision)
bash examples/run_finetuned.sh photo.jpg "Describe this image."

# Or the official base model from HuggingFace (mmproj auto-detected)
llama-cli -hf LiquidAI/LFM2-VL-450M-GGUF:Q4_0 --image photo.jpg -p "Describe this image." --temp 0.0
```

> **Important**: When using `-m` with a local fine-tuned GGUF, you must also pass `--mmproj` pointing to the base model's vision projector. The `-hf` flag handles this automatically. See `examples/run_finetuned.sh` for the full command.

## Usage Examples

### Local model with image

```bash
llama-cli \
    -m stored/liquid_quantized_gguf/model.gguf \
    --image product.jpg \
    -p "Assign a category, price and weight for this product."
```

### HuggingFace model (downloads automatically)

```bash
# Q4_0 quantization (smallest, ~219 MB)
llama-cli -hf LiquidAI/LFM2-VL-450M-GGUF:Q4_0 --image photo.jpg -p "Describe this image."

# Q8_0 quantization (better quality, ~379 MB)
llama-cli -hf LiquidAI/LFM2-VL-450M-GGUF:Q8_0 --image photo.jpg -p "Describe this image."

# F16 full precision (~711 MB)
llama-cli -hf LiquidAI/LFM2-VL-450M-GGUF:F16 --image photo.jpg -p "Describe this image."
```

### Interactive mode

```bash
llama-cli -m stored/liquid_quantized_gguf/model.gguf --interactive
```

Type prompts and use `/image photo.jpg` to load images during the conversation.

### Common options

| Option | Description |
|---|---|
| `-m PATH` | Path to local GGUF model file |
| `--mmproj PATH` | Path to vision projector GGUF (required for local fine-tuned models) |
| `-hf REPO:QUANT` | Download from HuggingFace (e.g. `LiquidAI/LFM2-VL-450M-GGUF:Q4_0`) |
| `--image PATH` | Path to image file |
| `-p "PROMPT"` | Text prompt |
| `--interactive` | Interactive chat mode |
| `-n 256` | Max tokens to generate (default: 256) |
| `--temp 0.1` | Temperature — 0.0 = greedy, default 0.8 |
| `-ngl 99` | Number of layers to offload to GPU (use 99 for all) |
| `--threads 8` | Number of CPU threads |

### About `--mmproj` (vision projector)

Vision-language models have two parts:
- **Model GGUF** — the text decoder (your fine-tuned weights)
- **mmproj GGUF** — the vision encoder/projector (processes images into tokens)

When using `-hf`, both files are downloaded and linked automatically. When using `-m` with a local fine-tuned model, you must provide `--mmproj` separately because the fine-tuned GGUF only contains text weights.

The mmproj comes from the **base model** and is shared across all fine-tuned variants:

```bash
# Fine-tuned model needs explicit mmproj
llama-cli \
    -m stored/liquid_quantized_gguf/model.gguf \
    --mmproj ~/.cache/huggingface/hub/models--LiquidAI--LFM2-VL-450M-GGUF/snapshots/*/mmproj-LFM2-VL-450M-Q8_0.gguf \
    --image photo.jpg \
    -p "Describe this image."

# Base model from HuggingFace — mmproj auto-detected, no --mmproj needed
llama-cli -hf LiquidAI/LFM2-VL-450M-GGUF:Q4_0 --image photo.jpg -p "Describe this image."
```

The `examples/run_finetuned.sh` script handles the mmproj path automatically.

### GPU acceleration

```bash
# Apple Silicon (Metal) — automatic, no flags needed
llama-cli -m stored/liquid_quantized_gguf/model.gguf --image photo.jpg -p "Describe this image."

# NVIDIA GPU — offload all layers
llama-cli -m stored/liquid_quantized_gguf/model.gguf --image photo.jpg -p "Describe this image." -ngl 99
```

## Example Scripts

| Script | Description |
|---|---|
| [examples/run_finetuned.sh](examples/run_finetuned.sh) | Run fine-tuned model with image (auto-finds mmproj) |
| [examples/run_from_hf.sh](examples/run_from_hf.sh) | Run the official base model from HuggingFace |
| [examples/run_with_image.sh](examples/run_with_image.sh) | Run local model with image (no mmproj) |
| [examples/run_classify.sh](examples/run_classify.sh) | Product classification example |

## Available Quantizations

| Quantization | Size | Quality | Speed |
|---|---|---|---|
| Q4_0 | ~219 MB | Good | Fastest |
| Q8_0 | ~379 MB | Better | Fast |
| F16 | ~711 MB | Best | Slower |

Your fine-tuned model from Ximilar is pre-quantized — check the file size to determine the quantization level.

## Troubleshooting

### `llama-cli: command not found`

llama.cpp is not installed or not on your PATH.

```bash
# macOS
brew install llama.cpp

# Linux — after building from source, add to PATH
export PATH="/path/to/llama.cpp/build/bin:$PATH"
```

### Slow inference

- Use GPU offloading: `-ngl 99` (NVIDIA) or automatic on Apple Silicon
- Use a smaller quantization (Q4_0 instead of F16)
- Increase thread count: `--threads $(nproc)`

### Image not recognized

Make sure you pass `--image photo.jpg` as a flag. `llama-cli` auto-detects vision models and loads the mmproj (vision projector) automatically when using `-hf`. For local fine-tuned models, you may need to pass `--mmproj` explicitly — see `examples/run_finetuned.sh`.

## Project structure

```
llama-cpp/
├── examples/
│   ├── run_finetuned.sh           # Fine-tuned model + image (auto-finds mmproj)
│   ├── run_from_hf.sh             # Official base model from HuggingFace
│   ├── run_with_image.sh          # Local model + image
│   └── run_classify.sh            # Product classification example
├── stored/                        # Your downloaded GGUF models (gitignored)
└── README.md
```
