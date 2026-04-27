# Ximilar VLM Scripts — Transformers

Simple inference scripts for running your trained VLM (Vision-Language) models from the [Ximilar Platform](https://www.ximilar.com) using HuggingFace Transformers.

Each model has its own `run.py` script. Your model can be:

- **LoRA adapter** (`.safetensors`) — on top of a base model from HuggingFace
- **Full model** (`.safetensors`) — fully fine-tuned weights
- **Full model** (`.pt`) — PyTorch state_dict export

The scripts auto-detect the format from the model directory and handle all three automatically.

## Supported Models

| Model                   | Script                                                                   | Base Model (HuggingFace)                                                      |
|-------------------------|--------------------------------------------------------------------------|-------------------------------------------------------------------------------|
| LiquidAI LFM2-VL-450M   | [models/LFM2-VL-450M/run.py](models/LFM2-VL-450M/run.py)                 | [LiquidAI/LFM2-VL-450M](https://huggingface.co/LiquidAI/LFM2-VL-450M)         |
| LiquidAI LFM2-VL-1.6B   | [models/LFM2-VL-1.6B/run.py](models/LFM2-VL-1.6B/run.py)                 | [LiquidAI/LFM2-VL-1.6B](https://huggingface.co/LiquidAI/LFM2-VL-1.6B)         |
| LiquidAI LFM2-VL-3B     | [models/LFM2-VL-3B/run.py](models/LFM2-VL-3B/run.py)                     | [LiquidAI/LFM2-VL-3B](https://huggingface.co/LiquidAI/LFM2-VL-3B)             |
| LiquidAI LFM2.5-VL-450M | [models/LFM2.5-VL-450M/run.py](models/LFM2.5-VL-450M/run.py)             | [LiquidAI/LFM2.5-VL-450M](https://huggingface.co/LiquidAI/LFM2.5-VL-450M)     |
| LiquidAI LFM2.5-VL-1.6B | [models/LFM2.5-VL-1.6B/run.py](models/LFM2.5-VL-1.6B/run.py)             | [LiquidAI/LFM2.5-VL-1.6B](https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B)     |
| Google Gemma 3 4B PT    | [models/gemma-3-4b-pt/run.py](models/gemma-3-4b-pt/run.py)               | [google/gemma-3-4b-pt](https://huggingface.co/google/gemma-3-4b-pt)           |
| Google Gemma 3 4B       | [models/gemma-3-4b-it/run.py](models/gemma-3-4b-it/run.py)               | [google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it)           |
| Google Gemma 4 E2B      | [models/gemma-4-E2B-it/run.py](models/gemma-4-E2B-it/run.py)             | [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it)         |
| Qwen3-VL 2B             | [models/Qwen3-VL-2B-Instruct/run.py](models/Qwen3-VL-2B-Instruct/run.py) | [Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) |
| Qwen3-VL 4B             | [models/Qwen3-VL-4B-Instruct/run.py](models/Qwen3-VL-4B-Instruct/run.py) | [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) |

## Requirements

- Python 3.12+
- NVIDIA GPU recommended (CUDA 12.x) — CPU and Apple Silicon (MPS) also work

### Library versions

| Library      | Version   |
|--------------|-----------|
| torch        | >= 2.10.0 |
| transformers | >= 5.1.0  |
| peft         | >= 0.18.1 |
| accelerate   | >= 1.12.0 |
| safetensors  | >= 0.7.0  |
| pillow       | >= 10.0   |

## Setup

```bash
# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv .venv --python 3.12
source .venv/bin/activate

# Install dependencies
uv pip install "transformers>=5.1.0" "peft>=0.18.1" "accelerate>=1.12.0" "safetensors>=0.7.0" pillow
```

Then install PyTorch for your platform:

```bash
# Linux with NVIDIA GPU (CUDA 12.8, but you can try to specify yours version)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# macOS (Apple Silicon / CPU) or Linux CPU-only
uv pip install torch torchvision
```

### HuggingFace authentication (required for some models)

Some base models (e.g. **google/gemma-3-4b-it**) are gated — you must accept the license on HuggingFace and authenticate before downloading:

1. Go to the model page (e.g. [google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it)) and accept the license agreement
2. Create an access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Authenticate:

```bash
export HF_TOKEN=hf_your_token_here
```

To make it permanent, add the line to your `~/.bashrc` or `~/.zshrc`.

Without this, gated models will fail with `OSError: ... does not appear to have a file named model.safetensors`.

## Usage

All commands below assume you are inside the `transformers/` directory:

```bash
cd transformers
source .venv/bin/activate
```

### Basic usage

```bash
python models/LFM2.5-VL-1.6B/run.py \
    --model_path /path/to/your/model \
    --images photo.jpg \
    --user_prompt "Describe this image."
```

On Apple Silicon, `LiquidAI/LFM2.5-VL-*` models require explicit MPS CPU fallback opt-in before Python starts:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python models/LFM2.5-VL-1.6B/run.py \
    --model_path /path/to/your/model \
    --images photo.jpg
```

### With URL images and system prompt

```bash
python models/gemma-3-4b-it/run.py \
    --model_path /path/to/your/model \
    --images https://example.com/image1.jpg https://example.com/image2.jpg \
    --user_prompt "Compare these two images." \
    --system_prompt "You are an expert image analyst."
```

### With generation options

```bash
python models/Qwen3-VL-2B-Instruct/run.py \
    --model_path /path/to/your/model \
    --images product.jpg \
    --user_prompt "Classify this product." \
    --max_tokens 512 \
    --temperature 0.7 \
    --resize 768
```

### With debug output

```bash
python models/LFM2.5-VL-1.6B/run.py \
    --model_path /path/to/your/model \
    --images photo.jpg \
    --user_prompt "Describe this image." \
    --debug
```

This prints token counts, input tensor shapes, generation time, and tokens/sec.

### Arguments

| Argument          | Required | Default                | Description                                     |
|-------------------|----------|------------------------|-------------------------------------------------|
| `--model_path`    | Yes      | -                      | Path to your downloaded model directory         |
| `--images`        | No       | none                   | One or more image file paths or URLs            |
| `--user_prompt`   | No       | "Describe this image." | Text prompt for the model                       |
| `--system_prompt` | No       | None                   | System instruction (optional)                   |
| `--max_tokens`    | No       | per model              | Maximum tokens to generate                      |
| `--temperature`   | No       | per model              | Sampling temperature (0.0 = greedy)             |
| `--device`        | No       | auto                   | Device: auto, cpu, cuda, cuda:0, mps            |
| `--dtype`         | No       | auto                   | Dtype: auto, float32, float16, bfloat16         |
| `--resize`        | No       | None                   | Max image resolution — downscale proportionally |
| `--debug`         | No       | off                    | Show token counts, timing, and input details    |

### Default generation parameters

Each model has sensible defaults from its HuggingFace config. You can override them with `--max_tokens` and `--temperature`.

| Model                | max_tokens | temperature  | Notes                                          |
|----------------------|------------|--------------|------------------------------------------------|
| LFM2-VL-450M         | 256        | 0.0 (greedy) | No sampling defaults in HF config              |
| LFM2-VL-1.6B         | 256        | 0.0 (greedy) | Falls back to script defaults                  |
| LFM2-VL-3B           | 256        | 0.0 (greedy) | Falls back to script defaults                  |
| LFM2.5-VL-450M       | 256        | 0.0 (greedy) | Falls back to script defaults                  |
| LFM2.5-VL-1.6B       | 256        | 0.0 (greedy) | No sampling defaults in HF config              |
| gemma-3-4b-pt        | 256        | 0.0 (greedy) | Falls back to script defaults                  |
| gemma-3-4b-it        | 256        | 0.0 (greedy) | HF default is 1.0 but causes issues in float16 |
| gemma-4-E2B-it       | 256        | 0.0 (greedy) | Falls back to script defaults                  |
| Qwen3-VL-2B-Instruct | 256        | 0.7          | HF: do_sample=True, top_p=0.8, top_k=20        |
| Qwen3-VL-4B-Instruct | 256        | 0.7          | HF: do_sample=True, top_p=0.8, top_k=20        |

## How it works

The script auto-detects the model format from the directory contents:

1. **LoRA adapter**: If `adapter_config.json` is found, it downloads the base model from HuggingFace and applies your LoRA weights on top.
2. **Full model (.pt)**: If `model.pt` is found, it loads the PyTorch state_dict via `torch.load()` and builds the model from `config.json`.
3. **Full model (safetensors)**: Otherwise, it loads `.safetensors` weights directly via `from_pretrained()`.

The base model (for LoRA) is automatically cached in `~/.cache/huggingface/hub` after the first download.

### Image processing and tiling

Each model family handles images differently:

| Model                    | Image handling                                                                                                                                    |
|--------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------|
| **Liquid (LFM2/LFM2.5)** | Image tiling (`do_image_splitting`) with token budget (`min_image_tokens`, `max_image_tokens`). Splits large images into tiles for better detail. |
| **Gemma 3**              | Fixed resolution, no tiling.                                                                                                                      |
| **Qwen3-VL**             | Dynamic resolution — images are rescaled to fit within a pixel budget while preserving aspect ratio.                                              |

**Important**: Your model's training settings are always respected. If your model was trained with `do_image_splitting=False`, the script detects this from the saved `preprocessor_config.json` and does not override it. Default processor kwargs (like tiling) are only applied when the model's own config doesn't specify them.

## Troubleshooting

### Gemma 3: NaN / inf errors during generation

```
RuntimeError: probability tensor contains either `inf`, `nan` or element < 0
```

This is a known issue with Gemma 3 models caused by multiple bugs in the transformers library:

- **SDPA attention + padding** produces NaN on CPU/MPS. Fix: we use `attn_implementation="eager"` by default.
- **float16 overflow** in RMSNorm layers. Fix: on CUDA/CPU we use `bfloat16` by default, which matches the model's training precision. On MPS, the scripts fall back to `float16` because `bfloat16` is not universally supported there.
- **float32 embedding scale mismatch** — the model was trained with bfloat16-rounded scale values, so float32 produces slightly different logits that accumulate into NaN.

If you still see this error, try:

```bash
# Force greedy decoding (no sampling)
--temperature 0.0

# Or force bfloat16 explicitly (CUDA/CPU only)
--dtype bfloat16
```

### MPS (Apple Silicon): Out of memory

```
RuntimeError: Invalid buffer size: 8.01 GiB
```

MPS cannot allocate large contiguous memory blocks. The scripts load models on CPU first, then move to MPS incrementally. However, larger models (4B+) may still exceed available GPU memory.

### LFM2.5 on Apple Silicon: explicit MPS fallback required

`LiquidAI/LFM2.5-VL-*` models can hit MPS -> CPU fallback ops during image preprocessing. The scripts no longer enable that fallback automatically.

If you want to run those models on MPS, set the env var yourself before Python starts:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python models/LFM2.5-VL-1.6B/run.py ...
```

If you do not want MPS, pass `--device cpu`.

Workarounds:

```bash
# Run on CPU instead (slower but works)
--device cpu

# MPS uses float16 by default because bfloat16 is not always supported
--dtype float16
```

**Recommended**: Models up to ~2B parameters work well on MPS. For 4B+ models, use CPU or a CUDA GPU.

### MPS: Slow or stuck generation

Generation on MPS can appear stuck for larger models — it's not frozen, just slow. Vision models are especially heavy because the image encoder runs before text generation.

Tips:

- Use `--resize 384` to reduce image size and speed up processing
- Use `--max_tokens 50` for quick tests
- Use `--device cpu` which can be faster than MPS for some model sizes

### Gated models: "does not appear to have a file named model.safetensors"

```
OSError: google/gemma-3-4b-it does not appear to have a file named model.safetensors
```

This means HuggingFace authentication failed. The error message is misleading — the files exist but you don't have access. See the [HuggingFace authentication](#huggingface-authentication-required-for-some-models) section above.

Common causes:

- `HF_TOKEN` not set in your shell
- Token is a **fine-grained** token without read access — use a **Read** token instead
- License not accepted on the model's HuggingFace page

### LoRA + Gemma 3: Very slow on CPU

Gemma 3 (4B parameters) with a LoRA adapter on CPU is very slow — expect **5-15 minutes** per generation with images. This is because:

1. The 4B vision model is large for CPU inference
2. LoRA adds overhead vs the base model (adapter computations on every forward pass)
3. The vision encoder processes pixel values before text generation

For faster results:

- Use a **CUDA GPU** (`--device cuda`) — this is how Gemma 3 is meant to run
- Use **Liquid LFM2-VL-450M** on CPU — 10x smaller and much faster
- Use `--resize 384` to reduce image processing time
- Test text-only first (omit `--images`) to verify the model works

### General: `torch_dtype` deprecation warning

```
`torch_dtype` is deprecated! Use `dtype` instead!
```

This is a harmless warning from transformers >= 5.x. The parameter still works. You can safely ignore it.

## Project structure

```
transformers/
├── base.py                         # Shared inference utilities
├── models/
│   ├── LFM2-VL-450M/run.py        # LiquidAI LFM2-VL-450M
│   ├── LFM2.5-VL-1.6B/run.py      # LiquidAI LFM2.5-VL-1.6B
│   ├── gemma-3-4b-it/run.py        # Google Gemma 3 4B
│   ├── Qwen3-VL-2B-Instruct/run.py # Qwen3-VL 2B
│   └── Qwen3-VL-4B-Instruct/run.py # Qwen3-VL 4B
├── examples/                       # Ready-to-run bash examples
├── stored/                         # Downloaded model directories
│   ├── lf2-450-lora/              # LoRA adapter (.safetensors)
│   ├── qwen3-2b-lora/            # LoRA adapter (.safetensors)
│   ├── gemma-lora/                # LoRA adapter (.safetensors)
│   └── model_quantized_pt/        # Full model (.pt)
└── README.md
```
