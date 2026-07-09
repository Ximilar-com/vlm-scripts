# Ximilar VLM Scripts — llama.cpp

Simple shell scripts for running your trained VLM (Vision-Language) models from the [Ximilar Platform](https://www.ximilar.com) as GGUF with [llama.cpp](https://github.com/ggml-org/llama.cpp) — fast local inference on CPU, Apple Silicon (Metal), and NVIDIA GPUs.

Each model has its own `run.sh`. The script first downloads the model on first use (a tiny Python helper, `uv run download_model.py`), then runs the prompt with **llama-cli via `nix run nixpkgs#llama-cpp`** — that's the whole setup. Edit the variables at the top of the script to change the prompt or image.

## Supported Models

| Model                   | Script                                                                   | Env variable                            |
|-------------------------|--------------------------------------------------------------------------|------------------------------------------|
| LiquidAI LFM2-VL-450M   | [models/LFM2-VL-450M/run.sh](models/LFM2-VL-450M/run.sh)                 | `MODEL_URL_GGUF_LFM2_VL_450M`            |
| LiquidAI LFM2-VL-1.6B   | [models/LFM2-VL-1.6B/run.sh](models/LFM2-VL-1.6B/run.sh)                 | `MODEL_URL_GGUF_LFM2_VL_1_6B`            |
| LiquidAI LFM2-VL-3B     | [models/LFM2-VL-3B/run.sh](models/LFM2-VL-3B/run.sh)                     | `MODEL_URL_GGUF_LFM2_VL_3B`              |
| LiquidAI LFM2.5-VL-450M | [models/LFM2.5-VL-450M/run.sh](models/LFM2.5-VL-450M/run.sh)             | `MODEL_URL_GGUF_LFM2_5_450M`             |
| LiquidAI LFM2.5-VL-1.6B | [models/LFM2.5-VL-1.6B/run.sh](models/LFM2.5-VL-1.6B/run.sh)             | `MODEL_URL_GGUF_LFM2_5_1_6B`             |
| Google Gemma 3 4B PT    | [models/gemma-3-4b-pt/run.sh](models/gemma-3-4b-pt/run.sh)               | `MODEL_URL_GGUF_GEMMA_3_4B_PT`           |
| Google Gemma 3 4B       | [models/gemma-3-4b-it/run.sh](models/gemma-3-4b-it/run.sh)               | `MODEL_URL_GGUF_GEMMA_3_4B_IT`           |
| Google Gemma 4 E2B      | [models/gemma-4-E2B-it/run.sh](models/gemma-4-E2B-it/run.sh)             | `MODEL_URL_GGUF_GEMMA_4_E2B_IT`          |
| Qwen3-VL 2B             | [models/Qwen3-VL-2B-Instruct/run.sh](models/Qwen3-VL-2B-Instruct/run.sh) | `MODEL_URL_GGUF_QWEN3_VL_2B_INSTRUCT`    |
| Qwen3-VL 4B             | [models/Qwen3-VL-4B-Instruct/run.sh](models/Qwen3-VL-4B-Instruct/run.sh) | `MODEL_URL_GGUF_QWEN3_VL_4B_INSTRUCT`    |

**Agentic models** (thinking + tool calls): `models/Qwen3-VL-2B-Thinking/run.sh` (`MODEL_URL_GGUF_QWEN3_VL_2B_THINKING`), `models/Qwen3-VL-4B-Thinking/run.sh` (`MODEL_URL_GGUF_QWEN3_VL_4B_THINKING`), and `run_agentic.sh` next to the LFM2.5-VL and gemma-4-E2B-it scripts. These describe the tool in the system prompt and print the model's raw thinking / tool-call markers as-is (`-sp`). For **parsed** output and a **simulated tool-execution round**, use the [transformers examples](../transformers/README.md#agentic-models) — llama-cli is a one-shot CLI and keeps this side simple by design.

## Requirements

- [Nix](https://nixos.org/download/) — llama-cli runs via `nix run nixpkgs#llama-cpp` (fetched automatically on first use). To use a local llama.cpp install instead, follow the official install guide https://github.com/ggml-org/llama.cpp/blob/master/docs/install.md and set `export LLAMA_CLI=llama-cli` (any command prefix works).
- [uv](https://docs.astral.sh/uv/) with Python 3.12+ — only for the model download helper (`download_model.py`, stdlib-only, no dependencies).

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `MODEL_DOWNLOAD_DIRECTORY` | **Yes** | Base directory where models are downloaded and extracted. Each model gets its own subfolder (e.g. `./stored/LFM2.5-VL-1.6B-GGUF/`). |
| `MODEL_URL_GGUF_<MODEL>` | **Yes** (first run) | The model's download URL — one variable per model, listed in the table above (e.g. `MODEL_URL_GGUF_LFM2_5_450M`). Copy it from the model details page at https://app.ximilar.com/platform/vlm/tasks/; the link is valid for 24 hours. Not needed again once the model is downloaded. |
| `LLAMA_CLI` | No | Overrides how llama-cli is launched. Default: `nix run nixpkgs#llama-cpp --`. Set to `llama-cli` (or any command prefix) to use a local llama.cpp install. |

```bash
export MODEL_DOWNLOAD_DIRECTORY=./stored
export MODEL_URL_GGUF_LFM2_5_1_6B='https://...'
export LLAMA_CLI=llama-cli                  # optional: local llama.cpp instead of nix
```

## Usage

1. Open your task at **https://app.ximilar.com/platform/vlm/tasks/**, pick the trained model, and click its **download** action:

   ![Model download action on the task detail page](../docs/media/model-download-url.png)

2. In the dialog, click **Copy link** — the link is valid for **24 hours**:

   ![Copy link in the Model Download dialog](../docs/media/model-download-url-2.png)

3. Export **`MODEL_DOWNLOAD_DIRECTORY`** and the copied URL, then run the script from the `llama-cpp/` directory:

```bash
export MODEL_DOWNLOAD_DIRECTORY=./stored
export MODEL_URL_GGUF_LFM2_5_1_6B='https://...'
bash models/LFM2.5-VL-1.6B/run.sh
```

The GGUF artifact is an **archive containing two files**: the fine-tuned text model GGUF and the **mmproj** (vision projector) GGUF — both are required by llama-cli. On the first run it is downloaded and extracted into `$MODEL_DOWNLOAD_DIRECTORY/<model-name>-GGUF/`; later runs reuse it and skip the download. To re-download (e.g. after re-training), delete that subfolder.

To change what the script does, **edit the variables at the top**:

```bash
IMAGE="media/photo.jpg"
PROMPT="Describe the product in the image."
SYSTEM_PROMPT=""
MAX_TOKENS=256
TEMPERATURE=0.0
```

A sample image ships at [media/photo.jpg](media/photo.jpg).

## Troubleshooting

- **"Environment variable ... is not set"** — export `MODEL_DOWNLOAD_DIRECTORY` and the model's `MODEL_URL_GGUF_*` variable (URL from https://app.ximilar.com/platform/vlm/tasks/, valid 24 hours).
- **"Expected exactly one model .gguf and one mmproj .gguf"** — the downloaded archive didn't contain both files; check the error's file list and re-download the correct artifact.
- **`nix: command not found`** — install nix (https://nixos.org/download/) or install llama.cpp locally (https://github.com/ggml-org/llama.cpp/blob/master/docs/install.md) and set `LLAMA_CLI`.
- **Slow inference** — Metal is automatic on Apple Silicon; the nixpkgs package is CPU-only for NVIDIA, so use a CUDA llama.cpp build via `LLAMA_CLI`; lower `MAX_TOKENS` for quick tests.

## Project structure

```
llama-cpp/
├── download_model.py          # The only Python: download + extract the GGUF artifact (stdlib-only)
├── media/photo.jpg            # Sample image used by the scripts
├── models/<MODEL>/run.sh      # One shell script per model
└── models/<MODEL>/run_agentic.sh  # Agentic variant (LFM2.5, gemma-4): raw thinking + tool calls
```
