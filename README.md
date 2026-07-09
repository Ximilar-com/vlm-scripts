# Ximilar VLM Scripts

Example scripts for running your trained Vision-Language Models (VLMs) from the [Ximilar Platform](https://www.ximilar.com).

When you train a VLM on Ximilar, you can download the model and run it locally. This repository shows you how.

Both frameworks download models automatically, driven by two environment variables: **`MODEL_DOWNLOAD_DIRECTORY`** (base directory for downloaded models, one subfolder per model) and a **per-model URL variable** (`MODEL_URL_HF_<MODEL>` for transformers, `MODEL_URL_GGUF_<MODEL>` for llama.cpp) holding the download URL from your model's details page at https://app.ximilar.com/platform/vlm/tasks/. Each framework README has an "Environment variables" section with the full list.

## Frameworks

### [Transformers](transformers/)

Run your models using Python with HuggingFace Transformers and PEFT (for LoRA adapters).

- Simple `run.py` script per model — no command-line arguments, just edit the constants at the top (prompt, images, tools)
- Models download automatically on first run: export `MODEL_DOWNLOAD_DIRECTORY` (base directory for downloaded models, one subfolder per model) and the `MODEL_URL_HF_<MODEL>` env variable with the download URL from your model's details page at https://app.ximilar.com/platform/vlm/tasks/
- Supports LoRA adapters (`.safetensors`), full models (`.safetensors`), and PyTorch exports (`.pt`)
- Auto-detects model format from directory contents
- Works on NVIDIA GPU (CUDA), Apple Silicon (MPS), and CPU

**Where to get the model download URL** — on your task's detail page, click the **download** action on the trained model, then **Copy link** in the dialog (the link is valid for 24 hours):

![Model download action on the task detail page](docs/media/model-download-url.png)

![Copy link in the Model Download dialog](docs/media/model-download-url-2.png)

**Supported models**: LiquidAI LFM2-VL (450M / 1.6B / 3B), LiquidAI LFM2.5-VL (450M / 1.6B), Google Gemma 3 4B (it / pt), Google Gemma 4 E2B, Qwen3-VL 2B / 4B (Instruct and Thinking)

**Agentic models** (thinking + tool calls, not just an answer): LiquidAI LFM2.5-VL, Google Gemma 4 E2B, Qwen3-VL Thinking — see [Agentic Models](transformers/README.md#agentic-models).

See the [transformers/README.md](transformers/README.md) for setup, usage, and troubleshooting.

### [llama.cpp](llama-cpp/)

Run quantized GGUF models locally using [llama.cpp](https://github.com/ggml-org/llama.cpp) — full vision support, fast inference, no torch.

- Simple `run.sh` shell script per model: a tiny Python helper downloads the model on first run, then `llama-cli` runs the prompt via `nix run nixpkgs#llama-cpp` (no manual llama.cpp install; `LLAMA_CLI` env var switches to a local build — see the [install guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/install.md))
- Models download automatically: export `MODEL_DOWNLOAD_DIRECTORY` and the `MODEL_URL_GGUF_<MODEL>` env variable with the download URL from your model's details page (the GGUF artifact contains the model + mmproj vision projector)
- Agentic models print their raw thinking + tool-call output; for parsed output and a simulated tool round use the transformers examples
- GPU acceleration on Apple Silicon (Metal); CUDA via a local llama.cpp build

**Supported models**: the same 12 models as the transformers examples — see [llama-cpp/README.md](llama-cpp/README.md) for setup, usage, and troubleshooting.
