# Ximilar VLM Scripts

Example scripts for running your trained Vision-Language Models (VLMs) from the [Ximilar Platform](https://www.ximilar.com).

When you train a VLM on Ximilar, you can download the model and run it locally. This repository shows you how.

## Frameworks

### [Transformers](transformers/)

Run your models using Python with HuggingFace Transformers and PEFT (for LoRA adapters).

- Simple `run.py` script per model
- Supports LoRA adapters (`.safetensors`), full models (`.safetensors`), and PyTorch exports (`.pt`)
- Auto-detects model format from directory contents
- Works on NVIDIA GPU (CUDA), Apple Silicon (MPS), and CPU

**Supported models**: LiquidAI LFM2-VL-450M, LiquidAI LFM2.5-VL-1.6B, Google Gemma 3 4B, Qwen3-VL 2B, Qwen3-VL 4B

See the [transformers/README.md](transformers/README.md) for setup, usage, and troubleshooting.

### [llama.cpp](llama-cpp/)

Run quantized GGUF models locally using [llama.cpp](https://github.com/ggml-org/llama.cpp) — full vision support, fast inference.

- Single command with `llama-mtmd-cli`
- Full image/vision support (unlike Ollama which doesn't support LFM2-VL vision yet)
- GPU acceleration on Apple Silicon (Metal) and NVIDIA (CUDA)

**Supported models**: LiquidAI LFM2-VL-450M (GGUF quantized)

See the [llama-cpp/README.md](llama-cpp/README.md) for setup, usage, and troubleshooting.
