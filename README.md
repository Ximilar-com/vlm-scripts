# Ximilar VLM Scripts

Example scripts for running your trained Vision-Language Models (VLMs) from the [Ximilar Platform](https://www.ximilar.com).

When you train a VLM on Ximilar, you can download the model and run it locally. This repository shows you how.

## Frameworks

### [HuggingFace + PEFT](huggingface/)

Run your models using Python with HuggingFace Transformers and PEFT (for LoRA adapters).

- Simple `run.py` script per model
- Supports both fully fine-tuned models and LoRA adapters
- Works on NVIDIA GPU (CUDA), Apple Silicon (MPS), and CPU

**Supported models**: LiquidAI LFM2-VL-450M, LiquidAI LFM2.5-VL-1.6B, Google Gemma 3 4B, Qwen3-VL 2B, Qwen3-VL 4B

See the [huggingface/README.md](huggingface/README.md) for setup, usage, and troubleshooting.
