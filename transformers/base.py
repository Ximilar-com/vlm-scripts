"""
Shared utilities for running Ximilar VLM models.

This module provides all the building blocks for downloading a model,
loading it, processing images, and running vision-language inference.
Each model's run.py script imports from here and only defines
model-specific constants.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image


def _patch_torch_autocast_compat(torch_module=None) -> bool:
    """Make older torch builds accept the newer transformers autocast call style."""
    torch_module = torch if torch_module is None else torch_module

    try:
        torch_module.is_autocast_enabled("cpu")
        return False
    except TypeError:
        original = torch_module.is_autocast_enabled

        def is_autocast_enabled(device_type=None):
            return original()

        torch_module.is_autocast_enabled = is_autocast_enabled
        return True


_patch_torch_autocast_compat()

from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.image_utils import load_image

import client

logger = logging.getLogger(__name__)

# Suppress noisy HTTP request logs from huggingface_hub / httpx / httpcore
for _name in ("httpx", "httpcore", "urllib3", "huggingface_hub.file_download"):
    logging.getLogger(_name).setLevel(logging.WARNING)

# Env var naming the base directory all models are downloaded/extracted into.
MODEL_DOWNLOAD_DIRECTORY_ENV = "MODEL_DOWNLOAD_DIRECTORY"

MODEL_DETAILS_PAGE = "https://app.ximilar.com/platform/vlm/tasks/"


# ---------------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------------


def ensure_model(model_url_env: str, model_id: str) -> str:
    """Return a local model directory, downloading the model on first use.

    Two environment variables drive the download:
    - MODEL_DOWNLOAD_DIRECTORY (required): base directory where models are
      downloaded and extracted; each model gets its own subfolder named
      after the model.
    - The env var named by model_url_env (required on first download): the
      model archive URL — copy it from your model's details page at
      https://app.ximilar.com/platform/vlm/tasks/ (the link is valid for
      24 hours).

    Later runs find the extracted subfolder and skip the download entirely.

    Args:
        model_url_env: Name of the env var holding the model download URL
            (e.g. "MODEL_URL_HF_LFM2_5_450M").
        model_id: HuggingFace model ID (e.g. "LiquidAI/LFM2.5-VL-450M");
            its last segment names the per-model subfolder.

    Returns:
        Path to the local model directory, ready for load_model().
    """
    base_dir = os.environ.get(MODEL_DOWNLOAD_DIRECTORY_ENV)
    if not base_dir:
        raise RuntimeError(
            f"Environment variable {MODEL_DOWNLOAD_DIRECTORY_ENV} is not set.\n"
            f"Set it to the base directory where models should be downloaded and extracted\n"
            f"(each model gets its own subfolder), e.g.:\n"
            f"    export {MODEL_DOWNLOAD_DIRECTORY_ENV}=./stored"
        )

    cache_dir = Path(base_dir).expanduser().resolve() / model_id.split("/")[-1]
    if cache_dir.is_dir():
        try:
            model_dir = client.resolve_model_dir(cache_dir)
            client.validate_model_dir(model_dir)
            logger.info("Using cached model: %s", model_dir)
            return str(model_dir)
        except RuntimeError:
            logger.warning("Cached model at %s is incomplete — re-downloading", cache_dir)

    download_url = os.environ.get(model_url_env)
    if not download_url:
        raise RuntimeError(
            f"Environment variable {model_url_env} is not set.\n"
            f"Set it to your model's download URL, e.g.:\n"
            f"    export {model_url_env}='https://...'\n"
            f"You can copy the URL from the model details page at {MODEL_DETAILS_PAGE}\n"
            f"(open your task, click the download action on the trained model, then Copy link)."
        )

    logger.info("Downloading model to %s ...", cache_dir)
    model_dir = client.download_model_from_url(download_url, cache_dir)
    logger.info("Model downloaded: %s", model_dir)
    return str(model_dir)


# ---------------------------------------------------------------------------
# Device and dtype helpers
# ---------------------------------------------------------------------------


def resolve_device(device: str) -> str:
    """Resolve 'auto' to the best available device.

    Priority: CUDA > MPS (Apple Silicon) > CPU.

    Args:
        device: Device string from user ("auto", "cpu", "cuda", "cuda:0", "mps").

    Returns:
        Resolved device string (e.g. "cuda", "cpu").
    """
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def resolve_dtype(device: str, dtype: str = "auto") -> torch.dtype:
    """Pick the optimal dtype for the given device, or use explicit override.

    When dtype is "auto":
      CUDA   -> bfloat16 (fast, low memory)
      MPS    -> bfloat16 (matches training precision, avoids NaN with Gemma)
      CPU    -> bfloat16 (matches training precision for all supported models)

    Args:
        device: Device string (e.g. "cuda", "cpu", "mps").
        dtype: Explicit dtype string or "auto" to resolve from device.

    Returns:
        Resolved torch.dtype.
    """
    if dtype != "auto":
        if dtype not in _DTYPE_MAP:
            raise ValueError(f"Unknown dtype '{dtype}'. Supported: {list(_DTYPE_MAP.keys())}")
        return _DTYPE_MAP[dtype]
    # bfloat16 is the safest default for all devices — it matches training
    # precision and avoids NaN issues with Gemma 3 (embedding scale mismatch
    # in float32, overflow in float16). CPU and MPS have software bfloat16 support.
    return torch.bfloat16


# ---------------------------------------------------------------------------
# Image loading and resizing
# ---------------------------------------------------------------------------


def load_images(sources: List[str], max_size: Optional[int] = None) -> List[Image.Image]:
    """Load images from file paths or URLs, with optional proportional downscaling.

    Images are converted to RGB. If max_size is set, images larger than
    max_size in either dimension are downscaled while preserving aspect ratio.
    Images smaller than max_size are left unchanged (never upscaled).

    Args:
        sources: List of file paths or URLs to load.
        max_size: Maximum dimension (width or height) in pixels. None to skip resizing.

    Returns:
        List of PIL Image objects.
    """
    images = []
    for src in sources:
        logger.info("Loading image: %s", src)
        img = load_image(src).convert("RGB")
        original_size = img.size

        # Proportional downscale: shrink to fit within max_size x max_size box
        if max_size and (img.width > max_size or img.height > max_size):
            scale = min(max_size / img.width, max_size / img.height)
            new_w = int(img.width * scale)
            new_h = int(img.height * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            logger.info("  Resized %dx%d -> %dx%d", original_size[0], original_size[1], new_w, new_h)
        else:
            logger.info("  Size: %dx%d", img.width, img.height)

        images.append(img)
    return images


# ---------------------------------------------------------------------------
# Message building
# ---------------------------------------------------------------------------


def build_messages(
        images: List[Image.Image],
        user_prompt: str,
        system_prompt: Optional[str] = None,
) -> list:
    """Build chat messages in OpenAI format with PIL images.

    The message format is compatible with HuggingFace's apply_chat_template().
    Images are passed as PIL objects (not URLs) so the processor can handle
    them directly.

    Args:
        images: List of PIL Image objects to include in the user message.
        user_prompt: The text prompt for the model.
        system_prompt: Optional system instruction prepended to the conversation.

    Returns:
        List of message dicts ready for apply_chat_template().
    """
    messages = []

    # System message (optional)
    if system_prompt:
        messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})

    # User message: images first, then text prompt
    user_content = [{"type": "image", "image": img} for img in images]
    user_content.append({"type": "text", "text": user_prompt})
    messages.append({"role": "user", "content": user_content})

    return messages


# ---------------------------------------------------------------------------
# Config display
# ---------------------------------------------------------------------------


def print_config(
        model_id: str,
        model_path: str,
        processor,
        images: List[Image.Image],
        device: str,
        dtype: torch.dtype,
        max_tokens: int,
        temperature: float,
        resize: Optional[int],
        user_prompt: str,
        system_prompt: Optional[str],
) -> None:
    """Print a summary of all inference parameters before running the model.

    Shows model info, device, generation settings, image details, prompt
    previews, and effective processor settings (tiling, token budgets, etc.).

    Args:
        model_id: HuggingFace model ID.
        model_path: Local path to the model.
        processor: Loaded HuggingFace processor.
        images: List of loaded PIL images.
        device: Device string.
        dtype: Torch dtype.
        max_tokens: Max tokens to generate.
        temperature: Sampling temperature.
        resize: Max image resolution (None if not set).
        user_prompt: User prompt text.
        system_prompt: System prompt text (or None).
    """
    is_lora = (Path(model_path) / "adapter_config.json").exists()
    is_pt = (Path(model_path) / "model.pt").exists()

    # ANSI color codes for terminal output
    G = "\033[32m"  # green
    C = "\033[36m"  # cyan
    R = "\033[0m"  # reset

    def kv(key: str, value: object) -> str:
        """Format a key-value line with green key."""
        return f"  {G}{key + ':':19s}{R} {value}"

    print("=" * 60)
    print(f"  {C}Inference Configuration{R}")
    print("=" * 60)

    # Model
    print(kv("Model ID", model_id))
    print(kv("Model path", model_path))
    model_type_str = "LoRA adapter" if is_lora else "Full model (.pt)" if is_pt else "Full model (safetensors)"
    print(kv("Model type", model_type_str))

    # Device and dtype
    print(kv("Device", device))
    print(kv("Dtype", dtype))

    # Generation
    do_sample = temperature > 0.0
    print(kv("Max tokens", max_tokens))
    print(kv("Temperature", f"{temperature} ({'sampling' if do_sample else 'greedy'})"))

    # Image processor settings — read effective values from the loaded processor
    ip = getattr(processor, "image_processor", None)
    if ip is not None:
        print(f"  {C}--- Image Processor ---{R}")
        attrs = [
            ("do_image_splitting", "Image tiling"),
            ("min_image_tokens", "Min image tokens"),
            ("max_image_tokens", "Max image tokens"),
            ("min_pixels", "Min pixels"),
            ("max_pixels", "Max pixels"),
            ("merge_size", "Merge size"),
            ("do_resize", "Auto resize"),
        ]
        for attr, label in attrs:
            value = getattr(ip, attr, None)
            if value is not None:
                print(kv(label, value))

    # Images
    print(f"  {C}--- Images ({len(images)}) ---{R}")
    if resize:
        print(kv("Resize", f"{resize}px max"))
    for i, img in enumerate(images):
        print(kv(f"Image {i + 1}", f"{img.width}x{img.height}"))

    # Prompts — show full text
    print(f"  {C}--- Prompts ---{R}")
    if system_prompt:
        print(kv("System", system_prompt))
    print(kv("User", user_prompt))

    print("=" * 60)
    print()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _read_saved_config(model_path: str) -> dict:
    """Read processor config keys already saved in the model directory.

    Checks preprocessor_config.json and tokenizer_config.json. Returns a
    merged dict of all keys found. For paths that don't exist (e.g. HuggingFace
    hub IDs), returns an empty dict.

    Args:
        model_path: Local path to model directory.

    Returns:
        Dict of config keys and values from saved config files.
    """
    config = {}
    base = Path(model_path)
    if not base.is_dir():
        return config
    for fname in ("preprocessor_config.json", "tokenizer_config.json"):
        config_path = base / fname
        if config_path.exists():
            with open(config_path) as f:
                config.update(json.load(f))
    return config


def _filter_processor_kwargs(processor_kwargs: dict, model_path: str) -> dict:
    """Filter processor kwargs to avoid overriding the model's saved config.

    When a model was trained with specific processor settings (e.g.
    do_image_splitting=False), those settings are saved in its config files.
    Passing a conflicting kwarg would override them. This function removes
    any kwarg that is already present in the saved config, so the model's
    own training settings are respected.

    Args:
        processor_kwargs: Default kwargs to pass to AutoProcessor.from_pretrained().
        model_path: Local path to model directory (checked for saved configs).

    Returns:
        Filtered dict with only kwargs not already in the saved config.
    """
    saved_config = _read_saved_config(model_path)
    filtered = {k: v for k, v in processor_kwargs.items() if k not in saved_config}

    # Log what was skipped so users understand the behavior
    skipped = {k: v for k, v in processor_kwargs.items() if k in saved_config}
    if skipped:
        for k, v in skipped.items():
            logger.info("Using saved config %s=%s (default %s skipped)", k, saved_config[k], v)

    return filtered


def load_model(
        model_id: str,
        model_path: str,
        device: str,
        dtype: torch.dtype,
        processor_kwargs: Optional[dict] = None,
        auto_model_class=AutoModelForImageTextToText,
) -> Tuple:
    """Load a model (full or LoRA) and its processor.

    Automatically detects whether model_path is a LoRA adapter or a fully
    fine-tuned model by checking for adapter_config.json:

    - LoRA adapter: Downloads the base model from HuggingFace (model_id),
      then applies the LoRA weights from model_path on top.
    - Full model: Loads everything directly from model_path.

    Args:
        model_id: HuggingFace model ID for the base model (e.g. "LiquidAI/LFM2.5-VL-1.6B").
        model_path: Local path to the downloaded model directory.
        device: Device to load on ("cpu", "cuda", "mps", etc.).
        dtype: Torch dtype for model weights (e.g. torch.bfloat16).
        processor_kwargs: Optional kwargs for AutoProcessor.from_pretrained()
            (e.g. min_image_tokens, padding_side).
        auto_model_class: HuggingFace auto class used to load the model
            (default AutoModelForImageTextToText). Some families use a
            different class, e.g. gemma-4 uses AutoModelForMultimodalLM.

    Returns:
        Tuple of (model, processor) ready for inference.
    """
    is_lora = (Path(model_path) / "adapter_config.json").exists()
    is_pt = (Path(model_path) / "model.pt").exists()

    # Filter processor kwargs: skip defaults that are already saved in the
    # model's config files. This prevents overriding settings from training
    # (e.g. a model trained with do_image_splitting=False should keep it).
    _processor_kwargs = _filter_processor_kwargs(processor_kwargs or {}, model_path)

    # Pass HF_TOKEN explicitly for gated models (e.g. google/gemma-3-4b-it)
    hf_token = os.environ.get("HF_TOKEN")

    # MPS can't allocate large contiguous buffers, so load on CPU first then move.
    # CUDA and CPU can load directly with device_map.
    load_device = "cpu" if "mps" in device else device

    if is_lora:
        # LoRA adapter: load base model from HuggingFace, then apply adapter
        from peft import PeftModel

        logger.info("Detected LoRA adapter (adapter_config.json found)")
        logger.info("Loading base model from HuggingFace: %s", model_id)
        model = auto_model_class.from_pretrained(
            model_id,
            device_map=load_device,
            torch_dtype=dtype,
            token=hf_token,
            attn_implementation="eager",  # avoids SDPA NaN with padding (Gemma 3 bug)
        )

        logger.info("Applying LoRA adapter: %s", model_path)
        model = PeftModel.from_pretrained(model, model_path, is_trainable=False)
        model.eval()

        # Move to MPS after loading if needed
        if "mps" in device:
            logger.info("Moving model to MPS...")
            model = model.to(device)

        # Use processor from LoRA dir if it has its own config, otherwise from base
        has_local_processor = (Path(model_path) / "preprocessor_config.json").exists()
        processor_path = model_path if has_local_processor else model_id
        processor = AutoProcessor.from_pretrained(processor_path, token=hf_token, **_processor_kwargs)
    elif is_pt:
        # PyTorch .pt file: build model from config, then load state_dict
        from transformers import AutoConfig

        pt_file = Path(model_path) / "model.pt"
        logger.info("Detected PyTorch weights (model.pt found)")
        logger.info("Loading state_dict from: %s", pt_file)
        state_dict = torch.load(pt_file, map_location=load_device, weights_only=True)

        config = AutoConfig.from_pretrained(model_path)
        model = auto_model_class.from_config(config, attn_implementation="eager")
        model.load_state_dict(state_dict)
        del state_dict  # free memory before dtype/device transfer
        model = model.to(dtype=dtype, device=load_device)
        model.eval()

        if "mps" in device:
            logger.info("Moving model to MPS...")
            model = model.to(device)

        processor = AutoProcessor.from_pretrained(model_path, **_processor_kwargs)
    else:
        # Fully fine-tuned model (safetensors): load directly from the directory
        logger.info("Loading full model: %s", model_path)
        model = auto_model_class.from_pretrained(
            model_path,
            device_map=load_device,
            torch_dtype=dtype,
            attn_implementation="eager",  # avoids SDPA NaN with padding (Gemma 3 bug)
        )
        model.eval()

        if "mps" in device:
            logger.info("Moving model to MPS...")
            model = model.to(device)

        processor = AutoProcessor.from_pretrained(model_path, **_processor_kwargs)

    logger.info("Model loaded on %s with dtype %s", device, dtype)
    return model, processor


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def run_inference(
        model,
        processor,
        messages: list,
        images: List[Image.Image],
        max_tokens: int = 256,
        temperature: float = 0.0,
        debug: bool = False,
) -> str:
    """Run inference on a loaded model and return the generated text.

    Steps:
    1. Convert messages to a text prompt using the model's chat template.
    2. Tokenize the prompt and images into model inputs.
    3. Generate output tokens.
    4. Decode the generated tokens back to text.

    Args:
        model: Loaded HuggingFace model (or PeftModel for LoRA).
        processor: Loaded HuggingFace processor (tokenizer + image processor).
        messages: Chat messages from build_messages().
        images: List of PIL images (same ones used in messages).
        max_tokens: Maximum number of new tokens to generate.
        temperature: Sampling temperature. 0.0 = greedy (deterministic).
        debug: If True, print token counts and timing information.

    Returns:
        Generated text string (stripped of special tokens).
    """
    # Step 1: Apply chat template to convert messages -> text prompt
    text_prompt = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )

    if debug:
        print(f"\n--- Debug: Text prompt ---\n{text_prompt}\n")

    # Step 2: Tokenize text and process images into model inputs
    processor_kwargs = dict(text=text_prompt, return_tensors="pt", padding=True)
    if images:
        processor_kwargs["images"] = images

    inputs = processor(**processor_kwargs)

    # Move all tensors to the model's device (GPU/CPU/MPS)
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    input_len = inputs["input_ids"].shape[1]

    if debug:
        print(f"--- Debug: Input tokens: {input_len}")
        for key, val in inputs.items():
            if hasattr(val, "shape"):
                print(f"  {key}: shape={val.shape}, dtype={val.dtype}")

    # Step 3: Generate output tokens
    do_sample = temperature > 0.0
    generate_kwargs = dict(**inputs, max_new_tokens=max_tokens, do_sample=do_sample)
    if do_sample:
        generate_kwargs["temperature"] = temperature

    start_time = time.time()
    with torch.inference_mode():
        outputs = model.generate(**generate_kwargs)
    elapsed = time.time() - start_time

    # Step 4: Decode — strip the input tokens, keep only generated output
    output_len = outputs.shape[1]
    completion_len = output_len - input_len
    generated_ids = outputs[0][input_len:]
    result = processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    if debug:
        print(f"--- Debug: Output tokens: {completion_len}")
        print(f"--- Debug: Total tokens: {output_len} (input={input_len} + completion={completion_len})")
        print(f"--- Debug: Generation time: {elapsed:.2f}s ({completion_len / elapsed:.1f} tokens/sec)")
        print()

    return result
