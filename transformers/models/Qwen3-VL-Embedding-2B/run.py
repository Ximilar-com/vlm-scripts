#!/usr/bin/env python3
"""Run a small text/image embedding example with Qwen3-VL-Embedding-2B."""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel, Qwen3VLPreTrainedModel

from base import ensure_model


MODEL_ID = "Qwen/Qwen3-VL-Embedding-2B"
MODEL_URL_ENV = "MODEL_URL_HF_QWEN3_VL_EMBEDDING_2B"
INSTRUCTION = "Represent the user's input."
QUERY = "A product photographed on a table."
DOCUMENT = "A product shown in the image."
IMAGE_PATH = ROOT_DIR / "media/photo.jpg"


def embed(
    model: Qwen3VLPreTrainedModel,
    processor: AutoProcessor,
    text: str,
    image: Image.Image | None = None,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Create one normalized embedding for text, an image, or both."""
    content = []
    if image is not None:
        content.append({"type": "image"})
    content.append({"type": "text", "text": text})
    messages = [
        {"role": "system", "content": [{"type": "text", "text": INSTRUCTION}]},
        {"role": "user", "content": content},
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=prompt,
        images=[image] if image is not None else None,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    print(f"Input tokens: {inputs['attention_mask'].sum().item()}")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        hidden_state = model(**inputs).last_hidden_state

    attention_mask = inputs["attention_mask"]
    last_token = attention_mask.sum(dim=1) - 1
    return hidden_state[torch.arange(hidden_state.shape[0], device=device), last_token]


def main() -> None:
    model_path = ensure_model(MODEL_URL_ENV, MODEL_ID)
    dtype = torch.float32
    device = torch.device("cpu")
    is_lora = (Path(model_path) / "adapter_config.json").exists()
    processor = AutoProcessor.from_pretrained(model_path, padding_side="right")
    if is_lora:
        from peft import PeftModel
        model = Qwen3VLModel.from_pretrained(MODEL_ID, torch_dtype=dtype)
        model = PeftModel.from_pretrained(model, model_path, is_trainable=False)
    else:
        model = Qwen3VLPreTrainedModel.from_pretrained(model_path, torch_dtype=dtype)
    model = model.to(device).eval()

    print("Processing query")
    query_embedding = embed(model, processor, QUERY, device=device)
    print("Processing document")
    document_embedding = embed(model, processor, DOCUMENT, image=Image.open(IMAGE_PATH).convert("RGB"), device=device)
    similarity = F.cosine_similarity(query_embedding, document_embedding)

    print(f"Query embedding shape: {tuple(query_embedding.shape)}")
    print(f"Document embedding shape: {tuple(document_embedding.shape)}")
    print(f"Cosine similarity: {similarity.item():.4f}")
    print(f"First 8 query values: {query_embedding[0, :8].float().cpu().tolist()}")


if __name__ == "__main__":
    main()
