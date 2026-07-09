"""
Agentic inference utilities for Ximilar VLM models fine-tuned to emit
thinking + tool calls + a final answer.

Three families are supported ("qwen3vl", "lfm2.5", "gemma4" — each run
script declares its own FAMILY constant): Qwen3-VL-Thinking, LiquidAI
LFM2.5-VL, and Google Gemma 4. Each was fine-tuned to emit thinking and
tool calls in its own wire format. Parsing is delegated to HuggingFace's
native tokenizer.parse_response(), driven by the per-family response
schemas defined in this module (attached to the tokenizer by
run_agentic_inference). Requires transformers >= 5.1.

Scope: parse-only — no real tool execution. One generation goes in, a
ParsedGeneration (thinking / tool_calls / answer) comes out. When the model
decides to call a tool, the example scripts feed back a prepared simulated
result (see append_simulated_tool_results) and generate once more so the
model can produce its final answer.
"""

import json
import logging
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, List, Optional

import torch
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Response schemas (standalone port of the training pipeline's schemas)
# ---------------------------------------------------------------------------
# Hugging Face response schemas are intentionally dynamic JSON-like structures
# consumed directly by parse_response().
# See: https://huggingface.co/docs/transformers/main/en/chat_templating

type ResponseSchema = dict[str, object]


_GENERIC_CONTENT_SCHEMA: Final[ResponseSchema] = {
    "type": "object",
    "properties": {
        "role": {"const": "assistant"},
        "content": {
            "type": "string",
            "x-regex": r"^\s*(.*?)(?:<\|im_end\|>|<turn\|>|<end_of_turn>|<\|endoftext\|>|$)",
        },
    },
}

_LFM25_RESPONSE_SCHEMA: Final[ResponseSchema] = {
    "type": "object",
    "properties": {
        "role": {"const": "assistant"},
        "thinking": {
            "type": "string",
            "x-regex": r"<think>\n?(.*?)\n?</think>",
        },
        "content": {
            "type": "string",
            "x-regex": r"^(?!.*<\|tool_call_start\|>)(?:<think>.*?</think>)?\s*(.*?)(?:<\|im_end\|>|$)",
        },
        "tool_calls": {
            "type": "array",
            "x-regex": r"<\|tool_call_start\|>\[(.*?)\]<\|tool_call_end\|>",
            "x-regex-iterator": r"([A-Za-z_]\w*\s*\([^)]*\))",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"const": "function"},
                    "function": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "x-regex": r"^\s*([A-Za-z_]\w*)\s*\(",
                            },
                            "arguments": {
                                "type": "object",
                                "x-regex-substitutions": [
                                    (r"^[^(]*\(", "{"),
                                    (r"\)\s*$", "}"),
                                    (r"([A-Za-z_]\w*)\s*=", r'"\1":'),
                                    (r"\bTrue\b", "true"),
                                    (r"\bFalse\b", "false"),
                                    (r"\bNone\b", "null"),
                                    (r"'([^']*)'", r'"\1"'),
                                ],
                                "x-parser": "json",
                                "additionalProperties": True,
                            },
                        },
                        "required": ["name", "arguments"],
                    },
                },
                "required": ["function"],
            },
        },
    },
}

_QWEN3VL_RESPONSE_SCHEMA: Final[ResponseSchema] = {
    "type": "object",
    "properties": {
        "role": {"const": "assistant"},
        # Qwen3-VL-Thinking's chat template renders the assistant turn as
        # "<think>\n{reasoning}\n</think>\n\n{content}". During generation the prompt already
        # ends with "...<think>\n" (add_generation_prompt=True puts the opener in the PROMPT),
        # and only the generated suffix is decoded, so the opening <think> is routinely absent
        # from the raw text even though the model is thinking. Both regexes below therefore
        # treat the opener as optional and key off the closing </think> instead.
        # See: https://qwen.readthedocs.io/en/latest/getting_started/concepts.html
        "thinking": {
            "type": "string",
            "x-regex": r"(?s)(?:<think>)?\s*(.*?)\s*</think>",
        },
        "content": {
            "type": "string",
            "x-regex": r"(?s)^(?!.*<tool_call>)(?:.*?</think>\s*)?(.*?)\s*(?:<\|im_end\|>|<\|endoftext\|>|$)",
        },
        "tool_calls": {
            "type": "array",
            "x-regex-iterator": r"<tool_call>\s*(.*?)\s*</tool_call>",
            "items": {
                "type": "object",
                "x-parser": "json",
            },
        },
    },
}

_GEMMA4_RESPONSE_SCHEMA: Final[ResponseSchema] = {
    "type": "object",
    "properties": {
        "role": {"const": "assistant"},
        "thinking": {
            "type": "string",
            "x-regex": r"<\|channel>thought\n?(.*?)\n?<channel\|>",
        },
        "content": {
            "type": "string",
            "x-regex": r"^(?!.*<\|tool_call>)(?:.*?<channel\|>)?\s*(.*?)(?:<turn\|>|<\|endoftext\|>|$)",
        },
        "tool_calls": {
            "type": "array",
            "x-regex-iterator": r"<\|tool_call>(.*?)<tool_call\|>",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"const": "function"},
                    "function": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "x-regex": r"call:([A-Za-z_]\w*)\s*\{",
                            },
                            "arguments": {
                                "type": "object",
                                "x-regex": r"call:[A-Za-z_]\w*\s*(\{.*\})",
                                "x-parser": "gemma4-tool-call",
                                "additionalProperties": True,
                            },
                        },
                        "required": ["name", "arguments"],
                    },
                },
                "required": ["function"],
            },
        },
    },
}

_RESPONSE_SCHEMAS: Final[dict[str, ResponseSchema]] = {
    "generic": _GENERIC_CONTENT_SCHEMA,
    "gemma4": _GEMMA4_RESPONSE_SCHEMA,
    "lfm2.5": _LFM25_RESPONSE_SCHEMA,
    "qwen3vl": _QWEN3VL_RESPONSE_SCHEMA,
}


def response_schema_for_model(model_identifier: str) -> ResponseSchema:
    """Return a copy of the fallback response schema for a supported model id/family."""
    schema_key = _response_schema_key(model_identifier)
    if schema_key is None:
        raise ValueError(f"No response schema is registered for model family {model_identifier!r}.")
    return deepcopy(_RESPONSE_SCHEMAS[schema_key])


def ensure_model_response_schema(processor_or_tokenizer: object | None, *, model_family: str) -> bool:
    """Attach a fallback response_schema when a supported tokenizer lacks a native one."""
    tokenizer = _processor_tokenizer(processor_or_tokenizer)
    if tokenizer is None:
        return False
    if getattr(tokenizer, "response_schema", None) is not None:
        return False

    try:
        response_schema = response_schema_for_model(model_family)
    except ValueError:
        return False

    tokenizer.response_schema = response_schema
    return True


def _processor_tokenizer(processor_or_tokenizer: object | None) -> object | None:
    if processor_or_tokenizer is None:
        return None
    tokenizer = getattr(processor_or_tokenizer, "tokenizer", None)
    if tokenizer is not None:
        return tokenizer
    return processor_or_tokenizer


def _response_schema_key(model_identifier: str) -> str | None:
    normalized_identifier = model_identifier.lower()
    if "lfm2.5" in normalized_identifier:
        return "lfm2.5"
    is_qwen3vl_identifier = "qwen3-vl" in normalized_identifier or "qwen3vl" in normalized_identifier
    if is_qwen3vl_identifier and "instruct" in normalized_identifier:
        return "generic"
    if is_qwen3vl_identifier:
        return "qwen3vl"
    if "gemma-4" in normalized_identifier or "gemma4" in normalized_identifier:
        return "gemma4"
    if "gemma-3" in normalized_identifier or "gemma3" in normalized_identifier:
        return "generic"
    if "lfm2-vl" in normalized_identifier or normalized_identifier == "lfm2":
        return "generic"
    return None


# ---------------------------------------------------------------------------
# Parsed result
# ---------------------------------------------------------------------------


@dataclass
class ParsedGeneration:
    """Structured view of one agentic generation."""

    raw: str
    thinking: Optional[str] = None
    tool_calls: List[dict] = field(default_factory=list)
    answer: str = ""
    finish_reason: str = "stop"  # "stop" | "tool_call" | "parse_error"
    parse_errors: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parsing via tokenizer.parse_response()
# ---------------------------------------------------------------------------


def _normalize_tool_call(item: object) -> Optional[dict]:
    """Map one parse_response tool-call item to the flat {"name", "arguments"} shape.

    The lfm2.5 and gemma4 schemas emit the OpenAI-style wrapper
    {"type": "function", "function": {"name", "arguments"}}, while the
    qwen3vl schema emits the raw JSON the model produced, which is already
    flat. Returns None for anything else.
    """
    if not isinstance(item, dict):
        return None
    call = item.get("function", item)
    if not isinstance(call, dict) or not isinstance(call.get("name"), str):
        return None
    return {"name": call["name"], "arguments": call.get("arguments", {})}


def parse_agentic(raw: str, processor) -> ParsedGeneration:
    """Parse a raw agentic generation into thinking / tool_calls / answer.

    Delegates to processor.tokenizer.parse_response(raw); the response
    schema is attached to the tokenizer by run_agentic_inference().

    Never raises: a parse_response failure (including a malformed tool
    call, which fails the whole parse) is surfaced as a parse_error result
    with the raw text preserved, so one malformed generation can't crash
    the script.
    """
    if getattr(processor.tokenizer, "response_schema", None) is None:
        return ParsedGeneration(
            raw=raw,
            finish_reason="parse_error",
            parse_errors=[
                "tokenizer has no response_schema attached — call "
                "ensure_model_response_schema() first (run_agentic_inference does this automatically)"
            ],
        )

    try:
        result = processor.tokenizer.parse_response(raw)
    except Exception as exc:  # noqa: BLE001 - last-resort guard, see docstring
        logger.warning("tokenizer.parse_response failed: %s", exc)
        return ParsedGeneration(raw=raw, finish_reason="parse_error", parse_errors=[str(exc)])

    if isinstance(result, str):
        return ParsedGeneration(raw=raw, answer=result.strip())

    parsed = ParsedGeneration(raw=raw)
    thinking = result.get("thinking")
    if isinstance(thinking, str):
        parsed.thinking = thinking.strip()

    for item in result.get("tool_calls") or []:
        call = _normalize_tool_call(item)
        if call is None:
            parsed.parse_errors.append(f"unrecognized tool_call shape: {item!r}")
        else:
            parsed.tool_calls.append(call)

    # Successfully parsed tool calls win over content (a completed tool call
    # is the whole point of that turn); tool-call errors mark the result as
    # failed so callers surface them instead of a silently truncated answer.
    if parsed.tool_calls:
        parsed.finish_reason = "tool_call"
    else:
        parsed.answer = (result.get("content") or "").strip()
        if parsed.parse_errors:
            parsed.finish_reason = "parse_error"
    return parsed


# ---------------------------------------------------------------------------
# Tool formatting (for the prompt, not the response)
# ---------------------------------------------------------------------------


def format_tools(tools: List[dict], family: str) -> List[dict]:
    """Shape user-supplied tool definitions for a family's chat template.

    Input tools use the simple shape {"name", "description", "parameters"}
    (parameters is a JSON Schema object). Qwen3-VL and Gemma 4 expect the
    HuggingFace/OpenAI function-calling wrapper; LFM2.5's chat template
    embeds whatever it's given as-is and was fine-tuned on the flat shape.
    """
    if family == "lfm2.5":
        return [{"name": t["name"], "description": t.get("description", ""), "parameters": t.get("parameters", {})} for t in tools]
    return [
        {"type": "function", "function": {"name": t["name"], "description": t.get("description", ""), "parameters": t.get("parameters", {})}}
        for t in tools
    ]


# ---------------------------------------------------------------------------
# Simulated tool results (the examples don't execute real tools)
# ---------------------------------------------------------------------------


def append_simulated_tool_results(
        messages: list,
        tool_calls: List[dict],
        simulated_results: dict,
        root_dir: Optional[Path] = None,
) -> List[Image.Image]:
    """Append the assistant's tool-call turn plus simulated tool results to messages.

    The agentic models were fine-tuned to have their tool calls really
    executed, but these examples can't do that — so when the model decides
    to call a tool, feed it a prepared result and generate once more for
    the final answer.

    simulated_results maps tool name -> result: a string, or a
    JSON-serializable dict. A dict may carry an optional "images" key with
    local image paths (resolved against root_dir when relative) — those
    are attached to the tool message as media and returned so the caller
    can extend the image list passed to the processor. Tool names without
    a prepared result get {"status": "ok"}.

    Returns:
        PIL images attached to the tool results (append them to the images
        passed to run_agentic_inference for the next turn).
    """
    messages.append({
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"type": "function", "function": {"name": call["name"], "arguments": call["arguments"]}}
            for call in tool_calls
        ],
    })

    tool_images: List[Image.Image] = []
    for call in tool_calls:
        result = simulated_results.get(call["name"], {"status": "ok"})
        image_paths = []
        if isinstance(result, dict):
            result = dict(result)  # don't mutate the caller's constant
            image_paths = result.pop("images", [])
        text = result if isinstance(result, str) else json.dumps(result)

        content = []
        for path in image_paths:
            path = Path(path)
            if root_dir is not None and not path.is_absolute():
                path = Path(root_dir) / path
            image = Image.open(path).convert("RGB")
            tool_images.append(image)
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": text})
        messages.append({"role": "tool", "name": call["name"], "content": content})

    return tool_images


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def run_agentic_inference(
        model,
        processor,
        messages: list,
        images: List[Image.Image],
        family: str,
        tools: Optional[List[dict]] = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
        enable_thinking: bool = True,
        debug: bool = False,
) -> str:
    """Run one agentic generation and return the raw decoded text.

    Differs from base.run_inference in three ways agentic models need:
    1. Passes `tools=` (and gemma-4's `enable_thinking` / LFM2.5's
       `keep_past_thinking`) to apply_chat_template so the model sees its
       tool definitions and thinking mode.
    2. Decodes with skip_special_tokens=False so markers like <think>,
       <|tool_call_start|>, and <|channel> survive for parsing.
    3. Returns the raw string un-parsed — call parse_agentic() on it.

    Also attaches the family's fallback response schema to the tokenizer
    (idempotent) so a later parse_agentic() call just works.
    """
    if ensure_model_response_schema(processor, model_family=family):
        logger.info("Attached fallback response schema for family %r", family)

    template_kwargs = {}
    if tools:
        template_kwargs["tools"] = format_tools(tools, family)
    if family == "gemma4":
        template_kwargs["enable_thinking"] = enable_thinking
    elif family == "lfm2.5":
        template_kwargs["keep_past_thinking"] = True

    text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **template_kwargs)

    if debug:
        print(f"\n--- Debug: Text prompt ---\n{text_prompt}\n")

    processor_kwargs = dict(text=text_prompt, return_tensors="pt", padding=True)
    if images:
        processor_kwargs["images"] = images
    inputs = processor(**processor_kwargs)
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    if debug:
        print(f"--- Debug: Input tokens: {input_len}")

    do_sample = temperature > 0.0
    generate_kwargs = dict(**inputs, max_new_tokens=max_tokens, do_sample=do_sample)
    if do_sample:
        generate_kwargs["temperature"] = temperature

    start_time = time.time()
    with torch.inference_mode():
        outputs = model.generate(**generate_kwargs)
    elapsed = time.time() - start_time

    output_len = outputs.shape[1]
    completion_len = output_len - input_len
    generated_ids = outputs[0][input_len:]
    raw = processor.tokenizer.decode(generated_ids, skip_special_tokens=False).strip()

    if debug:
        print(f"--- Debug: Output tokens: {completion_len}")
        print(f"--- Debug: Generation time: {elapsed:.2f}s ({completion_len / elapsed:.1f} tokens/sec)")
        print()

    return raw


# ---------------------------------------------------------------------------
# Output display
# ---------------------------------------------------------------------------


def print_parsed(parsed: ParsedGeneration, show_raw: bool = False) -> None:
    """Pretty-print a ParsedGeneration: Thinking / Tool calls / Answer blocks."""
    C = "\033[36m"  # cyan
    G = "\033[32m"  # green
    Y = "\033[33m"  # yellow
    R = "\033[0m"  # reset

    if parsed.thinking:
        print(f"{C}--- Thinking ---{R}")
        print(parsed.thinking)
        print()

    if parsed.tool_calls:
        print(f"{C}--- Tool calls ({len(parsed.tool_calls)}) ---{R}")
        for call in parsed.tool_calls:
            print(f"  {G}{call['name']}{R}({json.dumps(call['arguments'])})")
        print()

    print(f"{C}--- Answer ---{R}")
    print(parsed.answer or "(none)")

    if parsed.finish_reason == "parse_error":
        print()
        print(f"{Y}--- Parse errors ---{R}")
        for err in parsed.parse_errors:
            print(f"  {err}")

    if show_raw:
        print()
        print(f"{C}--- Raw generation ---{R}")
        print(parsed.raw)
