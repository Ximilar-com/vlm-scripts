"""Offline tests for the shared example modules — no model weights or network needed.

This is the single test file for the repo. It covers:
- parse_agentic() + the per-family response schemas, exercised through the
  real transformers parse_response machinery via a weightless tokenizer
- response-schema selection by model id / family
- base.ensure_model() error handling for a missing download-URL env var
"""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from agentic import (
    _response_schema_key,
    append_simulated_tool_results,
    ensure_model_response_schema,
    parse_agentic,
    response_schema_for_model,
)
from base import ensure_model


def _bare_tokenizer() -> PreTrainedTokenizerFast:
    return PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]")),
        unk_token="[UNK]",
    )


def _processor_for(family: str) -> SimpleNamespace:
    tokenizer = _bare_tokenizer()
    assert ensure_model_response_schema(tokenizer, model_family=family)
    return SimpleNamespace(tokenizer=tokenizer)


# ---------------------------------------------------------------------------
# lfm2.5
# ---------------------------------------------------------------------------


def test_lfm25_tool_call():
    raw = (
        "<think>\nThe region needs cropping.\n</think>\n"
        "<|tool_call_start|>[cut_object(bbox=[0.0, 0.0, 0.998, 0.398], label='region')]<|tool_call_end|><|im_end|>"
    )
    parsed = parse_agentic(raw, _processor_for("lfm2.5"))

    assert parsed.finish_reason == "tool_call"
    assert parsed.thinking == "The region needs cropping."
    assert parsed.tool_calls == [
        {"name": "cut_object", "arguments": {"bbox": [0.0, 0.0, 0.998, 0.398], "label": "region"}}
    ]
    assert parsed.answer == ""


def test_lfm25_answer_only():
    raw = "<think>\nJust a red shoe.\n</think>\nA red running shoe.<|im_end|>"
    parsed = parse_agentic(raw, _processor_for("lfm2.5"))

    assert parsed.finish_reason == "stop"
    assert parsed.thinking == "Just a red shoe."
    assert parsed.answer == "A red running shoe."
    assert parsed.tool_calls == []


@pytest.mark.parametrize("family", ["lfm2.5", "qwen3vl", "gemma4"])
def test_plain_response_without_markers(family):
    # A checkpoint may skip thinking/tool-call markers entirely — the plain
    # text must simply become the answer.
    raw = "A red running shoe on a white background."
    parsed = parse_agentic(raw, _processor_for(family))

    assert parsed.finish_reason == "stop"
    assert parsed.thinking is None
    assert parsed.tool_calls == []
    assert parsed.answer == "A red running shoe on a white background."


# ---------------------------------------------------------------------------
# qwen3vl
# ---------------------------------------------------------------------------


def test_qwen3vl_tool_call():
    raw = (
        "<think>\nI should classify this.\n</think>\n\n"
        '<tool_call>\n{"name": "submit_product_classification", "arguments": {"category": "shoes", "price": 29.99}}\n</tool_call><|im_end|>'
    )
    parsed = parse_agentic(raw, _processor_for("qwen3vl"))

    assert parsed.finish_reason == "tool_call"
    assert parsed.thinking == "I should classify this."
    assert parsed.tool_calls == [
        {"name": "submit_product_classification", "arguments": {"category": "shoes", "price": 29.99}}
    ]


def test_qwen3vl_missing_think_opener():
    # add_generation_prompt=True leaves the <think> opener in the prompt, so
    # the generated suffix starts mid-thought and only has the closer.
    raw = "The image shows a shoe.\n</think>\n\nA red running shoe.<|im_end|>"
    parsed = parse_agentic(raw, _processor_for("qwen3vl"))

    assert parsed.finish_reason == "stop"
    assert parsed.thinking == "The image shows a shoe."
    assert parsed.answer == "A red running shoe."


def test_qwen3vl_malformed_tool_call_json():
    raw = "<tool_call>{not valid json}</tool_call><|im_end|>"
    parsed = parse_agentic(raw, _processor_for("qwen3vl"))

    assert parsed.finish_reason == "parse_error"
    assert parsed.raw == raw
    assert parsed.tool_calls == []
    assert parsed.parse_errors


# ---------------------------------------------------------------------------
# gemma4
# ---------------------------------------------------------------------------


def test_gemma4_tool_call():
    raw = (
        "<|channel>thought\nLooks like footwear.\n<channel|>"
        '<|tool_call>call:submit_product_classification{category:<|"|>shoes<|"|>,price:29.99}<tool_call|><|endoftext|>'
    )
    parsed = parse_agentic(raw, _processor_for("gemma4"))

    assert parsed.finish_reason == "tool_call"
    assert parsed.thinking == "Looks like footwear."
    assert parsed.tool_calls == [
        {"name": "submit_product_classification", "arguments": {"category": "shoes", "price": 29.99}}
    ]


def test_gemma4_answer_only():
    raw = "<|channel>thought\nSimple case.\n<channel|>A red running shoe.<turn|>"
    parsed = parse_agentic(raw, _processor_for("gemma4"))

    assert parsed.finish_reason == "stop"
    assert parsed.thinking == "Simple case."
    assert parsed.answer == "A red running shoe."


# ---------------------------------------------------------------------------
# schema selection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model_identifier", "expected_key"),
    [
        ("Qwen/Qwen3-VL-2B-Thinking", "qwen3vl"),
        ("Qwen/Qwen3-VL-2B-Instruct", "generic"),
        ("LiquidAI/LFM2.5-VL-450M", "lfm2.5"),
        ("LiquidAI/LFM2-VL-1.6B", "generic"),
        ("google/gemma-4-E2B-it", "gemma4"),
        ("google/gemma-3-4b-it", "generic"),
        ("qwen3vl", "qwen3vl"),
        ("lfm2.5", "lfm2.5"),
        ("gemma4", "gemma4"),
    ],
)
def test_response_schema_key(model_identifier, expected_key):
    assert _response_schema_key(model_identifier) == expected_key


def test_response_schema_for_unknown_model_raises():
    with pytest.raises(ValueError):
        response_schema_for_model("some/unknown-model")


def test_ensure_respects_native_schema():
    processor = _processor_for("qwen3vl")
    # Second call must be a no-op: a schema is already attached.
    assert not ensure_model_response_schema(processor.tokenizer, model_family="qwen3vl")


def test_parse_agentic_without_schema_is_actionable_error():
    parsed = parse_agentic("anything", SimpleNamespace(tokenizer=_bare_tokenizer()))

    assert parsed.finish_reason == "parse_error"
    assert "response_schema" in parsed.parse_errors[0]


# ---------------------------------------------------------------------------
# simulated tool results
# ---------------------------------------------------------------------------


def test_append_simulated_tool_results_builds_tool_turn():
    messages = [{"role": "user", "content": [{"type": "text", "text": "classify"}]}]
    tool_calls = [{"name": "submit_product_classification", "arguments": {"category": "shoes"}}]
    simulated = {"submit_product_classification": {"status": "accepted"}}

    tool_images = append_simulated_tool_results(messages, tool_calls, simulated)

    assert tool_images == []
    assert len(messages) == 3
    assistant = messages[1]
    assert assistant["role"] == "assistant"
    assert assistant["tool_calls"] == [
        {"type": "function", "function": {"name": "submit_product_classification", "arguments": {"category": "shoes"}}}
    ]
    tool = messages[2]
    assert tool["role"] == "tool"
    assert tool["name"] == "submit_product_classification"
    assert tool["content"] == [{"type": "text", "text": '{"status": "accepted"}'}]
    # the caller's constant must not be mutated
    assert simulated == {"submit_product_classification": {"status": "accepted"}}


def test_append_simulated_tool_results_unknown_tool_gets_default():
    messages = []
    append_simulated_tool_results(messages, [{"name": "unknown_tool", "arguments": {}}], {})

    assert messages[1]["content"] == [{"type": "text", "text": '{"status": "ok"}'}]


def test_append_simulated_tool_results_attaches_images():
    photo = Path(__file__).resolve().parent.parent / "media" / "photo.jpg"
    messages = []
    simulated = {"cut_object": {"status": "done", "images": [str(photo)]}}

    tool_images = append_simulated_tool_results(messages, [{"name": "cut_object", "arguments": {}}], simulated)

    assert len(tool_images) == 1
    content = messages[1]["content"]
    assert content[0]["type"] == "image"
    assert content[0]["image"] is tool_images[0]
    assert content[1] == {"type": "text", "text": '{"status": "done"}'}
    # "images" is stripped from the serialized result but kept in the constant
    assert simulated["cut_object"]["images"] == [str(photo)]


# ---------------------------------------------------------------------------
# model auto-download
# ---------------------------------------------------------------------------


def test_ensure_model_missing_download_directory_is_actionable(monkeypatch):
    monkeypatch.delenv("MODEL_DOWNLOAD_DIRECTORY", raising=False)

    with pytest.raises(RuntimeError) as exc_info:
        ensure_model("MODEL_URL_HF_TEST_MODEL", "test/Test-Model")

    message = str(exc_info.value)
    assert "MODEL_DOWNLOAD_DIRECTORY" in message


def test_ensure_model_missing_url_env_var_is_actionable(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_DOWNLOAD_DIRECTORY", str(tmp_path))
    monkeypatch.delenv("MODEL_URL_HF_TEST_MODEL", raising=False)

    with pytest.raises(RuntimeError) as exc_info:
        ensure_model("MODEL_URL_HF_TEST_MODEL", "test/Test-Model")

    message = str(exc_info.value)
    assert "MODEL_URL_HF_TEST_MODEL" in message
    assert "https://app.ximilar.com/platform/vlm/tasks/" in message


def test_ensure_model_uses_per_model_subfolder(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_DOWNLOAD_DIRECTORY", str(tmp_path))
    model_dir = tmp_path / "Test-Model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")

    resolved = ensure_model("MODEL_URL_HF_TEST_MODEL", "test/Test-Model")

    assert resolved == str(model_dir)  # cached — no URL env var needed
