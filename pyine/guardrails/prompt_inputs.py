"""Shared monitor/decision-model inputs; debate retains its stricter role policy."""

import hashlib
import json
import typing

import pyine.evals.correctness.types as correctness_types
import pyine.prompts.utils


def sanitize_for_json_encoding(text: str) -> str:
    """Replace lone surrogates using the legacy monitor's UTF-8 round trip.

    Args:
        text: Prompt text that may contain invalid Unicode surrogates.

    Returns:
        UTF-8-encodable text. A lone surrogate can expand into multiple replacement
        characters, so the original string length is not necessarily preserved.
    """
    return text.encode("utf-8", errors="surrogatepass").decode("utf-8", errors="replace")


def format_prompt_messages(prompt_messages: list[dict[str, typing.Any]]) -> str:
    """Flatten message content, preserving the monitor's role-free rendering.

    Unlike the debate renderer, assistant-only messages are accepted. Non-string
    text values are stringified; missing or null text uses the part's representation.

    Args:
        prompt_messages: Nonempty chat-message list with string or content-part payloads.

    Returns:
        Message contents joined with blank lines, omitting role names. Content
        parts within a message are joined with single newlines.

    Raises:
        AssertionError: The message list is empty.
    """
    assert len(prompt_messages) > 0, "prompt_messages must not be empty"
    parts: list[str] = []
    for message in prompt_messages:
        content = message.get("content", "")
        if isinstance(content, list):
            rendered_parts: list[str] = []
            for part in typing.cast("list[typing.Any]", content):
                if isinstance(part, dict):
                    part_dict = typing.cast("dict[str, typing.Any]", part)
                    text_value: typing.Any = part_dict.get("text")
                    rendered_parts.append(str(text_value) if text_value is not None else repr(part_dict))
                else:
                    rendered_parts.append(str(part))
            content = "\n".join(rendered_parts)
        elif not isinstance(content, str):
            content = str(content)
        parts.append(content)
    return "\n\n".join(parts)


def get_original_prompt(record: correctness_types.EvalRecord) -> str:
    """Select the original prompt using the monitor's prompt-first precedence.

    Args:
        record: Evaluation record containing the predictor's original input.

    Returns:
        The stored prompt if it is not None; otherwise flattened prompt_messages.

    Raises:
        AssertionError: Both prompt and prompt_messages are missing or None, or
            the fallback message list is empty.
    """
    prompt = record.record.get("prompt")
    if prompt is None:
        messages = record.record.get("prompt_messages")
        assert messages is not None, "EvalRecord must contain either 'prompt' or 'prompt_messages'"
        return format_prompt_messages(messages)
    return prompt


def get_judge_input_variables(record: correctness_types.EvalRecord) -> dict[str, str]:
    """Select and sanitize evidence without exposing labels or references.

    Args:
        record: Evaluation record with an extracted final answer.

    Returns:
        Only the prompt, model_output, and final_answer template variables, with
        lone Unicode surrogates replaced for JSON encoding.

    Raises:
        ValueError: The record has no extracted final answer.
    """
    if record.final_answer is None:
        raise ValueError("cannot render a correctness judgment without a final answer")
    return {
        "prompt": sanitize_for_json_encoding(get_original_prompt(record)),
        "model_output": sanitize_for_json_encoding(record.model_output),
        "final_answer": sanitize_for_json_encoding(record.final_answer),
    }


def get_prompt_provenance(
    prompt_config: pyine.prompts.utils.PromptConfig,
    requested_version: str | None,
) -> dict[str, typing.Any]:
    """Freeze the resolved template and its content hash for later reconstruction.

    Args:
        prompt_config: Fully resolved prompt configuration used by the scorer.
        requested_version: Explicit version requested by the caller, or None for the default.

    Returns:
        JSON-serializable template contents, requested and resolved versions,
        SHA-256 of the serialized template, and the shared input-format version.
    """
    template = prompt_config.model_dump(mode="json")
    serialized = json.dumps(template, sort_keys=True, ensure_ascii=True)
    return {
        "prompt_name": prompt_config.metadata.name,
        "prompt_version": requested_version,
        "resolved_prompt_version": prompt_config.metadata.version,
        "prompt_template": template,
        "prompt_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
        "input_format_version": "monitor_v2",
    }
