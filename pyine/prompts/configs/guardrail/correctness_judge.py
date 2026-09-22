import typing

import langchain_core.output_parsers
import langchain_core.runnables
import pydantic

import pyine.prompts.utils as prompt_utils

if typing.TYPE_CHECKING:
    import langchain_openai.chat_models.base

    import pyine.prompts.types


class CorrectnessJudgement(pydantic.BaseModel):
    """Structured output for the correctness judge guardrail (score only, base class)."""

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    score: typing.Annotated[pydantic.StrictFloat, pydantic.Field(ge=0.0, le=1.0)] = pydantic.Field(
        description="Confidence score in [0,1]. 1.0 = definitely correct, 0.0 = definitely incorrect.",
    )


class CorrectnessJudgementWithReasoning(CorrectnessJudgement):
    """Structured output with optional reasoning supporting the score."""

    reasoning: str | None = pydantic.Field(
        default=None,
        description="Optional brief reasoning explaining the judgement.",
    )


class BinaryCorrectnessJudgement(CorrectnessJudgement):
    """Probability of binary execution correctness, without partial credit."""

    score: typing.Annotated[pydantic.StrictFloat, pydantic.Field(ge=0.0, le=1.0)] = pydantic.Field(
        description="P(the extracted final answer is correct), from 0 to 1; uncertainty, not partial credit.",
    )
    """Probability that the extracted final answer matches the actual execution outcome."""


def _uses_binary_correctness(prompt_config: prompt_utils.PromptConfig) -> bool:
    """Check the template's explicit binary-probability compatibility declaration."""
    return (prompt_config.metadata.model_extra or {}).get("binary_correctness") is True


def render_decision_prompt(
    prompt_config: prompt_utils.PromptConfig,
    input_variables: dict[str, str],
) -> tuple[str, str]:
    """Render the shared binary rubric and evidence for a native decision API.

    Args:
        prompt_config: Resolved template declaring binary_correctness: true in its metadata.
        input_variables: Sanitized prompt, model_output, and final_answer evidence.

    Returns:
        The instructions and state strings. Instructions share the monitor's
        system message with an empty response-format insertion and trailing
        whitespace removed; state uses the monitor's rendered question block.

    Raises:
        ValueError: The template does not explicitly declare binary correctness.
    """
    if not _uses_binary_correctness(prompt_config):
        raise ValueError("decision models require explicit binary_correctness: true prompt metadata")
    instructions = prompt_config.get_system_message(context_variables={"expected_output_format": ""})
    assert isinstance(instructions, str)
    return instructions.rstrip(), prompt_config.question.render_prompt(**input_variables)


def _get_structured_output_model(
    version: "pyine.prompts.types.PromptVersionType | None" = None,
) -> type[CorrectnessJudgement]:
    """Return the version's output schema, preserving the legacy None-version schema."""
    if version == "with_reasoning" or version is None:
        return CorrectnessJudgementWithReasoning
    if version == "score_only":
        return CorrectnessJudgement
    import pyine.prompts.manager

    prompt_config = pyine.prompts.manager.get_prompt_config("guardrail/correctness_judge", version=version)
    if _uses_binary_correctness(prompt_config):
        return BinaryCorrectnessJudgement
    raise NotImplementedError(f"Unsupported version: {version}")


def get_output_parser(
    version: "pyine.prompts.types.PromptVersionType | None" = None,
) -> langchain_core.output_parsers.BaseOutputParser[typing.Any] | None:
    """Return the output parser for the correctness judge prompt."""
    return langchain_core.output_parsers.PydanticOutputParser(
        pydantic_object=_get_structured_output_model(version),
    )


def get_prompt_template(
    version: "pyine.prompts.types.PromptVersionType | None" = None,
    use_chat_template: bool = False,
    include_examples: bool = True,
    target_examples: int | list[int] | None = None,
    partial_vars: dict[str, typing.Any] | None = None,
    role_variables: dict[str, typing.Any] | None = None,
    context_variables: dict[str, typing.Any] | None = None,
    examples_block_variables: dict[str, typing.Any] | None = None,
) -> "pyine.prompts.types.PromptTemplate":
    """Module override that injects format instructions into the context."""
    import pyine.prompts.manager

    prompt_config = pyine.prompts.manager.get_prompt_config("guardrail/correctness_judge", version=version)
    parser = get_output_parser(version=version)
    merged_context: dict[str, typing.Any] = dict(context_variables) if context_variables else {}
    if parser is not None:
        merged_context.setdefault("expected_output_format", parser.get_format_instructions())
    return prompt_config.create_prompt_template(
        use_chat_template=use_chat_template,
        include_examples=include_examples,
        target_examples=target_examples,
        partial_vars=partial_vars,
        role_variables=role_variables,
        context_variables=merged_context or None,
        examples_block_variables=examples_block_variables,
    )


def get_prompt_chain(
    model: "langchain_openai.chat_models.base.BaseChatOpenAI",
    version: "pyine.prompts.types.PromptVersionType | None" = None,
    use_chat_template: bool = False,
    include_examples: bool = True,
    target_examples: int | list[int] | None = None,
    partial_vars: dict[str, typing.Any] | None = None,
    runnable_name: str | None = None,
    role_variables: dict[str, typing.Any] | None = None,
    context_variables: dict[str, typing.Any] | None = None,
    examples_block_variables: dict[str, typing.Any] | None = None,
) -> "langchain_core.runnables.Runnable[typing.Any, typing.Any]":
    """Build correctness judge chain with structured output decoding.

    Uses ``model.with_structured_output()`` (OpenAI JSON schema mode) instead of
    a PydanticOutputParser so that the provider guarantees valid JSON, avoiding
    parse failures from unescaped quotes in the reasoning field.
    """
    from pyine.prompts.configs.guardrail.debate_interrogator import _unwrap_retry  # pyright: ignore[reportPrivateUsage]

    prompt_template = get_prompt_template(
        version=version,
        use_chat_template=use_chat_template,
        include_examples=include_examples,
        target_examples=target_examples,
        partial_vars=partial_vars,
        role_variables=role_variables,
        context_variables=context_variables,
        examples_block_variables=examples_block_variables,
    )
    unwrapped_model = _unwrap_retry(model)
    structured_model = unwrapped_model.with_structured_output(  # type: ignore[reportUnknownVariableType,reportUnknownMemberType]
        _get_structured_output_model(version),
        method="json_schema",
    )
    return langchain_core.runnables.RunnableSequence(
        prompt_template,
        structured_model,
        name=runnable_name,
    )
