"""Prompt config module for the investigator verdict over one fixed follow-up turn."""

import typing

import langchain_core.output_parsers
import langchain_core.runnables
import pydantic

if typing.TYPE_CHECKING:
    import langchain_openai.chat_models.base

    import pyine.prompts.types


class ReconsiderVerdictOutput(pydantic.BaseModel):
    """Structured verdict about the predictor's original answer."""

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    score: typing.Annotated[pydantic.StrictFloat, pydantic.Field(ge=0.0, le=1.0)] = pydantic.Field(
        description="Confidence the ORIGINAL answer was correct. 1.0 = definitely correct, 0.0 = definitely incorrect.",
    )
    reasoning: str | None = pydantic.Field(
        default=None,
        description="Optional brief reasoning explaining the verdict.",
    )


def get_output_parser(
    version: "pyine.prompts.types.PromptVersionType | None" = None,
) -> langchain_core.output_parsers.BaseOutputParser[typing.Any] | None:
    """Return the output parser for the investigator verdict prompt."""
    if version not in ("with_reasoning", "with_signal_hints", None):
        raise NotImplementedError(f"Unsupported version: {version}")
    return langchain_core.output_parsers.PydanticOutputParser(pydantic_object=ReconsiderVerdictOutput)


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
    """Module override that injects the structured-output format instructions."""
    import pyine.prompts.manager

    prompt_config = pyine.prompts.manager.get_prompt_config("guardrail/investigator_verdict", version=version)
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
    """Build the verdict chain with decode-time JSON schema enforcement."""
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
    structured_model = _unwrap_retry(model).with_structured_output(  # type: ignore[reportUnknownVariableType,reportUnknownMemberType]
        ReconsiderVerdictOutput,
        method="json_schema",
    )
    return langchain_core.runnables.RunnableSequence(prompt_template, structured_model, name=runnable_name)
