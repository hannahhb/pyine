import pathlib
import typing
import unittest.mock

import pytest

import pyine.guardrails.decision_model.configs as decision_configs
import pyine.guardrails.decision_model.scorer as decision_scorer
import pyine.guardrails.prompt_inputs as inputs
import pyine.guardrails.prompted_llm.configs as monitor_configs
import pyine.guardrails.prompted_llm.scorer as monitor_scorer
import pyine.prompts.configs.guardrail.correctness_judge as judge
import pyine.prompts.manager
import pyine.prompts.utils
import pyine.utils.llm_providers
import tests.guardrails.decision_model.helpers as helpers


class TestSharedBinaryPrompt:
    @pytest.mark.parametrize(
        "version, fixture_name",
        [
            (None, "legacy_score_only.txt"),
            ("with_reasoning", "legacy_with_reasoning.txt"),
            ("score_only", "legacy_score_only.txt"),
        ],
    )
    def test_legacy_rendering_snapshot(
        self,
        version: str | None,
        fixture_name: str,
    ) -> None:
        """Compare legacy text with pre-change fixtures, normalizing dependency-owned format text."""
        rendered = judge.get_prompt_template(version=version, use_chat_template=True).format(
            prompt="print(2 + 3)",
            model_output="The answer is 5.",
            final_answer="5",
        )
        parser = judge.get_output_parser(version=version)
        normalized = rendered.replace(parser.get_format_instructions(), "<OUTPUT_FORMAT_INSTRUCTIONS>")
        expected = (pathlib.Path(__file__).parent / "fixtures" / fixture_name).read_text().removesuffix("\n")
        assert normalized == expected
        config = pyine.prompts.manager.get_prompt_config("guardrail/correctness_judge", version=version)
        assert config.metadata.version == (version or "score_only")
        assert parser.pydantic_object == (
            judge.CorrectnessJudgement if version == "score_only" else judge.CorrectnessJudgementWithReasoning
        )

    def test_identical_evidence_and_task_except_response_protocol(self) -> None:
        """Keep native decisions and LLM monitors on the same binary rubric and evidence."""
        variables = inputs.get_judge_input_variables(helpers.make_record())
        prompt_config = pyine.prompts.manager.get_prompt_config(
            "guardrail/correctness_judge",
            version="binary_correctness_v1",
        )
        instructions, state = judge.render_decision_prompt(prompt_config, variables)
        chat = judge.get_prompt_template(version="binary_correctness_v1", use_chat_template=True)
        messages = chat.invoke(variables).to_messages()
        format_instructions = judge.get_output_parser(version="binary_correctness_v1").get_format_instructions()
        assert messages[0].content.replace(format_instructions, "").rstrip() == instructions
        assert instructions == instructions.rstrip()
        assert messages[1].content == state
        assert "partial-credit" in instructions
        assert "JSON" not in instructions
        assert "reasoning" not in judge.BinaryCorrectnessJudgement.model_fields

    def test_rejects_legacy_decision_prompt(self) -> None:
        """Reject legacy rubrics that mix partial credit with correctness uncertainty."""
        legacy = pyine.prompts.manager.get_prompt_config("guardrail/correctness_judge", version="score_only")
        with pytest.raises(ValueError, match="explicit binary"):
            judge.render_decision_prompt(legacy, inputs.get_judge_input_variables(helpers.make_record()))

    def test_future_binary_version_uses_metadata(
        self,
        config: decision_configs.DecisionModelGuardrailConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Accept a new version through metadata for both the native scorer and monitor parser."""
        original = pyine.prompts.manager.get_prompt_config(
            "guardrail/correctness_judge", version="binary_correctness_v1"
        )
        future = original.model_copy(
            update={"metadata": original.metadata.model_copy(update={"version": "binary_correctness_v2"})}
        )

        def get_future_prompt(
            prompt_name: str,
            version: str | None = None,
        ) -> pyine.prompts.utils.PromptConfig:
            """Resolve the synthetic future version through the usual prompt-manager interface."""
            assert prompt_name == "guardrail/correctness_judge"
            assert version == "binary_correctness_v2"
            return future

        monkeypatch.setattr(pyine.prompts.manager, "get_prompt_config", get_future_prompt)
        future_config = decision_configs.DecisionModelGuardrailConfig(
            **{**config.model_dump(), "prompt_version": "binary_correctness_v2"}
        )
        scorer = decision_scorer.DecisionModelGuardrailScorer(future_config, helpers.FakeClient())
        try:
            assert scorer.score_records([helpers.make_record()]).scores == [0.99]
            assert scorer.get_metadata()["resolved_prompt_version"] == "binary_correctness_v2"
        finally:
            scorer.close()
        assert (
            judge.get_output_parser(version="binary_correctness_v2").pydantic_object is judge.BinaryCorrectnessJudgement
        )

    @pytest.mark.parametrize("declaration", [None, False, "true"])
    def test_binary_version_name_does_not_replace_metadata(
        self,
        declaration: typing.Any,
    ) -> None:
        """Reject templates without an actual true compatibility declaration, regardless of name."""
        original = pyine.prompts.manager.get_prompt_config(
            "guardrail/correctness_judge", version="binary_correctness_v1"
        )
        unmarked = original.model_copy(
            update={"metadata": original.metadata.model_copy(update={"binary_correctness": declaration})}
        )
        with pytest.raises(ValueError, match="explicit binary"):
            judge.render_decision_prompt(unmarked, inputs.get_judge_input_variables(helpers.make_record()))


class TestPromptInputs:
    def test_prompt_fallback_precedence_unicode_and_roles(self) -> None:
        """Preserve prompt precedence, Unicode sanitization, and role-free fallback rendering."""
        record = helpers.make_record(
            record={
                "prompt": "primary\ud800",
                "prompt_messages": [{"role": "assistant", "content": "fallback"}],
            }
        )
        variables = inputs.get_judge_input_variables(record)
        assert variables["prompt"] == "primary\ufffd\ufffd\ufffd"
        assert "fallback" not in variables["prompt"]
        fallback = helpers.make_record(
            record={
                "prompt_messages": [
                    {"role": "system", "content": "setup"},
                    {"role": "assistant", "content": [{"text": "code"}, "extra"]},
                ],
            }
        )
        assert inputs.get_judge_input_variables(fallback)["prompt"] == "setup\n\ncode\nextra"

    @pytest.mark.parametrize(
        "part, expected",
        [({"text": None}, "{'text': None}"), ({"text": 42}, "42"), ({"image": "payload"}, "{'image': 'payload'}")],
    )
    def test_non_string_content_parts(
        self,
        part: dict[str, typing.Any],
        expected: str,
    ) -> None:
        """Render null and non-string message parts consistently with the debate renderer."""
        assert inputs.format_prompt_messages([{"content": [part]}]) == expected


class TestScorerInputParity:
    @pytest.fixture
    def monitor(self) -> monitor_scorer.PromptedLLMGuardrailScorer:
        """Build a monitor with a fake chain so input handling can be compared without network calls."""
        config = monitor_configs.PromptedLLMGuardrailConfig(
            llm_provider=pyine.utils.llm_providers.LLMProviderConfig(provider="openai"),
            prompt_version="binary_correctness_v1",
            max_workers=1,
        )
        chain = unittest.mock.Mock()
        chain.invoke.return_value = {"score": 0.99}
        with (
            unittest.mock.patch.object(pyine.utils.llm_providers.LLMProviderConfig, "get_model"),
            unittest.mock.patch.object(pyine.prompts.manager, "get_prompt_chain", return_value=chain),
        ):
            return monitor_scorer.PromptedLLMGuardrailScorer(config)

    def test_missing_answer_needs_no_prompt(
        self,
        config: decision_configs.DecisionModelGuardrailConfig,
        monitor: monitor_scorer.PromptedLLMGuardrailScorer,
    ) -> None:
        """Assign the missing-answer policy before rendering irrelevant evidence in both scorers."""
        client = helpers.FakeClient()
        scorer = decision_scorer.DecisionModelGuardrailScorer(config, client)
        record = helpers.make_record(final_answer=None, record={})
        try:
            assert scorer.score_records([record]).scores == monitor.score_records([record]).scores == [0.0]
            assert not client.calls
        finally:
            scorer.close()

    def test_monitor_renders_original_prompt_once(
        self,
        monitor: monitor_scorer.PromptedLLMGuardrailScorer,
    ) -> None:
        """Render each scorable monitor record once through the shared evidence helper."""
        record = helpers.make_record(record={"prompt_messages": [{"content": [{"text": None}]}]})
        with unittest.mock.patch.object(inputs, "get_original_prompt", wraps=inputs.get_original_prompt) as render:
            assert monitor.score_records([record]).scores == [0.99]
        render.assert_called_once_with(record)
