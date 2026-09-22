"""PromptedLLMGuardrailScorer - GuardrailScorer for prompted (non-fine-tuned) LLMs."""

from __future__ import annotations

import concurrent.futures
import dataclasses
import logging
import threading
import typing

import tqdm

import pyine.evals.correctness.types as correctness_types
import pyine.guardrails.prompt_inputs
import pyine.prompts.manager
import pyine.utils.langchain

if typing.TYPE_CHECKING:
    from pyine.guardrails.prompted_llm.configs import PromptedLLMGuardrailConfig

logger = logging.getLogger(__name__)


ScoringDecisionType = typing.Literal["llm_judge", "missing_answer", "llm_format_error", "llm_call_error"]


@dataclasses.dataclass(frozen=True)
class _ScoringOutcome:
    """Result of scoring a single record."""

    score: float
    token_count: float
    reasoning: str | None
    decision_type: ScoringDecisionType


_sanitize_for_json_encoding = pyine.guardrails.prompt_inputs.sanitize_for_json_encoding


class PromptedLLMGuardrailScorer:
    """GuardrailScorer adapter for a prompted (non-fine-tuned) LLM judge.

    Uses a LangChain chain (prompt | model | parser) to ask an LLM to judge whether a model's code
    execution prediction is correct. Returns continuous confidence scores (0-1) and tracks token
    costs per record.

    Concurrency is achieved via ThreadPoolExecutor with sync chain.invoke() calls, which is safe to
    call from within an already-running asyncio event loop (unlike asyncio.run(), which would crash).
    """

    def __init__(self, config: PromptedLLMGuardrailConfig) -> None:
        """Initializes the scorer (constructing the LLM prompting chain)."""
        self._config = config
        self._llm = config.llm_provider.get_model()
        self._chain = pyine.prompts.manager.get_prompt_chain(
            model=self._llm,
            prompt_name=config.prompt_name,
            version=config.prompt_version,
            use_chat_template=config.use_chat_template,
            runnable_name="correctness_judge",
        )
        self._prompt_provenance = pyine.guardrails.prompt_inputs.get_prompt_provenance(
            pyine.prompts.manager.get_prompt_config(config.prompt_name, version=config.prompt_version),
            config.prompt_version,
        )
        self._prompt_provenance["response_prompt_template"] = pyine.prompts.manager.get_prompt_template(
            prompt_name=config.prompt_name,
            version=config.prompt_version,
            use_chat_template=config.use_chat_template,
        ).pretty_repr()
        self._error_count: int = 0
        self._skipped_no_final_answer: int = 0
        self._error_lock = threading.Lock()
        self._total_scored: int = 0

    def score_records(
        self,
        records: list[correctness_types.EvalRecord],
    ) -> correctness_types.ScoringResult:
        """Score records by asking an LLM to judge correctness.

        Uses ThreadPoolExecutor for concurrent I/O-bound LLM API calls. This is safe to call from
        within a running asyncio event loop (the eval pipeline's evaluate_wrapped_model is async).
        """
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self._config.max_workers,
        ) as executor:
            future_to_idx = {executor.submit(self._score_single, record): idx for idx, record in enumerate(records)}
            outcomes: list[_ScoringOutcome | None] = [None] * len(records)
            progress = tqdm.tqdm(
                total=len(records),
                desc=f"scoring ({self._config.llm_provider.provider})",
                unit="rec",
            )
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                outcomes[idx] = future.result()
                progress.update(1)
            progress.close()

        scores: list[float] = []
        costs: list[float] = []
        attempt_metadata: dict[correctness_types.ScoredAttemptKey, dict[str, typing.Any]] = {}
        assert len(outcomes) == len(records), "expected as many outcomes as records"
        for res_idx, outcome in enumerate(outcomes):
            assert outcome is not None, "all futures should have completed successfully"
            scores.append(outcome.score)
            costs.append(outcome.token_count)
            scored_attempt_key = (records[res_idx].sample_id, records[res_idx].attempt_index, res_idx)
            attempt_metadata[scored_attempt_key] = {
                "decision_type": outcome.decision_type,
                "reasoning": outcome.reasoning or "",
            }

        self._total_scored += len(records)
        return correctness_types.ScoringResult(
            scores=scores,
            verification_costs=costs,
            attempt_metadata=attempt_metadata,
        )

    def _score_single(
        self,
        record: correctness_types.EvalRecord,
    ) -> _ScoringOutcome:
        """Score a single record via sync chain.invoke().

        Called from worker threads. Error counting is protected by a threading.Lock.
        """
        if record.final_answer is None:
            logger.debug(
                f"record {record.sample_id} has no final_answer; "
                f"skipping, using default score ({self._config.default_score_on_missing_answer})"
            )
            with self._error_lock:
                self._skipped_no_final_answer += 1
            return _ScoringOutcome(
                score=self._config.default_score_on_missing_answer,
                token_count=0.0,
                reasoning=None,
                decision_type="missing_answer",
            )
        input_vars = pyine.guardrails.prompt_inputs.get_judge_input_variables(record)

        handler = pyine.utils.langchain.CaptureLLMHandler()
        reasoning: str | None = None
        decision_type: ScoringDecisionType = "llm_judge"

        try:
            result: typing.Any = self._chain.invoke(
                input_vars,
                config={"callbacks": [handler]},
            )
            # extract score from structured output (Pydantic model or dict)
            if hasattr(result, "score"):
                score = float(result.score)
            elif isinstance(result, dict) and "score" in result:
                result_dict = typing.cast("dict[str, typing.Any]", result)
                score = float(result_dict["score"])
            else:
                result_type_name = type(typing.cast("typing.Any", result)).__name__
                logger.warning(f"LLM returned unexpected format for {record.sample_id}: {result_type_name}")
                score = self._config.default_score_on_error
                decision_type = "llm_format_error"
                with self._error_lock:
                    self._error_count += 1
            score = max(0.0, min(1.0, score))

            # try to extract the reasoning from the structured output (there might not be any)
            if hasattr(result, "reasoning") and result.reasoning is not None:  # type: ignore
                reasoning = typing.cast("str", result.reasoning)  # type: ignore
            elif isinstance(result, dict) and "reasoning" in result and result["reasoning"] is not None:
                reasoning = typing.cast("str", result["reasoning"])
            if reasoning is not None and not isinstance(reasoning, str):  # type: ignore
                reasoning_type_name = type(reasoning).__name__
                logger.warning(f"LLM returned unexpected type for reasoning: {reasoning_type_name}")
                reasoning = None

        except Exception:
            logger.warning(
                "LLM call failed for record %s, using default score",
                record.sample_id,
                exc_info=True,
            )
            score = self._config.default_score_on_error
            decision_type = "llm_call_error"
            with self._error_lock:
                self._error_count += 1

        token_count = self._extract_token_count(handler)
        return _ScoringOutcome(
            score=score,
            token_count=token_count,
            reasoning=reasoning,
            decision_type=decision_type,
        )

    _format_prompt_messages = staticmethod(pyine.guardrails.prompt_inputs.format_prompt_messages)

    @staticmethod
    def _extract_token_count(
        handler: pyine.utils.langchain.CaptureLLMHandler,
    ) -> float:
        """Extract total token count from the capture handler."""
        return pyine.utils.langchain.extract_token_count_from_handler(handler, field="total_tokens")

    def get_metadata(self) -> dict[str, typing.Any]:
        """Return guardrail metadata for reporting."""
        return {
            "scorer_type": "prompted_llm",
            **self._prompt_provenance,
            "provider": self._config.llm_provider.provider,
            "model_kwargs": self._config.llm_provider.model_kwargs,
            "max_workers": self._config.max_workers,
            "default_score_on_error": self._config.default_score_on_error,
            "default_score_on_missing_answer": self._config.default_score_on_missing_answer,
            "total_scored": self._total_scored,
            "error_count": self._error_count,
            "skipped_no_final_answer": self._skipped_no_final_answer,
        }

    def get_verification_cost_unit(self) -> str | None:
        """Return 'tokens' as the verification cost unit."""
        return "tokens"
