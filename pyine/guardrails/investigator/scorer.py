"""ReconsiderGuardrailScorer -- single-turn, fixed-question investigator baseline."""

from __future__ import annotations

import concurrent.futures
import dataclasses
import logging
import threading
import typing

import pyine.evals.correctness.types as correctness_types
import pyine.prompts.manager
import pyine.utils.langchain
import pyine.utils.parsing
from pyine.guardrails.investigator.types import ReconsiderTranscript, ReconsiderVerdict

if typing.TYPE_CHECKING:
    from pyine.guardrails.investigator.configs import ReconsiderGuardrailConfig

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class _Outcome:
    """Per-record scoring outcome carried back from the worker pool."""

    score: float
    cost: float
    transcript: ReconsiderTranscript | None


class ReconsiderGuardrailScorer:
    """GuardrailScorer that asks one fixed follow-up question, then scores the original answer.

    The follow-up is a constant template rather than a model-chosen probe, so the investigator
    never spends a call deciding what to ask. This is the control arm against which
    question-selection policies are measured.
    """

    def __init__(
        self,
        config: ReconsiderGuardrailConfig,
    ) -> None:
        """Build the responder and investigator chains from the provider configs."""
        self._config = config
        self._question_chain = pyine.prompts.manager.get_prompt_chain(
            model=config.responder_provider.get_model(),
            prompt_name=config.question_prompt_name,
            use_chat_template=config.use_chat_template,
            runnable_name="investigator_reconsider",
        )
        self._verdict_chain = pyine.prompts.manager.get_prompt_chain(
            model=config.investigator_provider.get_model(),
            prompt_name=config.verdict_prompt_name,
            version=config.verdict_prompt_version,
            use_chat_template=config.use_chat_template,
            runnable_name="investigator_verdict",
        )
        self._lock = threading.Lock()
        self._error_count = 0
        self._missing_answer_count = 0
        self._unparsed_revision_count = 0
        self._total_scored = 0

    def score_records(
        self,
        records: list[correctness_types.EvalRecord],
    ) -> correctness_types.ScoringResult:
        """Score every record with one responder call and one investigator call each."""
        outcomes: list[_Outcome | None] = [None] * len(records)
        with concurrent.futures.ThreadPoolExecutor(max_workers=self._config.max_workers) as executor:
            future_to_idx = {executor.submit(self._score_single, record): idx for idx, record in enumerate(records)}
            for future in concurrent.futures.as_completed(future_to_idx):
                outcomes[future_to_idx[future]] = future.result()
        scores: list[float] = []
        costs: list[float] = []
        attempt_metadata: dict[correctness_types.ScoredAttemptKey, dict[str, typing.Any]] = {}
        for idx, outcome in enumerate(outcomes):
            assert outcome is not None, "worker pool must populate every outcome"
            scores.append(outcome.score)
            costs.append(outcome.cost)
            if outcome.transcript is not None:
                key = (records[idx].sample_id, records[idx].attempt_index, idx)
                attempt_metadata[key] = outcome.transcript.model_dump()
        self._total_scored += len(records)
        return correctness_types.ScoringResult(
            scores=scores,
            verification_costs=costs,
            attempt_metadata=attempt_metadata,
        )

    def _score_single(
        self,
        record: correctness_types.EvalRecord,
    ) -> _Outcome:
        """Run the fixed follow-up and verdict for one record."""
        if record.final_answer is None:
            with self._lock:
                self._missing_answer_count += 1
            return _Outcome(score=self._config.default_score_on_missing_answer, cost=0.0, transcript=None)
        original_prompt = str(record.record.get("prompt", ""))
        responder_handler = pyine.utils.langchain.CaptureLLMHandler()
        investigator_handler = pyine.utils.langchain.CaptureLLMHandler()
        try:
            revised_output = self._question_chain.invoke(
                {
                    "original_prompt": original_prompt,
                    "original_output": record.model_output,
                    "original_answer": record.final_answer,
                    "question": self._config.fixed_question,
                },
                config={"callbacks": [responder_handler]},
            )
            revised_answer = self._extract_answer(str(revised_output))
            answer_changed = None if revised_answer is None else revised_answer.strip() != record.final_answer.strip()
            verdict_raw: typing.Any = self._verdict_chain.invoke(
                {
                    "original_prompt": original_prompt,
                    "original_output": record.model_output,
                    "original_answer": record.final_answer,
                    "revised_output": str(revised_output),
                    "revised_answer": "<unparsed>" if revised_answer is None else revised_answer,
                    "answer_changed": "unknown" if answer_changed is None else str(answer_changed).lower(),
                },
                config={"callbacks": [investigator_handler]},
            )
        except Exception:
            logger.exception("investigator call failed for record %s, using default score", record.sample_id)
            with self._lock:
                self._error_count += 1
            return _Outcome(score=self._config.default_score_on_error, cost=0.0, transcript=None)
        verdict = ReconsiderVerdict.model_validate(
            verdict_raw if isinstance(verdict_raw, dict) else verdict_raw.model_dump()
        )
        transcript = ReconsiderTranscript(
            question=self._config.fixed_question,
            revised_output=str(revised_output),
            revised_answer=revised_answer,
            answer_changed=answer_changed,
            verdict=verdict,
            responder_token_count=pyine.utils.langchain.extract_token_count_from_handler(responder_handler),
            investigator_token_count=pyine.utils.langchain.extract_token_count_from_handler(investigator_handler),
        )
        return _Outcome(score=verdict.score, cost=transcript.total_token_count, transcript=transcript)

    def _extract_answer(
        self,
        revised_output: str,
    ) -> str | None:
        """Pull the tagged final answer out of a free-text reply, or None when absent."""
        result = pyine.utils.parsing.extract_tag_blocks(revised_output, self._config.answer_tag)
        if result.block_count != 1:
            with self._lock:
                self._unparsed_revision_count += 1
            return None
        return result.blocks[0].strip()

    def get_metadata(self) -> dict[str, typing.Any]:
        """Return scorer provenance for the evaluation record."""
        return {
            "scorer_type": "reconsider_investigator",
            "responder_provider": self._config.responder_provider.provider,
            "responder_model_kwargs": self._config.responder_provider.model_kwargs,
            "investigator_provider": self._config.investigator_provider.provider,
            "investigator_model_kwargs": self._config.investigator_provider.model_kwargs,
            "question_prompt_name": self._config.question_prompt_name,
            "verdict_prompt_name": self._config.verdict_prompt_name,
            "answer_tag": self._config.answer_tag,
            "max_workers": self._config.max_workers,
            "total_scored": self._total_scored,
            "error_count": self._error_count,
            "missing_answer_count": self._missing_answer_count,
            "unparsed_revision_count": self._unparsed_revision_count,
        }

    def get_verification_cost_unit(self) -> str | None:
        """Report costs in generated tokens, matching the judge and debate scorers."""
        return "tokens"
