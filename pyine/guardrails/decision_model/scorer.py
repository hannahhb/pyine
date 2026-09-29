"""Ordered remote probabilities behind the existing GuardrailScorer interface."""

import concurrent.futures
import dataclasses
import hashlib
import json
import time
import typing

import langchain_core.rate_limiters

import pyine.evals.correctness.types as correctness_types
import pyine.guardrails.decision_model.configs as decision_configs
import pyine.guardrails.prompt_inputs
import pyine.prompts.configs.guardrail.correctness_judge as correctness_judge
import pyine.prompts.manager
import pyine.prompts.utils


@dataclasses.dataclass(frozen=True)
class _Request:
    """Rendered provider request and its identity for in-run prediction reuse."""

    instructions: str
    """Binary correctness question and judging rubric, without response-format instructions."""
    state: str
    """Rendered original prompt, model response, and extracted final answer."""
    fingerprint: str
    """SHA-256 of the provider, endpoint, model, instructions, and state."""


def _render_request(
    config: decision_configs.DecisionModelGuardrailConfig,
    prompt: pyine.prompts.utils.PromptConfig,
    record: correctness_types.EvalRecord,
) -> _Request:
    """Render permitted evidence and fingerprint the complete logical request."""
    instructions, state = correctness_judge.render_decision_prompt(
        prompt,
        pyine.guardrails.prompt_inputs.get_judge_input_variables(record),
    )
    payload = {
        "backend": config.provider.backend,
        "endpoint": str(config.provider.base_url),
        "model": config.provider.model,
        "state": state,
        "instructions": instructions,
    }
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode()).hexdigest()
    return _Request(instructions, state, fingerprint)


def preflight_records(
    config: decision_configs.DecisionModelGuardrailConfig,
    calibration_records: list[correctness_types.EvalRecord],
    evaluation_subsets: dict[str, list[correctness_types.EvalRecord]],
) -> dict[str, typing.Any]:
    """Validate all selected populations and size requests without opening a client.

    Args:
        config: Provider, prompt, and request-reuse settings for the planned run.
        calibration_records: Selected calibration draws, including duplicates.
        evaluation_subsets: Selected records keyed by guardrail_valid or guardrail_test.

    Returns:
        Per-population row counts, missing-answer counts, distinct request counts,
        and maximum request lengths in characters, plus the total distinct
        requests and expected logical API calls. SDK retries are excluded;
        exact context-token counts are unavailable before provider execution.

    Raises:
        ValueError: Evaluation subsets are empty or unsupported, calibration
            lacks either label, or a population is empty or has no final answers.
    """
    if not evaluation_subsets:
        raise ValueError("at least one evaluation subset is required")
    if set(evaluation_subsets) - {"guardrail_valid", "guardrail_test"}:
        raise ValueError("evaluation subsets must be guardrail_valid or guardrail_test")
    if {record.label for record in calibration_records} != {False, True}:
        raise ValueError("calibration requires both correct and incorrect labels")
    prompt = pyine.prompts.manager.get_prompt_config(config.prompt_name, version=config.prompt_version)
    populations = [("calibration", calibration_records), *evaluation_subsets.items()]
    stats: dict[str, typing.Any] = {}
    fingerprints: set[str] = set()
    expected_calls_without_reuse = 0
    for name, records in populations:
        if not records:
            raise ValueError(f"{name}: selected population is empty")
        missing = sum(record.final_answer is None for record in records)
        if missing == len(records):
            raise ValueError(f"{name}: every record lacks a final answer; check export parsing")
        requests = [_render_request(config, prompt, record) for record in records if record.final_answer is not None]
        unique = {request.fingerprint for request in requests}
        fingerprints.update(unique)
        multiplier = len(evaluation_subsets) if name == "calibration" else 1
        expected_calls_without_reuse += len(requests) * multiplier
        stats[name] = {
            "rows": len(records),
            "missing_answers": missing,
            "missing_answer_fraction": missing / len(records),
            "distinct_requests": len(unique),
            "max_request_characters": max(len(request.state) + len(request.instructions) for request in requests),
        }
    return {
        "populations": stats,
        "distinct_requests": len(fingerprints),
        "expected_api_calls": len(fingerprints) if config.cache_within_run else expected_calls_without_reuse,
        "exact_context_tokens_available": False,
    }


class DecisionModelGuardrailScorer:
    """Preserve native probabilities, ordered rows, and optional frozen-request reuse.

    Calls to score_records are sequential; individual provider calls run in worker
    threads. The scorer owns and closes its client, including an injected client.
    """

    def __init__(
        self,
        config: decision_configs.DecisionModelGuardrailConfig,
        client: decision_configs.DecisionModelClient | None = None,
    ) -> None:
        """Freeze the prompt and construct a client after app-level preflight.

        Args:
            config: Provider, prompt, concurrency, and scoring policies.
            client: Optional client supporting concurrent predictions. If None,
                construct the configured TypeSafe client. The scorer owns and
                closes either client; callers should invoke close after scoring.
        """
        self._config = config
        self._prompt = pyine.prompts.manager.get_prompt_config(config.prompt_name, version=config.prompt_version)
        self._provenance = pyine.guardrails.prompt_inputs.get_prompt_provenance(self._prompt, config.prompt_version)
        instructions, _ = correctness_judge.render_decision_prompt(
            self._prompt, {"prompt": "", "model_output": "", "final_answer": ""}
        )
        self._provenance.update({"instructions": instructions, "response_protocol": "native_noul_probability"})
        self._limiter = langchain_core.rate_limiters.InMemoryRateLimiter(
            requests_per_second=config.requests_per_second,
            max_bucket_size=1,
            check_every_n_seconds=min(0.1, 1.0 / config.requests_per_second),
        )
        self._cache: dict[str, tuple[decision_configs.DecisionPrediction, float]] = {}
        self._total_scored = 0
        self._missing_answers = 0
        self._cache_hits = 0
        self._completed_requests = 0
        self._input_tokens = 0
        self._output_tokens = 0
        self._requests_without_usage = 0
        self._request_seconds = 0.0
        self._closed = False
        if client is None:
            import pyine.guardrails.decision_model.typesafe as typesafe_adapter

            client = typesafe_adapter.TypeSafeSystemOneClient(config.provider)
        self._client = client

    def _predict(
        self,
        request: _Request,
    ) -> tuple[decision_configs.DecisionPrediction, float]:
        """Rate-limit and validate one prediction, returning its elapsed request seconds."""
        self._limiter.acquire(blocking=True)
        started = time.monotonic()
        prediction = self._client.predict(instructions=request.instructions, state=request.state)
        expected_model = self._config.provider.expected_model or self._config.provider.model
        if prediction.model != expected_model:
            raise ValueError(f"expected model {expected_model!r}, received {prediction.model!r}")
        if self._config.provider.require_token_usage and prediction.input_tokens is None:
            raise ValueError("the configured provider must report token usage")
        return prediction, time.monotonic() - started

    def score_records(
        self,
        records: list[correctness_types.EvalRecord],
    ) -> correctness_types.ScoringResult:
        """Score independent requests concurrently and restore every draw in order.

        Provider failures propagate after pending jobs are cancelled and running
        jobs finish. No fallback probability is assigned to failed requests.

        Args:
            records: Evaluation draws in their required output order, including duplicates.

        Returns:
            Native correctness probabilities, with policy-assigned zeros for
            missing answers and metadata keyed by sample, attempt, and draw index.
            When usage is required, verification costs are input plus output tokens
            per draw; reused predictions retain their original token counts.
            Otherwise verification costs are None. Metadata from get_metadata
            reports observed usage for completed API calls rather than per draw.

        Raises:
            RuntimeError: The scorer has already been closed.
            ValueError: A prediction reports a different model or lacks required usage.
        """
        if self._closed:
            raise RuntimeError("decision scorer is closed")
        requests = [
            _render_request(self._config, self._prompt, record) if record.final_answer is not None else None
            for record in records
        ]
        outcomes: dict[int, tuple[decision_configs.DecisionPrediction, float]] = {}
        reused: set[int] = set()
        grouped: dict[str, list[int]] = {}
        jobs: dict[int, _Request] = {}
        for idx, request in enumerate(requests):
            if request is None:
                continue
            if self._config.cache_within_run:
                if request.fingerprint in self._cache:
                    outcomes[idx] = self._cache[request.fingerprint]
                    reused.add(idx)
                    continue
                if request.fingerprint in grouped:
                    grouped[request.fingerprint].append(idx)
                    reused.add(idx)
                    continue
                grouped[request.fingerprint] = [idx]
            jobs[idx] = request
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=self._config.max_workers)
        try:
            futures = {executor.submit(self._predict, request): idx for idx, request in jobs.items()}
            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                outcome = future.result()
                prediction, seconds = outcome
                self._completed_requests += 1
                self._request_seconds += seconds
                if prediction.input_tokens is None:
                    self._requests_without_usage += 1
                else:
                    assert prediction.output_tokens is not None
                    self._input_tokens += prediction.input_tokens
                    self._output_tokens += prediction.output_tokens
                if self._config.cache_within_run:
                    fingerprint = jobs[idx].fingerprint
                    self._cache[fingerprint] = outcome
                    for draw_idx in grouped[fingerprint]:
                        outcomes[draw_idx] = outcome
                else:
                    outcomes[idx] = outcome
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
        scores: list[float] = []
        costs: list[float] = []
        metadata: dict[correctness_types.ScoredAttemptKey, dict[str, typing.Any]] = {}
        for idx, record in enumerate(records):
            key = (record.sample_id, record.attempt_index, idx)
            request = requests[idx]
            if request is None:
                scores.append(0.0)
                costs.append(0.0)
                metadata[key] = {"decision_type": "missing_answer", "probability_source": None, "cache_reused": False}
                self._missing_answers += 1
                continue
            prediction, seconds = outcomes[idx]
            scores.append(prediction.probability_true)
            if prediction.input_tokens is not None:
                assert prediction.output_tokens is not None
                costs.append(float(prediction.input_tokens + prediction.output_tokens))
            metadata[key] = {
                "decision_type": "decision_model",
                "probability_source": "native_binary_probability",
                **prediction.model_dump(),
                "request_sha256": request.fingerprint,
                "request_seconds": seconds,
                "cache_reused": idx in reused,
            }
        self._total_scored += len(records)
        self._cache_hits += len(reused)
        return correctness_types.ScoringResult(
            scores=scores,
            verification_costs=costs if self._config.provider.require_token_usage else None,
            attempt_metadata=metadata,
        )

    def get_metadata(self) -> dict[str, typing.Any]:
        """Return reproducible configuration and observed completed-request usage.

        Returns:
            Provider and prompt provenance, row/cache counters, and cumulative
            token counts and elapsed seconds for successful requests collected
            by the scorer. This excludes duplicate draws and failed requests;
            provider charges for failed attempts or retries are not inferred.
        """
        return {
            "scorer_type": "decision_model",
            **self._config.model_dump(mode="json", exclude={"prompt_name", "prompt_version"}),
            **self._provenance,
            **self._client.get_metadata(),
            "total_scored": self._total_scored,
            "missing_answers": self._missing_answers,
            "cache_hits": self._cache_hits,
            "completed_api_requests": self._completed_requests,
            "reported_input_tokens": self._input_tokens,
            "reported_output_tokens": self._output_tokens,
            "requests_without_usage": self._requests_without_usage,
            "total_request_seconds": self._request_seconds,
        }

    def get_verification_cost_unit(self) -> str | None:
        """Return the unit for per-draw verification costs.

        Returns:
            "tokens" for required input-plus-output usage, otherwise None.
        """
        return "tokens" if self._config.provider.require_token_usage else None

    def close(self) -> None:
        """Release network resources once, after worker shutdown."""
        if not self._closed:
            self._client.close()
            self._closed = True
