import dataclasses
import threading
import unittest.mock

import pytest

import pyine.guardrails.decision_model.configs as configs
import pyine.guardrails.decision_model.scorer as scorer_module
import tests.guardrails.decision_model.helpers as helpers


class TestScorer:
    def test_reuse_and_draw_alignment(
        self,
        config: configs.DecisionModelGuardrailConfig,
    ) -> None:
        """Reuse predictions while preserving duplicate draws, missing answers, and usage totals."""
        client = helpers.FakeClient()
        scorer = scorer_module.DecisionModelGuardrailScorer(config, client)
        correct = helpers.make_record()
        wrong = helpers.make_record(final_answer="6", label=False, attempt_index=1)
        missing = helpers.make_record(final_answer=None)
        result = scorer.score_records([correct, wrong, correct, missing])
        assert result.scores == [0.99, 0.01, 0.99, 0.0]
        assert result.verification_costs == [120, 120, 120, 0]
        assert list(result.attempt_metadata) == [
            (correct.sample_id, 0, 0),
            (wrong.sample_id, 1, 1),
            (correct.sample_id, 0, 2),
            (missing.sample_id, 0, 3),
        ]
        assert result.attempt_metadata[(correct.sample_id, 0, 2)]["cache_reused"]
        assert "probability_true" not in result.attempt_metadata[(missing.sample_id, 0, 3)]
        assert scorer.score_records([correct]).scores == [0.99]
        assert len(client.calls) == 2
        metadata = scorer.get_metadata()
        assert metadata["completed_api_requests"] == 2
        assert metadata["reported_input_tokens"] == 200
        assert metadata["cache_hits"] == 2
        assert metadata["resolved_prompt_version"] == "binary_correctness_v1"
        assert "expected_output" not in client.calls[0][1]
        assert scorer.score_records([]).scores == []
        scorer.close()
        assert client.closed
        with pytest.raises(RuntimeError, match="closed"):
            scorer.score_records([correct])

    def test_disable_reuse_and_unknown_costs(
        self,
        config: configs.DecisionModelGuardrailConfig,
    ) -> None:
        """Allow explicit independent requests and unknown verification costs."""
        config = config.model_copy(
            update={
                "cache_within_run": False,
                "provider": config.provider.model_copy(update={"require_token_usage": False}),
            }
        )

        class UnknownUsageClient(helpers.FakeClient):
            def predict(
                self,
                instructions: str,
                state: str,
            ) -> configs.DecisionPrediction:
                """Return a valid prediction without provider token counts."""
                self.calls.append((instructions, state))
                return configs.DecisionPrediction(probability_true=0.5, model="jev-test")

        client = UnknownUsageClient()
        scorer = scorer_module.DecisionModelGuardrailScorer(config, client)
        record = helpers.make_record()
        result = scorer.score_records([record, record])
        assert len(client.calls) == 2
        assert result.verification_costs is None
        assert scorer.get_verification_cost_unit() is None
        assert scorer.get_metadata()["requests_without_usage"] == 2
        scorer.close()

    def test_labels_references_and_identifiers_do_not_change_requests(
        self,
        config: configs.DecisionModelGuardrailConfig,
    ) -> None:
        """Exclude labels, references, and sample identifiers from provider evidence."""
        client = helpers.FakeClient()
        scorer = scorer_module.DecisionModelGuardrailScorer(config, client)
        first = helpers.make_record()
        second = dataclasses.replace(
            first,
            sample_id="other",
            label=False,
            expected_output="TOP SECRET",
            difficulty_score=0.9,
            record={**first.record, "expected_output": "TOP SECRET", "labels": "SECRET", "sample": {"secret": 1}},
        )
        scorer.score_records([first, second])
        assert len(client.calls) == 1
        assert "SECRET" not in str(client.calls)
        scorer.close()

    def test_out_of_order_completion_and_limiter(
        self,
        config: configs.DecisionModelGuardrailConfig,
    ) -> None:
        """Restore input order after synchronized out-of-order calls and acquire the limiter."""
        second_completed = threading.Event()
        limiter = unittest.mock.Mock()

        class ReorderedClient(helpers.FakeClient):
            def predict(
                self,
                instructions: str,
                state: str,
            ) -> configs.DecisionPrediction:
                """Complete the second request first without relying on sleeps."""
                if "<final>\n5\n" in state:
                    assert second_completed.wait(timeout=5)
                else:
                    second_completed.set()
                assert limiter.acquire.called
                return super().predict(instructions, state)

        with unittest.mock.patch.object(
            scorer_module.langchain_core.rate_limiters, "InMemoryRateLimiter", return_value=limiter
        ):
            scorer = scorer_module.DecisionModelGuardrailScorer(config, ReorderedClient())
        assert scorer.score_records([helpers.make_record(), helpers.make_record(final_answer="6")]).scores == [
            0.99,
            0.01,
        ]
        assert limiter.acquire.call_args_list == [unittest.mock.call(blocking=True)] * 2
        scorer.close()

    @pytest.mark.parametrize("failure", ["exception", "model", "usage"])
    def test_failures_raise_and_are_not_cached(
        self,
        config: configs.DecisionModelGuardrailConfig,
        failure: str,
    ) -> None:
        """Propagate prediction failures and permit a later successful uncached request."""
        client = unittest.mock.Mock(spec=helpers.FakeClient)
        if failure == "exception":
            client.predict.side_effect = RuntimeError("outage")
        else:
            client.predict.return_value = configs.DecisionPrediction(
                probability_true=0.5,
                model="wrong" if failure == "model" else "jev-test",
            )
        scorer = scorer_module.DecisionModelGuardrailScorer(config, client)
        with pytest.raises((ValueError, RuntimeError)):
            scorer.score_records([helpers.make_record()])
        client.predict.side_effect = None
        client.predict.return_value = configs.DecisionPrediction(
            probability_true=1,
            model="jev-test",
            input_tokens=1,
            output_tokens=0,
        )
        assert scorer.score_records([helpers.make_record()]).scores == [1]
        assert client.predict.call_count == 2
        scorer.close()
        client.close.assert_called_once()


class TestPreflight:
    def test_training_subset_is_rejected(
        self,
        config: configs.DecisionModelGuardrailConfig,
    ) -> None:
        """Reject guardrail training data as an evaluation population."""
        records = [helpers.make_record(), helpers.make_record(label=False)]
        with pytest.raises(ValueError, match="evaluation subsets"):
            scorer_module.preflight_records(config, records, {"guardrail_train": records})

    @pytest.mark.parametrize("bad_population", ["empty", "missing", "single_class"])
    def test_invalid_populations_raise(
        self,
        config: configs.DecisionModelGuardrailConfig,
        bad_population: str,
    ) -> None:
        """Reject empty evaluations, missing-answer populations, and single-label calibration."""
        calibration = [helpers.make_record(), helpers.make_record(label=False)]
        evaluation = [helpers.make_record()]
        if bad_population == "empty":
            evaluation = []
        elif bad_population == "missing":
            evaluation = [helpers.make_record(final_answer=None)]
        else:
            calibration = [helpers.make_record()]
        with pytest.raises(ValueError):
            scorer_module.preflight_records(config, calibration, {"guardrail_test": evaluation})

    def test_counts_include_calibration_for_each_subset(
        self,
        config: configs.DecisionModelGuardrailConfig,
    ) -> None:
        """Count repeated calibration calls when request reuse is disabled."""
        records = [helpers.make_record(), helpers.make_record(final_answer="6", label=False)]
        stats = scorer_module.preflight_records(
            config, records, {"guardrail_test": records, "guardrail_valid": records}
        )
        assert stats["expected_api_calls"] == 2
        stats = scorer_module.preflight_records(
            config.model_copy(update={"cache_within_run": False}),
            records,
            {"guardrail_test": records, "guardrail_valid": records},
        )
        assert stats["expected_api_calls"] == 8
