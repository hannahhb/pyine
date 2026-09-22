"""Synthetic records and provider clients shared by decision-model tests."""

import typing

import pyine.evals.correctness.types as correctness_types
import pyine.guardrails.decision_model.configs as configs


def make_record(
    sample_id: str = "DEBUG/VALID/p000000/s0000/t0000",
    final_answer: str | None = "5",
    label: bool = True,
    **overrides: typing.Any,
) -> correctness_types.EvalRecord:
    """Build a small correctness record with explicit per-test overrides."""
    values = {
        "sample_id": sample_id,
        "problem_id": sample_id.rsplit("/", 2)[0],
        "attempt_index": 0,
        "model_output": f"The answer is {final_answer}.",
        "final_answer": final_answer,
        "expected_output": "5",
        "label": label,
        "code_type": "original",
        "tags": [],
        "record": {"prompt": "Predict the output: print(2 + 3)"},
        "difficulty_score": None,
    }
    values.update(overrides)
    return correctness_types.EvalRecord(**values)


class FakeClient:
    def __init__(self) -> None:
        """Initialize request tracking and client-lifecycle state."""
        self.calls: list[tuple[str, str]] = []
        self.closed = False

    def predict(
        self,
        instructions: str,
        state: str,
    ) -> configs.DecisionPrediction:
        """Record the request and return deterministic probabilities and token counts."""
        self.calls.append((instructions, state))
        return configs.DecisionPrediction(
            probability_true=0.99 if "<final>\n5\n" in state else 0.01,
            model="jev-test",
            input_tokens=100,
            output_tokens=20,
        )

    def get_metadata(self) -> dict[str, typing.Any]:
        """Identify the fake SDK in evaluation provenance."""
        return {"sdk": "fake"}

    def close(self) -> None:
        """Record that the scorer released its client."""
        self.closed = True
