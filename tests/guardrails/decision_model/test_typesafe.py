import concurrent.futures
import contextlib
import json
import typing

import httpx2
import pydantic
import pytest
import typesafe_sdk

import pyine.guardrails.decision_model.configs as configs
import pyine.guardrails.decision_model.typesafe as typesafe_adapter


@pytest.fixture
def response_body() -> dict[str, typing.Any]:
    """Provide a valid System One probability response with paired usage counts."""
    return {
        "model": "jev-test",
        "answers": {"correct": {"type": "noul", "noul": 0.37}},
        "usage": {"input_tokens": 111, "output_tokens": 20},
    }


@pytest.fixture(autouse=True)
def api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set a fake API key so tests never depend on local credentials."""
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key-not-a-secret")


class TestTypeSafeSystemOneClient:
    @pytest.mark.parametrize("probability", [0, 0.37, 0.5, 1])
    def test_wire_contract(
        self,
        config: configs.DecisionModelGuardrailConfig,
        response_body: dict[str, typing.Any],
        probability: float,
    ) -> None:
        """Verify the SDK request shape, endpoint, credentials, and native probability extraction."""
        response_body["answers"]["correct"]["noul"] = probability
        requests: list[httpx2.Request] = []

        def handle(request: httpx2.Request) -> httpx2.Response:
            """Capture the outgoing request and return a successful response with a request ID."""
            requests.append(request)
            return httpx2.Response(200, json=response_body, headers={"x-typesafe-request-id": "req-1"})

        provider = config.provider.model_copy(
            update={"base_url": pydantic.AnyHttpUrl("https://compatible.example/api/")}
        )
        with contextlib.closing(
            typesafe_adapter.TypeSafeSystemOneClient(provider, httpx2.MockTransport(handle))
        ) as client:
            prediction = client.predict("Is the final answer correct?", "some evidence")
        assert prediction.probability_true == probability
        assert (prediction.input_tokens, prediction.output_tokens) == (111, 20)
        assert prediction.request_id == "req-1"
        assert str(requests[0].url) == "https://compatible.example/api/v1/systemone"
        assert json.loads(requests[0].content) == {
            "model": "jev-test",
            "state": "some evidence",
            "questions": {"correct": {"type": "noul", "instructions": "Is the final answer correct?"}},
        }
        assert requests[0].headers["Authorization"] == "Bearer test-key-not-a-secret"
        assert "test-key-not-a-secret" not in provider.model_dump_json()

    @pytest.mark.parametrize("value", [-0.1, 1.1, True, "0.5", None, float("nan"), float("inf")])
    def test_invalid_probabilities_fail(
        self,
        config: configs.DecisionModelGuardrailConfig,
        response_body: dict[str, typing.Any],
        value: typing.Any,
    ) -> None:
        """Reject invalid native probabilities instead of coercing or repairing them."""
        response_body["answers"]["correct"]["noul"] = value
        transport = httpx2.MockTransport(lambda _: httpx2.Response(200, content=json.dumps(response_body)))
        with (
            contextlib.closing(typesafe_adapter.TypeSafeSystemOneClient(config.provider, transport)) as client,
            pytest.raises((typesafe_sdk.TypeSafeError, pydantic.ValidationError, ValueError)),
        ):
            client.predict("correct?", "state")

    @pytest.mark.parametrize("bad_field", ["model", "missing_answer", "future_answer", "usage"])
    def test_invalid_response_fails(
        self,
        config: configs.DecisionModelGuardrailConfig,
        response_body: dict[str, typing.Any],
        bad_field: str,
    ) -> None:
        """Reject unexpected model IDs, missing answers, unsupported types, and missing usage."""
        if bad_field == "model":
            response_body["model"] = "a-different-pinned-model"
        elif bad_field == "missing_answer":
            response_body["answers"] = {}
        elif bad_field == "future_answer":
            response_body["answers"]["correct"] = {"type": "future"}
        else:
            response_body["usage"] = {}
        transport = httpx2.MockTransport(lambda _: httpx2.Response(200, json=response_body))
        with (
            contextlib.closing(typesafe_adapter.TypeSafeSystemOneClient(config.provider, transport)) as client,
            pytest.raises(ValueError),
        ):
            client.predict("correct?", "state")

    def test_alias_request_accepts_resolved_model(
        self,
        config: configs.DecisionModelGuardrailConfig,
        response_body: dict[str, typing.Any],
    ) -> None:
        """Accept an alias request whose response reports the pinned concrete model ID."""
        provider = config.provider.model_copy(update={"model": "jev-alias", "expected_model": "jev-test"})
        transport = httpx2.MockTransport(lambda _: httpx2.Response(200, json=response_body))
        with contextlib.closing(typesafe_adapter.TypeSafeSystemOneClient(provider, transport)) as client:
            prediction = client.predict("correct?", "state")
        assert prediction.model == "jev-test"

    def test_alias_request_rejects_unexpected_resolved_model(
        self,
        config: configs.DecisionModelGuardrailConfig,
        response_body: dict[str, typing.Any],
    ) -> None:
        """Abort when an alias resolves to a model other than the pinned concrete ID."""
        provider = config.provider.model_copy(update={"model": "jev-alias", "expected_model": "jev-other"})
        transport = httpx2.MockTransport(lambda _: httpx2.Response(200, json=response_body))
        with (
            contextlib.closing(typesafe_adapter.TypeSafeSystemOneClient(provider, transport)) as client,
            pytest.raises(ValueError, match="expected model 'jev-other'"),
        ):
            client.predict("correct?", "state")

    @pytest.mark.parametrize("status, expected_calls", [(400, 1), (401, 1), (429, 2), (503, 2)])
    def test_sdk_retries_are_bounded(
        self,
        config: configs.DecisionModelGuardrailConfig,
        status: int,
        expected_calls: int,
    ) -> None:
        """Retry transient SDK failures within the configured budget only."""
        calls = 0

        def handle(_: httpx2.Request) -> httpx2.Response:
            """Count HTTP attempts and return the selected failure status."""
            nonlocal calls
            calls += 1
            return httpx2.Response(status, json={"message": "failure"}, headers={"retry-after-ms": "0"})

        provider = config.provider.model_copy(update={"max_retries": 1})
        with (
            contextlib.closing(
                typesafe_adapter.TypeSafeSystemOneClient(provider, httpx2.MockTransport(handle))
            ) as client,
            pytest.raises(typesafe_sdk.TypeSafeAPIError),
        ):
            client.predict("correct?", "state")
        assert calls == expected_calls

    def test_timeout_and_concurrent_lifecycle(
        self,
        config: configs.DecisionModelGuardrailConfig,
        response_body: dict[str, typing.Any],
    ) -> None:
        """Share the client across workers, close its transport, and propagate configured timeouts."""

        class Transport(httpx2.MockTransport):
            closed = False

            def close(self) -> None:
                """Record transport cleanup after concurrent predictions finish."""
                self.closed = True

        transport = Transport(lambda _: httpx2.Response(200, json=response_body))
        with contextlib.closing(typesafe_adapter.TypeSafeSystemOneClient(config.provider, transport)) as client:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                predictions = list(executor.map(lambda idx: client.predict("correct?", str(idx)), range(8)))
            assert [prediction.probability_true for prediction in predictions] == [0.37] * 8
        assert transport.closed

        def timeout(request: httpx2.Request) -> typing.NoReturn:
            """Check the configured read timeout and simulate an expired request."""
            assert request.extensions["timeout"]["read"] == config.provider.timeout_seconds
            raise httpx2.ReadTimeout("timed out", request=request)

        with (
            contextlib.closing(
                typesafe_adapter.TypeSafeSystemOneClient(config.provider, httpx2.MockTransport(timeout))
            ) as client,
            pytest.raises(typesafe_sdk.TypeSafeAPITimeoutError),
        ):
            client.predict("correct?", "state")

    def test_usage_can_be_explicitly_unknown(
        self,
        config: configs.DecisionModelGuardrailConfig,
        response_body: dict[str, typing.Any],
    ) -> None:
        """Accept absent usage only when the provider configuration allows it."""
        response_body["usage"] = {}
        provider = config.provider.model_copy(update={"require_token_usage": False})
        transport = httpx2.MockTransport(lambda _: httpx2.Response(200, json=response_body))
        with contextlib.closing(typesafe_adapter.TypeSafeSystemOneClient(provider, transport)) as client:
            assert client.predict("correct?", "state").input_tokens is None

    def test_missing_key_is_rejected(
        self,
        monkeypatch: pytest.MonkeyPatch,
        config: configs.DecisionModelGuardrailConfig,
    ) -> None:
        """Fail client construction when its API-key environment variable is missing."""
        monkeypatch.delenv("TYPESAFE_API_KEY")
        with pytest.raises(ValueError, match="TYPESAFE_API_KEY"):
            typesafe_adapter.TypeSafeSystemOneClient(config.provider)
