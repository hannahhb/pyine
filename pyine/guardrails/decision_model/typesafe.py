"""Small official-SDK adapter for the TypeSafe System One/Noul contract."""

import os
import typing

import typesafe_sdk

import pyine.guardrails.decision_model.configs as decision_configs

if typing.TYPE_CHECKING:
    import httpx2


class TypeSafeSystemOneClient:
    """Own a synchronous SDK client whose HTTP connection pool is thread safe."""

    def __init__(
        self,
        config: decision_configs.DecisionModelProviderConfig,
        transport: "httpx2.BaseTransport | None" = None,
    ) -> None:
        """Resolve the credential and construct the owned synchronous SDK client.

        Args:
            config: Endpoint, pinned model, API-key environment name, and HTTP settings.
            transport: Optional SDK HTTP transport, primarily for isolated tests.
                If None, the SDK creates its default transport.

        Raises:
            ValueError: The configured API-key environment variable is empty or missing.
        """
        api_key = os.environ.get(config.api_key_env)
        if not api_key or not api_key.strip():
            raise ValueError(f"missing API key environment variable: {config.api_key_env}")
        self._config = config
        self._client = typesafe_sdk.TypeSafeClient(
            api_key=api_key,
            model=config.model,
            base_url=str(config.base_url),
            timeout=config.timeout_seconds,
            retry=typesafe_sdk.RetryPolicy(max_retries=config.max_retries),
            transport=transport,
        )

    def predict(
        self,
        instructions: str,
        state: str,
    ) -> decision_configs.DecisionPrediction:
        """Extract the native Noul probability without converting or repairing it.

        Args:
            instructions: Binary question and judging rubric, sent under the "correct" key.
            state: Rendered evidence to evaluate against the question.

        Returns:
            Native probability with the returned model ID, optional paired token
            counts, and optional provider request ID.

        Raises:
            ValueError: The response has an unexpected model, answer type, or missing
                required usage, or fails prediction validation.
            typesafe_sdk.TypeSafeError: The SDK request fails after configured retries.
        """
        response = self._client.system_one(  # pyright: ignore[reportUnknownMemberType]
            state=state,
            questions={"correct": typesafe_sdk.Noul(instructions=instructions)},
        )
        expected_model = self._config.expected_model or self._config.model
        if response.model != expected_model:
            raise ValueError(f"expected model {expected_model!r}, received {response.model!r}")
        answer = response.answers.get("correct")
        if not isinstance(answer, typesafe_sdk.NoulAnswer):
            raise ValueError("expected a Noul answer under 'correct'")
        if self._config.require_token_usage and (
            response.usage.input_tokens is None or response.usage.output_tokens is None
        ):
            raise ValueError("the configured provider must report input and output token counts")
        return decision_configs.DecisionPrediction(
            probability_true=answer.noul,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            request_id=response.raw_http_response.headers.get("x-typesafe-request-id"),
        )

    def get_metadata(self) -> dict[str, typing.Any]:
        """Return SDK provenance; configuration is recorded separately by the scorer.

        Returns:
            SDK name and installed version, without credentials.
        """
        return {"sdk": "typesafe-sdk", "sdk_version": typesafe_sdk.__version__}

    def close(self) -> None:
        """Release the SDK client and its connection pool."""
        self._client.close()
