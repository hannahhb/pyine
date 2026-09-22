"""Configuration and provider-neutral binary decision contract."""

import typing

import pydantic


class DecisionModelProviderConfig(pydantic.BaseModel):
    """Settings for a pinned model behind a TypeSafe-compatible endpoint.

    Credentials are resolved from the environment when the client is created;
    this configuration can be persisted as experiment metadata.
    """

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    backend: typing.Literal["typesafe_systemone"] = "typesafe_systemone"
    """Provider adapter implementing the System One binary-question API contract."""
    base_url: pydantic.AnyHttpUrl = pydantic.AnyHttpUrl("https://api.typesafe.ai")
    """SDK base URL; the adapter uses its System One route to submit questions.

    Credentials, query parameters, and fragments are forbidden because the URL
    is included in saved metadata and request fingerprints.
    """
    api_key_env: str = pydantic.Field(default="TYPESAFE_API_KEY", min_length=1)
    """Name of the environment variable containing the API key, never the key itself."""
    model: str = pydantic.Field(min_length=1)
    """Pinned model ID to request; every response must report this exact ID."""
    timeout_seconds: float = pydantic.Field(default=60.0, gt=0, allow_inf_nan=False)
    """HTTP timeout in seconds passed to the SDK, excluding retry backoff."""
    max_retries: int = pydantic.Field(default=3, ge=0)
    """Maximum SDK retries after the initial attempt; zero disables retries."""
    require_token_usage: bool = True
    """Whether every prediction must report both input and output token counts.

    Disable only for compatible backends without usage reporting. The scorer
    then returns no verification costs or cost unit, even if some counts exist.
    """

    @pydantic.field_validator("base_url")
    @classmethod
    def validate_endpoint(
        cls,
        value: pydantic.AnyHttpUrl,
    ) -> pydantic.AnyHttpUrl:
        """Validate that the endpoint can be persisted in experiment metadata.

        Args:
            value: Parsed SDK base URL.

        Returns:
            The validated URL, unchanged.

        Raises:
            ValueError: The URL contains credentials, query parameters, or a fragment.
        """
        if value.username or value.password or value.query or value.fragment:
            raise ValueError("base_url must not contain credentials, query parameters, or fragments")
        return value


class DecisionModelGuardrailConfig(pydantic.BaseModel):
    """Shared prompt, probability policy, and bounded remote execution."""

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    provider: DecisionModelProviderConfig
    """Provider endpoint, pinned model, authentication source, and HTTP settings."""
    prompt_name: str = "guardrail/correctness_judge"
    """PromptManager template name shared with the prompted LLM correctness judge."""
    prompt_version: str = pydantic.Field(default="binary_correctness_v1", min_length=1)
    """Explicit rubric version whose metadata declares binary_correctness: true.

    Compatibility is checked from the resolved template rather than its version
    name, so compatible future versions need no scorer changes.
    """
    max_workers: int = pydantic.Field(default=4, ge=1)
    """Maximum concurrent prediction calls in the scorer's thread pool."""
    requests_per_second: float = pydantic.Field(default=5.0, gt=0, allow_inf_nan=False)
    """Logical prediction starts per second, using InMemoryRateLimiter with a one-request bucket.

    SDK retries happen within each prediction and do not acquire this limiter.
    """
    missing_answer_policy: typing.Literal["score_zero"] = "score_zero"
    """Assign score zero to records without a final answer, without an API call.

    These policy-assigned scores are marked separately from native probabilities.
    App preflight rejects populations in which every final answer is missing.
    """
    cache_within_run: bool = True
    """Reuse successful predictions for identical requests across scorer calls.

    Identity includes provider, endpoint, model, instructions, and evidence.
    Duplicate draws retain their row positions and evaluation weights; failures
    are never cached. The cache lasts for the lifetime of the scorer.
    """


class DecisionPrediction(pydantic.BaseModel):
    """A native probability and optional usage, without provider-specific objects."""

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid", strict=True)

    probability_true: float = pydantic.Field(ge=0.0, le=1.0, allow_inf_nan=False)
    """Native probability that the requested statement is true, without rescaling."""
    model: str
    """Model ID reported by the provider and checked against the requested ID."""
    input_tokens: int | None = pydantic.Field(default=None, ge=0)
    """Provider-reported input tokens, or None when both usage counts are unavailable."""
    output_tokens: int | None = pydantic.Field(default=None, ge=0)
    """Provider-reported output tokens, or None when both usage counts are unavailable."""
    request_id: str | None = None
    """Provider request identifier for diagnostics, if returned by the API."""

    @pydantic.model_validator(mode="after")
    def validate_usage(self) -> typing.Self:
        """Require paired counts so missing usage cannot become a partial total.

        Returns:
            The validated prediction, unchanged.

        Raises:
            ValueError: Exactly one of the input and output token counts is missing.
        """
        if (self.input_tokens is None) != (self.output_tokens is None):
            raise ValueError("input_tokens and output_tokens must both be present or both absent")
        return self


class DecisionModelClient(typing.Protocol):
    """Synchronous client supporting concurrent independent binary questions.

    Implementations must permit predict calls from multiple worker threads and
    propagate request failures. The scorer owns the client and closes it after
    outstanding predictions finish.
    """

    def predict(
        self,
        instructions: str,
        state: str,
    ) -> DecisionPrediction:
        """Return a native probability, propagating provider or validation errors.

        Args:
            instructions: Binary question and the rubric for judging its truth.
            state: Rendered evidence needed to answer the question.

        Returns:
            Probability that the statement is true, with model and usage metadata.
        """
        ...

    def get_metadata(self) -> dict[str, typing.Any]:
        """Return provider provenance without credentials.

        Returns:
            JSON-serializable provider or SDK metadata for the saved evaluation.
        """
        ...

    def close(self) -> None:
        """Close owned network resources after all requests finish."""
        ...
