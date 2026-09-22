import pytest

import pyine.guardrails.decision_model.configs as configs


@pytest.fixture
def config() -> configs.DecisionModelGuardrailConfig:
    """Provide a pinned mock provider with minimal rate-limiter delay."""
    return configs.DecisionModelGuardrailConfig(
        provider=configs.DecisionModelProviderConfig(model="jev-test", max_retries=0),
        requests_per_second=100000,
    )
