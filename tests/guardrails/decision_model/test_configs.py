import pydantic
import pytest

import pyine.guardrails.decision_model.configs as configs


class TestDecisionModelProviderConfig:
    def test_credential_url_is_rejected(self) -> None:
        """Reject embedded credentials before endpoint URLs enter saved metadata."""
        with pytest.raises(pydantic.ValidationError):
            configs.DecisionModelProviderConfig(model="test", base_url="https://user:secret@example.com")
