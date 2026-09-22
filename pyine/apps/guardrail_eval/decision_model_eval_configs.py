"""Hydra-zen config builder for the decision model guardrail evaluation app."""

from __future__ import annotations

import typing

import pydantic

import pyine.configs.base
import pyine.configs.schemas
import pyine.configs.searchpath
import pyine.configs.utils
import pyine.evals.common
import pyine.evals.correctness.configs
import pyine.guardrails.decision_model.configs
import pyine.utils.reprod


class DecisionModelEvalAppConfig(pydantic.BaseModel):
    """Standalone evaluation app for the decision model guardrail.

    Requires no training - builds the scorer from a decision provider config
    and runs the correctness evaluation pipeline on an LMDB dataset.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    guardrail_config: pyine.guardrails.decision_model.configs.DecisionModelGuardrailConfig
    """Provider, prompt, concurrency, and scoring policies for the decision-model guardrail."""
    evals_config: pyine.evals.correctness.configs.CorrectnessEvalsConfig
    """Correctness evaluation pipeline, calibration, and result-persistence settings.

    The embedded CorrectnessDataModuleConfig selects the LMDB exports and problem splits.
    """
    use_wandb_logging: bool = False
    """Whether to log results to Weights & Biases."""
    wandb_project: str | None = None
    """W&B project name (only used when use_wandb_logging=True)."""


def _get_app_configs(group: str) -> list[pyine.configs.schemas.ConfigDescription]:
    """Build decision-model evaluation configuration descriptions for Hydra-Zen.

    Args:
        group: Hydra configuration group containing the app and nested evaluation settings.

    Returns:
        Base app configuration followed by the available evaluation configurations.
    """
    evals_configs = pyine.evals.correctness.configs.get_evals_configs(
        group=f"{group}/evals_config",
    )
    app_main_config = pyine.configs.utils.make_config_description(
        DecisionModelEvalAppConfig,
        name="base",
        group=group,
        description="Base settings for the decision model guardrail eval app.",
        config={
            "populate_full_signature": True,
            "hydra_convert": "object",
        },
    )
    return [app_main_config, *evals_configs]


def register_hydra_configs(
    eval_type: pyine.evals.common.EvalType,
) -> list[pyine.configs.schemas.ConfigDescription]:
    """Register the entrypoint, runtime, app, and external experiment configurations.

    Args:
        eval_type: Evaluation family used to select compatible external configurations.

    Returns:
        All registered configuration descriptions for CLI discovery and composition.
    """
    import pyine.apps.guardrail_eval.decision_model_eval as decision_app

    pyine.utils.reprod.load_dotenv()
    entrypoint_config = pyine.configs.utils.make_config_description(
        decision_app.async_decision_model_eval_main_wrapper,
        name="entrypoint",
        group=None,
        description="Entrypoint settings for the decision model guardrail eval app.",
        config={
            "populate_full_signature": True,
            "hydra_defaults": [
                "_self_",
                {"config": "base"},
                {"runtime": "default"},
                *pyine.configs.base.get_base_hydra_default_overrides(),
            ],
        },
    )
    store, base_configs = pyine.configs.base.get_base_store_and_configs("decision_model_eval")
    app_configs = _get_app_configs(group="config")
    configs_to_register = [entrypoint_config, *app_configs]
    external_configs = pyine.configs.searchpath.SearchPathPlugin.get_external_configs(
        app_name="decision_model_eval",
        eval_type=eval_type,
        entrypoint_config=entrypoint_config,
        app_configs=[*base_configs, *configs_to_register],
    )
    configs_to_register.extend(external_configs)
    for config in configs_to_register:
        assert config.name is not None
        store(
            typing.cast("typing.Any", config.config),
            name=config.name,
            group=config.group,
            package=config.package,
        )
    store.add_to_hydra_store(overwrite_ok=True)
    return [*base_configs, *configs_to_register]


if __name__ == "__main__":
    import sys

    pyine.configs.base.register_searchpath_plugin()
    pyine.configs.utils.print_experiment_configs(
        config_descriptions=register_hydra_configs(eval_type=pyine.evals.common.EvalType.CORRECTNESS),
        app_name="decision_model_eval",
        cli_args=sys.argv[1:],
    )
