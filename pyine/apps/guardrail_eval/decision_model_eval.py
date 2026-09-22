"""Standalone Hydra entrypoint for remote decision-model guardrail evaluation.

Runs the correctness evaluation pipeline with a remote model that returns native
binary probabilities as the guardrail scorer. No training step - the scorer is
built directly from a decision provider config. The initial adapter supports
TypeSafe Jev and compatible System One APIs.

Set TYPESAFE_API_KEY in the project .env, then provide predictor benchmark LMDB
exports and their original problem split source (a dataset name or split file).

CLI usage::

    uv run python -m pyine.apps.guardrail_eval.decision_model_eval \\
        +experiment=guardrail/decision_model_eval_jev \\
        'config.evals_config.datamodule_config.lmdb_paths=[/path/to/benchmark-export]' \\
        config.evals_config.datamodule_config.split_config.split_source=TACO

Add runtime.dry_run=true to validate configuration without loading datasets or
calling the API. See DECISION_MODEL_EVAL_GUIDE.md for comparison settings and
result interpretation.
"""

import asyncio
import contextlib
import logging
import typing

import pyine.apps.guardrail_eval.decision_model_eval_configs as app_configs
import pyine.configs.schemas
import pyine.evals.common
import pyine.evals.correctness._impl as correctness_impl
import pyine.evals.correctness.datamodule as correctness_datamodule
import pyine.evals.persistence
import pyine.evals.utils
import pyine.guardrails.decision_model.scorer as decision_scorer
import pyine.utils.reprod

logger = logging.getLogger(__name__)


async def main(
    config: app_configs.DecisionModelEvalAppConfig,
    runtime: pyine.configs.schemas.RuntimeConfig | None = None,
) -> None:
    """Preflight inputs, score calibration/evaluation populations, and persist results.

    Dataset and provider failures propagate after registered resources are closed.
    A dry run validates configuration without loading data or constructing a client.

    Args:
        config: Guardrail, evaluation, result-persistence, and optional W&B settings.
        runtime: Runtime setup and cleanup settings, normally supplied by Hydra.

    Raises:
        ValueError: Evaluation subsets are repeated or preflight rejects the selected data.
        RuntimeError: W&B logging is requested without an initialized runtime W&B run.
    """
    dry_run = runtime is not None and runtime.dry_run
    wandb_init_kwargs: dict[str, typing.Any] = {}
    if config.wandb_project is not None:
        wandb_init_kwargs["project"] = config.wandb_project
    with contextlib.ExitStack() as cleanup:
        if runtime is not None:
            cleanup.callback(runtime.finalize)
        pyine.utils.reprod.entrypoint_setup(
            runtime_config=runtime,
            use_wandb_logging=config.use_wandb_logging and not dry_run,
            wandb_init_kwargs=wandb_init_kwargs or None,
            main_config=config,
        )
        if dry_run:
            logger.info("dry run: configuration validated; skipping datasets and API client construction")
            return
        datamodule = config.evals_config.datamodule_config.instantiate_datamodule()
        assert isinstance(datamodule, correctness_datamodule.CorrectnessDataModule)
        cleanup.callback(datamodule.teardown)  # register before setup to clean up partial initialization
        datamodule.prepare_data()
        datamodule.setup()
        subset_names = datamodule.config.eval_subset_names
        if len(set(subset_names)) != len(subset_names):
            raise ValueError("evaluation subsets must not be repeated")
        preflight = decision_scorer.preflight_records(
            config.guardrail_config,
            datamodule.get_records_for_calibration(config.evals_config.calibration_resampling),
            {name: datamodule.get_records_for_subset(name) for name in subset_names},
        )
        logger.info(f"decision-model preflight: {preflight}")
        scorer = decision_scorer.DecisionModelGuardrailScorer(config.guardrail_config)
        cleanup.callback(scorer.close)
        if config.use_wandb_logging and runtime is not None and runtime.wandb_run is not None:
            config.evals_config.define_metrics_for_wandb(
                wandb_run=runtime.wandb_run,
                eval_subset_names=subset_names,
            )
        evaluation_results: dict[str, pyine.evals.common.EvalResult] = {}
        for subset_name in subset_names:
            logger.info(f"evaluating subset: {subset_name}")
            result = await correctness_impl.evaluate_guardrail_replicas(
                config=config.evals_config,
                guardrails=[scorer],
                datamodule=datamodule,
                eval_subset_name=subset_name,
            )
            result.eval_metadata["decision_model_preflight"] = preflight
            pyine.evals.utils.print_metrics(result.metrics, subset_name, logger.info)
            pyine.evals.persistence.maybe_dump_eval_result(
                result=result,
                dump_dir=config.evals_config.result_dump_dir,
                eval_subset_name=subset_name,
                overwrite=config.evals_config.result_dump_overwrite,
            )
            evaluation_results[subset_name] = result
        if config.use_wandb_logging and evaluation_results:
            if runtime is None or runtime.wandb_run is None:
                raise RuntimeError("runtime configuration with wandb run is required when logging to wandb")
            config.evals_config.log_metrics(wandb_run=runtime.wandb_run, results_by_subset=evaluation_results)
        logger.info(f"decision-model usage/provenance: {scorer.get_metadata()}")
    logger.info("decision-model guardrail evaluation complete")


def async_decision_model_eval_main_wrapper(
    config: app_configs.DecisionModelEvalAppConfig,
    runtime: pyine.configs.schemas.RuntimeConfig | None = None,
) -> None:
    """Run the async application from Hydra's synchronous entrypoint.

    Args:
        config: Fully instantiated decision-model evaluation configuration.
        runtime: Optional runtime settings forwarded to main.
    """
    asyncio.run(main(config=config, runtime=runtime))


if __name__ == "__main__":
    import pyine.apps.trainers.common

    pyine.apps.trainers.common.hydra_main(
        eval_type=pyine.evals.common.EvalType.CORRECTNESS,
        hydra_config_registration_fn=app_configs.register_hydra_configs,
        async_main_wrapper=async_decision_model_eval_main_wrapper,
    )
