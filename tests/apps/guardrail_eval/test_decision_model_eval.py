import dataclasses
import pathlib
import unittest.mock

import hydra
import hydra.utils
import pytest

import pyine.apps.guardrail_eval.decision_model_eval as app
import pyine.apps.guardrail_eval.decision_model_eval_configs as app_configs
import pyine.apps.guardrail_eval.prompted_llm_eval_configs as monitor_configs
import pyine.configs.base
import pyine.evals.common
import pyine.evals.correctness._impl as correctness_impl
import pyine.evals.correctness.configs as eval_configs
import pyine.evals.correctness.datamodule as datamodule
import pyine.evals.correctness.datamodule_configs as datamodule_configs
import pyine.evals.correctness.types as correctness_types
import pyine.evals.persistence
import pyine.guardrails.data.debug_dataset as debug_dataset
import pyine.guardrails.decision_model.configs as decision_configs
import pyine.guardrails.decision_model.typesafe as typesafe_adapter
import tests.guardrails.decision_model.helpers as helpers


@pytest.fixture
def config(tmp_path: pathlib.Path) -> app_configs.DecisionModelEvalAppConfig:
    """Create an isolated LMDB export, split file, and evaluation configuration."""
    lmdb = debug_dataset.create_debug_probe_lmdb(
        tmp_path / "lmdb",
        n_train=0,
        n_eval_families=12,
        n_test_families=4,
        noise_rate=0,
    )
    split_file = debug_dataset.create_debug_split_file(
        tmp_path / "splits.pkl",
        n_train=0,
        n_eval_families=12,
        n_test_families=4,
    )
    return app_configs.DecisionModelEvalAppConfig(
        guardrail_config=decision_configs.DecisionModelGuardrailConfig(
            provider=decision_configs.DecisionModelProviderConfig(model="jev-test", max_retries=0),
            requests_per_second=100000,
        ),
        evals_config=eval_configs.CorrectnessEvalsConfig(
            datamodule_config=datamodule_configs.CorrectnessDataModuleConfig(
                lmdb_paths=(lmdb,),
                split_config=correctness_types.GuardrailSplitConfig(split_source=str(split_file)),
                eval_subset_names=("guardrail_test",),
            ),
            target_fpr_values=[0.01, 0.1],
            num_bootstrap_replicates=2,
            bootstrap_num_workers=1,
            roc_fpr_grid_size=10,
            result_dump_dir=tmp_path / "results",
        ),
    )


@pytest.fixture
def mocked_setup(monkeypatch: pytest.MonkeyPatch) -> unittest.mock.Mock:
    """Replace runtime setup to avoid external logging and environment changes."""
    setup = unittest.mock.Mock()
    monkeypatch.setattr(app.pyine.utils.reprod, "entrypoint_setup", setup)
    return setup


@pytest.mark.integration
class TestDecisionModelApp:
    def test_matched_monitor_overlay_composes(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Compose the matched LLM monitor overlay with its shared binary rubric."""
        monkeypatch.setattr(app_configs.pyine.utils.reprod, "load_dotenv", lambda: None)
        pyine.configs.base.register_searchpath_plugin()
        monitor_configs.register_hydra_configs(pyine.evals.common.EvalType.CORRECTNESS)
        with hydra.initialize(version_base=None, config_path=None):
            composed = hydra.compose(
                config_name="entrypoint",
                overrides=[
                    "+experiment=guardrail/prompted_llm_eval_binary_openai",
                    "config.evals_config.datamodule_config.lmdb_paths=[/unused]",
                    "config.evals_config.datamodule_config.split_config.split_source=TACO",
                    "config.evals_config.result_dump_dir=null",
                ],
            )
            instantiated = hydra.utils.instantiate(composed.config)
            assert composed.runtime.run_name.startswith("gpt-5-mini/binary_correctness_v1/")
        assert isinstance(instantiated, monitor_configs.PromptedLLMEvalAppConfig)
        assert instantiated.guardrail_config.prompt_version == "binary_correctness_v1"
        assert instantiated.guardrail_config.default_score_on_missing_answer == 0
        assert instantiated.evals_config.datamodule_config.eval_subset_names == ("guardrail_test",)

    @pytest.mark.asyncio
    async def test_real_lmdb_pipeline_and_roundtrip(
        self,
        config: app_configs.DecisionModelEvalAppConfig,
        monkeypatch: pytest.MonkeyPatch,
        mocked_setup: unittest.mock.Mock,
    ) -> None:
        """Evaluate tied probabilities through the LMDB pipeline and reload persisted results."""

        class TiedClient(helpers.FakeClient):
            def predict(
                self,
                instructions: str,
                state: str,
            ) -> decision_configs.DecisionPrediction:
                """Return identical unit probabilities to exercise conservative FPR thresholds."""
                self.calls.append((instructions, state))
                return decision_configs.DecisionPrediction(
                    probability_true=1.0,
                    model="jev-test",
                    input_tokens=100,
                    output_tokens=20,
                )

        client = TiedClient()
        monkeypatch.setattr(typesafe_adapter, "TypeSafeSystemOneClient", lambda _: client)
        runtime = unittest.mock.Mock(dry_run=False, wandb_run=None)
        await app.main(config, runtime)
        assert client.closed
        runtime.finalize.assert_called_once()
        path = config.evals_config.result_dump_dir / "guardrail_test.pkl"
        result = pyine.evals.persistence.load_eval_result(path, correctness_impl.CorrectnessEvalResult)
        run = result.aggregated.per_run[0]
        assert run.guardrail_metadata["resolved_prompt_version"] == "binary_correctness_v1"
        assert run.guardrail_metadata["completed_api_requests"] == len(client.calls)
        assert run.attempt_metadata
        assert all(meta["probability_true"] == 1.0 for meta in run.attempt_metadata.values())
        assert "auroc/mean" in result.metrics
        assert run.attempt_metrics[0.01].threshold > 1.0
        assert run.attempt_metrics[0.01].fpr == 0.0

    @pytest.mark.asyncio
    @pytest.mark.parametrize("failure", ["prepare_data", "setup", "all_missing", "provider"])
    async def test_preflight_and_failure_cleanup(
        self,
        config: app_configs.DecisionModelEvalAppConfig,
        monkeypatch: pytest.MonkeyPatch,
        mocked_setup: unittest.mock.Mock,
        failure: str,
    ) -> None:
        """Release resources after partial dataset initialization, preflight, or provider failures."""
        client = unittest.mock.Mock(spec=helpers.FakeClient)
        client.predict.side_effect = RuntimeError("provider unavailable")
        factory = unittest.mock.Mock(return_value=client)
        monkeypatch.setattr(typesafe_adapter, "TypeSafeSystemOneClient", factory)
        runtime = unittest.mock.Mock(dry_run=False, wandb_run=None)
        if failure in {"prepare_data", "setup"}:
            monkeypatch.setattr(
                datamodule.CorrectnessDataModule,
                failure,
                unittest.mock.Mock(side_effect=ValueError("dataset initialization failed")),
            )
        original = datamodule.CorrectnessDataModule.get_records_for_subset
        if failure == "all_missing":

            def missing(
                self: datamodule.CorrectnessDataModule,
                subset_name: str,
            ) -> list[correctness_types.EvalRecord]:
                """Preserve dataset records while removing every extracted final answer."""
                records = original(self, subset_name)
                return [dataclasses.replace(record, final_answer=None) for record in records]

            monkeypatch.setattr(datamodule.CorrectnessDataModule, "get_records_for_subset", missing)
        teardown = unittest.mock.Mock()
        monkeypatch.setattr(datamodule.CorrectnessDataModule, "teardown", teardown)
        with pytest.raises((ValueError, RuntimeError)):
            await app.main(config, runtime)
        runtime.finalize.assert_called_once()
        teardown.assert_called_once()
        if failure != "provider":
            factory.assert_not_called()
        else:
            client.close.assert_called_once()
        assert not config.evals_config.result_dump_dir.exists()

    @pytest.mark.asyncio
    async def test_dry_run_without_key_or_dataset_access(
        self,
        config: app_configs.DecisionModelEvalAppConfig,
        monkeypatch: pytest.MonkeyPatch,
        mocked_setup: unittest.mock.Mock,
    ) -> None:
        """Skip dataset and client construction and W&B setup for a dry run."""
        monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
        forbidden = unittest.mock.Mock(side_effect=AssertionError("must not construct"))
        monkeypatch.setattr(typesafe_adapter, "TypeSafeSystemOneClient", forbidden)
        monkeypatch.setattr(datamodule_configs.CorrectnessDataModuleConfig, "instantiate_datamodule", forbidden)
        runtime = unittest.mock.Mock(dry_run=True)
        await app.main(config.model_copy(update={"use_wandb_logging": True}), runtime)
        assert mocked_setup.call_args.kwargs["use_wandb_logging"] is False
        forbidden.assert_not_called()
        runtime.finalize.assert_called_once()

    def test_hydra_composition_and_endpoint_override(
        self,
        config: app_configs.DecisionModelEvalAppConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Compose the Jev experiment with configurable provider endpoint and model ID."""
        monkeypatch.setattr(app_configs.pyine.utils.reprod, "load_dotenv", lambda: None)
        pyine.configs.base.register_searchpath_plugin()
        app_configs.register_hydra_configs(pyine.evals.common.EvalType.CORRECTNESS)
        with hydra.initialize(version_base=None, config_path=None):
            composed = hydra.compose(
                config_name="entrypoint",
                overrides=[
                    "+experiment=guardrail/decision_model_eval_jev",
                    f"config.evals_config.datamodule_config.lmdb_paths=[{config.evals_config.datamodule_config.lmdb_paths[0]}]",
                    f"config.evals_config.datamodule_config.split_config.split_source={config.evals_config.datamodule_config.split_config.split_source}",
                    "config.guardrail_config.provider.model=another-pinned-model",
                    "config.guardrail_config.provider.base_url=https://compatible.example",
                    "config.evals_config.result_dump_dir=null",
                ],
            )
            instantiated = hydra.utils.instantiate(composed.config)
            assert composed.runtime.run_name.startswith("another-pinned-model/binary_correctness_v1/")
        assert isinstance(instantiated, app_configs.DecisionModelEvalAppConfig)
        assert instantiated.guardrail_config.provider.model == "another-pinned-model"
        assert instantiated.guardrail_config.prompt_version == "binary_correctness_v1"
