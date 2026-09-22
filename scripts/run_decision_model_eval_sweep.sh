#!/usr/bin/env bash
# run the matched binary-correctness comparisons
#
# setup: uv sync --extra dev
# set TYPESAFE_API_KEY and OPENAI_API_KEY in the project .env; the apps load it
# (no GPU or local inference server is required)
#
# usage:
#   bash scripts/run_decision_model_eval_sweep.sh --print-only
#   bash scripts/run_decision_model_eval_sweep.sh --dry-run
#   bash scripts/run_decision_model_eval_sweep.sh
#   bash scripts/run_decision_model_eval_sweep.sh -- \
#     'config.evals_config.datamodule_config.lmdb_paths=[/path/to/benchmark-export]' \
#     config.evals_config.datamodule_config.split_config.split_source=/path/to/splits.pkl

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PRINT_ONLY=false
DRY_RUN=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --print-only) PRINT_ONLY=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help)
            cat <<'USAGE'
Usage: bash scripts/run_decision_model_eval_sweep.sh [--print-only] [--dry-run] [-- HYDRA_OVERRIDES...]

Runs jev-1.13.0 via decision_model_eval, then gpt-5-mini via prompted_llm_eval.
Both use binary_correctness_v1, soft-match labels, identical calibration and test
records, four API workers, and five logical requests/second. Jev caching is off
to match the monitor's independent per-row calls. GPT-5-mini uses its provider's
default reasoning effort and the overlay's 10,000-token output limit.

Defaults match the existing OpenAI sweep's dataset:
  ${PYINE_DATA_ROOT:-<repo>/data}/RL_HT_49/ckpt-model-org-exports
  TACO problem splits, guardrail_test evaluation, skewed_moderate_bias calibration

Set TYPESAFE_API_KEY and OPENAI_API_KEY in the project .env before a live run.
--print-only prints shell-escaped commands without starting either app.
--dry-run composes and validates both app configs without data loading or API calls.
Hydra overrides after -- are applied to both runs; use them for dataset/split changes.
Results use runtime.exp_name=decision_model_comparison with separate model/version
run folders. Each run writes benchmark_export/guardrail_test.pkl under its output directory.
USAGE
            exit 0
            ;;
        --) shift; break ;;
        *) break ;;
    esac
done
EXTRA_OVERRIDES=("$@")
SWEEP_TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"

SHARED_OVERRIDES=(
    "config.evals_config.datamodule_config.lmdb_paths=['\${oc.env:PYINE_DATA_ROOT,${REPO_ROOT}/data}/RL_HT_49/ckpt-model-org-exports']"
    config.evals_config.datamodule_config.split_config.split_source=TACO
    ++config.evals_config.datamodule_config.split_config.seed=42
    ++config.evals_config.datamodule_config.split_config.guardrail_valid_fraction=0.2
    config.evals_config.datamodule_config.label_type=soft_match
    config.evals_config.datamodule_config.resampling=null
    'config.evals_config.datamodule_config.eval_subset_names=[guardrail_test]'
    config/evals_config/calibration_resampling=skewed_moderate_bias
    config.evals_config.calibration_resampling.seed=0
    'config.evals_config.target_fpr_values=[0.001,0.01,0.05,0.1,0.2,0.5]'
    config.evals_config.num_bootstrap_replicates=1000
    config.evals_config.bootstrap_seed=0
    config.evals_config.bootstrap_num_workers=4
    config.guardrail_config.prompt_name=guardrail/correctness_judge
    config.guardrail_config.prompt_version=binary_correctness_v1
    config.guardrail_config.max_workers=4
    config.use_wandb_logging=false
    runtime.exp_name=decision_model_comparison
    runtime.seed=42
)

run_eval() {
    # print a reproducible command and execute one arm, stopping the sweep on failure
    local app="$1"
    local experiment="$2"
    local model="$3"
    shift 3
    local cmd=(
        uv run python -m "pyine.apps.guardrail_eval.${app}"
        "+experiment=guardrail/${experiment}"
        "${SHARED_OVERRIDES[@]}"
        "runtime.run_name=${SWEEP_TIMESTAMP}_${model}/\${config.guardrail_config.prompt_version}"
        "$@"
    )
    if [[ ${#EXTRA_OVERRIDES[@]} -gt 0 ]]; then
        cmd+=("${EXTRA_OVERRIDES[@]}")
    fi
    if [[ "${DRY_RUN}" == true ]]; then
        cmd+=(runtime.dry_run=true)
    fi
    printf '\nRunning %s:\n' "${model}"
    printf ' %q' "${cmd[@]}"
    printf '\n'
    if [[ "${PRINT_ONLY}" == false ]]; then
        "${cmd[@]}"
    fi
}

run_eval decision_model_eval decision_model_eval_jev jev-1.13.0 \
    config.guardrail_config.provider.model=jev-1.13.0 \
    config.guardrail_config.requests_per_second=5 \
    config.guardrail_config.cache_within_run=false \
    config.guardrail_config.missing_answer_policy=score_zero

run_eval prompted_llm_eval prompted_llm_eval_binary_openai gpt-5-mini \
    config.guardrail_config.llm_provider.model_kwargs.model=gpt-5-mini \
    config.guardrail_config.llm_provider.rate_limiter_config.requests_per_second=5 \
    config.guardrail_config.llm_provider.rate_limiter_config.max_bucket_size=1 \
    config.guardrail_config.default_score_on_missing_answer=0.0
