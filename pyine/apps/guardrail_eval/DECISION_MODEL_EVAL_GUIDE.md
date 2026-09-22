# Remote decision-model guardrail evaluation

`decision_model_eval` evaluates remote APIs that return a native probability of a
binary decision. The initial adapter uses the official `typesafe-sdk` to ask Jev
whether a predictor's final answer is correct. The existing correctness pipeline
handles splits, validation-derived thresholds, metrics, bootstrap intervals, and
result persistence.

## Run Jev

Install locked dependencies with `uv sync --extra dev` and set `TYPESAFE_API_KEY`
in the project `.env`. Only the environment variable's **name** appears in the
configuration; credentials are resolved when the adapter opens its client.

Provide predictor benchmark LMDB exports and their original problem split source:

```bash
uv run python -m pyine.apps.guardrail_eval.decision_model_eval \
  +experiment=guardrail/decision_model_eval_jev \
  'config.evals_config.datamodule_config.lmdb_paths=[/path/to/benchmark-export]' \
  config.evals_config.datamodule_config.split_config.split_source=TACO
```

The split source can also be a saved split file. Both fields are mandatory in
this overlay: there is no default private dataset directory. Add
`runtime.dry_run=true` to validate configuration without loading datasets, opening
an API client, or making W&B calls. A dry run still requires the two configuration
values, but they need not point to existing data.

The default model is pinned to `jev-1.13.0`, with four workers, five logical
requests/second, a one-request burst, 60-second HTTP timeouts, and at most three
SDK retries. Change these through `config.guardrail_config`. SDK-internal retries
are additional HTTP attempts and do not acquire the logical-request limiter.

Results are saved under the run's `benchmark_export/guardrail_test.pkl`. Load
trusted artifacts with `pyine.evals.persistence.load_eval_result` and use the
existing correctness analysis tools. No new result schema is required.

## Matched monitor comparison

Both models use the explicit `guardrail/correctness_judge:binary_correctness_v1`
version from [correctness_judge.yaml](../../prompts/templates/guardrail/correctness_judge.yaml).
It derives the role and Python execution task from the existing monitor template.
It distinguishes uncertainty about binary correctness from partial credit. Only
the response protocol differs: the LLM emits a JSON `score`, while the decision
API returns a Noul probability. Jev does not generate or parse a numeric answer.
The legacy `None`, `score_only`, and `with_reasoning` prompt behavior is preserved.
In particular, `None` still resolves the `score_only` text with the historical
reasoning-capable response schema. Use explicit versions for comparisons.

Compatible templates declare `binary_correctness: true` in their version metadata.
The default is `binary_correctness_v1`; a future binary version can be selected
through `config.guardrail_config.prompt_version` without editing the scorer or
adapter. Templates lacking the declaration are rejected before API calls. Native
instructions omit output-format text and strip trailing whitespace; the evidence
block stays identical to the monitor's. Both new overlays include the prompt
version in their run-directory names.

Run a fresh LLM comparison with the same LMDB/split arguments:

```bash
uv run python -m pyine.apps.guardrail_eval.prompted_llm_eval \
  +experiment=guardrail/prompted_llm_eval_binary_openai \
  'config.evals_config.datamodule_config.lmdb_paths=[/path/to/benchmark-export]' \
  config.evals_config.datamodule_config.split_config.split_source=TACO
```

Choose and record the LLM model through
`config.guardrail_config.llm_provider.model_kwargs.model`. Hold predictor attempts,
split seed, hard/soft label choice, calibration resampling, FPR targets, and missing
answer behavior fixed. Both overlays use the current monitor's
`skewed_moderate_bias` **subsampling** preset and evaluate `guardrail_test`.
Historical monitor scores used another rubric and are supplementary comparisons.

The input helper preserves the monitor's prompt-first precedence and message
flattening. Both arms see only the original prompt, full predictor response, and
extracted final answer. Reference answers, labels, IDs, and difficulty/category
metadata never enter the API payload. Debate keeps its existing renderer because
its role checks differ. Content-part values are stringified consistently in both
renderers; a missing or null `text` falls back to the part's dictionary representation.
This shared input behavior is recorded as `input_format_version=monitor_v2`.
Both scorers apply the missing-answer policy before rendering unused prompt evidence.

For the exact two-model execution setup, use
[`scripts/run_decision_model_eval_sweep.sh`](../../../scripts/run_decision_model_eval_sweep.sh):

```bash
bash scripts/run_decision_model_eval_sweep.sh --print-only
bash scripts/run_decision_model_eval_sweep.sh --dry-run
bash scripts/run_decision_model_eval_sweep.sh
```

This launcher uses the existing OpenAI sweep's `RL_HT_49/ckpt-model-org-exports`
dataset under `PYINE_DATA_ROOT` (or the repository's `data/` directory), with TACO
problem splits. Set both API keys in `.env`. It fixes shared soft-match labels,
split seed 42, calibration seed 0, bootstrap seed 0 with 1,000 replicates and four
bootstrap workers, four API workers, and five logical requests/second. GPT-5-mini
uses its default reasoning effort and a 10,000-token output limit. Jev's request
cache is disabled to match independent monitor calls. Results share the
`decision_model_comparison` experiment name, with separate model/version folders.
Append shared Hydra overrides after `--` to change the dataset or split; see
`--help` for an overview. The script stops if an app exits with an error.

Inspect monitor metadata for `llm_call_error`/`llm_format_error`; a matched run
should not include its infrastructure fallback scores as model judgments. If
preflight reports duplicated requests, set
`config.guardrail_config.cache_within_run=false` for a comparison with the current
monitor's independent per-row calls. With caching enabled, repeated requests reuse
one frozen decision and retain every row's statistical weight. Reuse extends
across calibration and evaluation calls within a scorer instance, never across
runs; no disk cache or resume protocol is implemented.

## Preflight and failure behavior

Before creating a client, the app checks calibration labels and every selected
population. Empty populations and populations where **every** `final_answer` is
`None` abort. A mixture of present/missing answers uses
`missing_answer_policy: score_zero`: missing answers make no request and receive
score/cost zero, with `decision_type=missing_answer` and no model probability.
An entirely unparsed export needs an explicit different input policy before use.

Authentication errors, exhausted retries/timeouts, wrong answer keys/types,
non-finite/out-of-range probabilities, missing required usage, and returned model
IDs different from the requested pinned ID abort evaluation. Errors do not
become 0.5 predictions. Clients, datamodules, and runtime resources are finalized
on success and failure. Pending queued requests are cancelled on failure; already
running calls finish before the client closes.

Context-limit errors also abort. The app does not truncate predictor evidence.
Preflight reports row/missing-answer counts, distinct requests, expected calls,
and maximum rendered character counts. Character counts are **not** model token
counts; the app cannot prove that all requests fit Jev's context in advance.

## Size and inspect an experiment

Start with a small development export containing both calibration classes. Use
reported token counts and latency to estimate the larger run. Freeze settings
before evaluating the held-out test split. At the currently documented input rate
of $0.042/million tokens, estimated spend is:

```text
expected_calls * mean_input_tokens * 0.042 / 1,000,000
```

The currently documented limits are 1,200 requests/minute, 250,000 tokens/second,
and 32k tokens for state plus the longest question. Recheck the
[provider's models/limits](https://docs.typesafe.ai/models) before a large run.
An average size below the context limit does not establish that the longest
request fits. Approximate throughput is bounded by the configured limiter,
workers divided by latency, and the provider's request/token rates. Allow headroom
for retries and long-tail requests.

Scores retain exact probabilities, including 0 and 1. Evaluation acceptance means
`score >= threshold`; FPR means accepting an incorrect predictor answer. Threshold
selection is not probability calibration. Brier/log-loss/reliability analysis can
be added offline later; no probability recalibration is applied here.

Per-attempt metadata stores the native probability, model ID, optional request
ID, input/output tokens, latency, request hash, and reuse status. Frozen prompt
text, requested/resolved version, template hash, SDK version, and execution policy
are stored in scorer metadata. The app also saves preflight counts in eval metadata.

`verification_costs` uses **input plus output tokens**, matching the monitor.
Repeated rows retain the original decision's token cost. This weighted cost array
is not the run's API bill. Scorer metadata separately counts completed API
requests and their reported input/output usage; failed retries can have unreported
usage. TypeSafe currently bills input only. Exclude deterministic missing-answer
outcomes when studying model probability calibration, and report their coverage.

## Other compatible models and testing

Override `config.guardrail_config.provider.base_url` (the API root, not the full
`/v1/systemone` endpoint), `model`, and `api_key_env` for another System One/Noul
endpoint. IDs must be pinned and match returned IDs. For a backend with no token
usage, explicitly set `require_token_usage=false`; verification costs and their
unit are then unknown, not zero. A different API schema needs a small adapter
implementing `DecisionModelClient`.

Offline tests use injected clients or the SDK's mock HTTP transport and temporary
LMDBs. They cover malformed responses, retries/timeouts, concurrency, input parity,
no reference leakage, resampling/order, missing answers, preflight, lifecycle,
Hydra composition, threshold ties, and persistence. Default tests never call a
live provider:

```bash
uv run pytest tests/guardrails/decision_model tests/apps/guardrail_eval \
  tests/guardrails/prompted_llm -q
```
