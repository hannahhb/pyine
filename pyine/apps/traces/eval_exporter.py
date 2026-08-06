"""Standalone CLI for the local PyINE evaluation-trace export.

The authoritative artifact contract, including schema, splitting, value encoding, outcome, and
certification semantics, is documented in ``pyine/data/traces/EVAL_EXPORT_SPEC.md``.

Example:

    ```bash
    python -m pyine.apps.traces.eval_exporter \
        --lmdb-pattern '10s10t-v1.*.lmdb' \
        --output-dir /path/to/pyine-v1-predictor-eval
    ```

Pass ``--lmdb-path`` repeatedly instead when the exact shard paths are already known.
"""

import pathlib

import click

import pyine.data.traces.dataset_utils
import pyine.data.traces.eval_export
import pyine.data.utils.splits
import pyine.utils.reprod


def _resolve_lmdb_paths(
    source_dataset_name: str,
    lmdb_paths: tuple[pathlib.Path, ...],
    lmdb_pattern: str | None,
) -> list[pathlib.Path]:
    """Resolve explicitly provided trace shards or a dataset-root pattern."""
    if lmdb_paths and lmdb_pattern is not None:
        raise click.UsageError("use either --lmdb-path or --lmdb-pattern, not both")
    if lmdb_paths:
        return list(lmdb_paths)
    if lmdb_pattern is None:
        raise click.UsageError("provide at least one --lmdb-path or an --lmdb-pattern")
    resolved_paths = pyine.data.traces.dataset_utils.get_matching_dataset_paths(
        source_dataset_name=source_dataset_name,
        pattern=lmdb_pattern,
    )
    if not resolved_paths:
        raise click.ClickException(f"no {source_dataset_name} trace dataset paths matched pattern: {lmdb_pattern!r}")
    return resolved_paths


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--lmdb-path",
    "lmdb_paths",
    type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path),
    multiple=True,
    help="Native trace LMDB shard path. Repeat for multiple shards.",
)
@click.option(
    "--lmdb-pattern",
    type=str,
    default=None,
    help="Pattern resolved under PyINE's trace data root for the configured source dataset.",
)
@click.option(
    "--source-dataset-name",
    type=click.Choice(pyine.data.traces.dataset_utils.SUPPORTED_SOURCE_DATASETS, case_sensitive=True),
    default="TACO",
    show_default=True,
    help="Source coding-problem dataset represented by the trace shards.",
)
@click.option(
    "--split-file",
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
    default=None,
    help="Original PyINE split file. Defaults to the source dataset's registered split.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=pathlib.Path),
    required=True,
    help="Local output directory for Parquet partitions, manifest, and certification log.",
)
@click.option("--seed", type=int, default=0, show_default=True, help="Deterministic cap and split seed.")
@click.option(
    "--train-fraction",
    type=click.FloatRange(0.0, 1.0),
    default=pyine.data.traces.eval_export.DEFAULT_TRAIN_FRACTION,
    show_default=True,
)
@click.option(
    "--validation-fraction",
    type=click.FloatRange(0.0, 1.0),
    default=pyine.data.traces.eval_export.DEFAULT_VALIDATION_FRACTION,
    show_default=True,
)
@click.option(
    "--test-fraction",
    type=click.FloatRange(0.0, 1.0),
    default=pyine.data.traces.eval_export.DEFAULT_TEST_FRACTION,
    show_default=True,
)
@click.option(
    "--max-problem-count",
    type=click.IntRange(min=1),
    default=None,
    help="Optional deterministic whole-problem cap. By default every allowed problem is exported.",
)
@click.option(
    "--reexecute-max-step-count",
    type=click.IntRange(min=0),
    default=None,
    help=(
        "Opt into re-execution up to this stored valid-step count. Omitted means integrity-only; "
        "the recommended exhaustive-dataset ceiling is "
        f"{pyine.data.traces.eval_export.DEFAULT_REEXECUTE_MAX_STEP_COUNT}."
    ),
)
@click.option(
    "--reexecute-all",
    is_flag=True,
    help="Re-execute every structurally valid trace, ignoring the stored step count.",
)
@click.option(
    "--recheck-timeout-seconds",
    type=click.FloatRange(min=0.0, min_open=True),
    default=pyine.data.traces.eval_export.DEFAULT_RECHECK_TIMEOUT_SECONDS,
    show_default=True,
    help="Timeout for each outcome-only certification re-execution.",
)
@click.option(
    "--execution-seed-override",
    type=int,
    default=None,
    help="Override the stored execution seed during re-execution.",
)
@click.option(
    "--certification-workers",
    type=click.IntRange(min=1),
    default=4,
    show_default=True,
    help="Concurrent certification workers.",
)
@click.option(
    "--parquet-batch-size",
    type=click.IntRange(min=1),
    default=256,
    show_default=True,
    help="Rows per Parquet row group and in-memory processing batch.",
)
@click.option(
    "--allow-partial-source",
    is_flag=True,
    help="Allow an intentionally incomplete set of conventionally numbered trace shards.",
)
def main(
    lmdb_paths: tuple[pathlib.Path, ...],
    lmdb_pattern: str | None,
    source_dataset_name: str,
    split_file: pathlib.Path | None,
    output_dir: pathlib.Path,
    seed: int,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    max_problem_count: int | None,
    reexecute_max_step_count: int | None,
    reexecute_all: bool,
    recheck_timeout_seconds: float,
    execution_seed_override: int | None,
    certification_workers: int,
    parquet_batch_size: int,
    allow_partial_source: bool,
) -> None:
    """Export original-train traces using the contract in ``pyine/data/traces/EVAL_EXPORT_SPEC.md``."""
    if reexecute_all and reexecute_max_step_count is not None:
        raise click.UsageError("use either --reexecute-all or --reexecute-max-step-count, not both")
    pyine.utils.reprod.entrypoint_setup()
    resolved_lmdb_paths = _resolve_lmdb_paths(source_dataset_name, lmdb_paths, lmdb_pattern)
    resolved_split_file = split_file or pyine.data.utils.splits.get_dataset_split_file_path(
        source_dataset_name,
        must_exist=True,
    )
    config = pyine.data.traces.eval_export.EvalExportConfig(
        source_lmdb_paths=resolved_lmdb_paths,
        split_file_path=resolved_split_file,
        output_dir=output_dir,
        source_dataset_name=source_dataset_name,
        seed=seed,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        max_problem_count=max_problem_count,
        reexecute_max_step_count=reexecute_max_step_count,
        reexecute_all=reexecute_all,
        recheck_timeout_seconds=recheck_timeout_seconds,
        execution_seed_override=execution_seed_override,
        certification_workers=certification_workers,
        parquet_batch_size=parquet_batch_size,
        allow_partial_source=allow_partial_source,
    )
    manifest = pyine.data.traces.eval_export.export_eval_traces(config)
    total_rows = sum(manifest.partition_row_counts.values())
    click.echo(f"Exported {total_rows} rows to {output_dir.resolve()}")
    click.echo(f"Problem counts: {manifest.partition_problem_counts}")
    click.echo(f"Integrity/re-execution counts: {manifest.certification_counts}")
    click.echo(f"Manifest: {(output_dir / 'export_manifest.json').resolve()}")


if __name__ == "__main__":
    main()
