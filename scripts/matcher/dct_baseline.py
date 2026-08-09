"""Command-line entry point for the read-only DCT retrieval baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from catalog.dct_baseline import (
    DEFAULT_CONDITIONS,
    DCTConfig,
    run_manifest_batch_experiment,
    run_query_experiment,
    run_synthetic_experiment,
)
from catalog.matcher_benchmark import CONDITIONS


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--harmonics", type=int, default=20)
    parser.add_argument("--min-coverage", type=float, default=0.50)
    parser.add_argument("--max-coverage", type=float, default=1.00)
    parser.add_argument("--coverage-steps", type=int, default=11)
    parser.add_argument(
        "--exclude-dc",
        action="store_true",
        help="Discard the zero-frequency coefficient after rim normalization.",
    )


def _config(args: argparse.Namespace) -> DCTConfig:
    return DCTConfig(
        samples=args.samples,
        harmonics=args.harmonics,
        min_reference_coverage=args.min_coverage,
        max_reference_coverage=args.max_coverage,
        coverage_steps=args.coverage_steps,
        include_dc=not args.exclude_dc,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the paper-inspired SherdScope DCT shape baseline."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    score = subparsers.add_parser("score", help="Rank references for one query")
    score.add_argument("project", type=Path)
    source = score.add_mutually_exclusive_group(required=True)
    source.add_argument("--query-id")
    source.add_argument("--query-artifact", type=Path)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--top-k", type=int, default=20)
    score.add_argument("--overwrite-output", action="store_true")
    _add_config_arguments(score)

    synthetic = subparsers.add_parser(
        "synthetic", help="Run paired synthetic-fragment evaluation"
    )
    synthetic.add_argument("project", type=Path)
    synthetic.add_argument("--output", type=Path, required=True)
    synthetic.add_argument("--sample-size", type=int, default=300)
    synthetic.add_argument("--seed", type=int, default=20260801)
    synthetic.add_argument(
        "--conditions",
        nargs="+",
        choices=tuple(CONDITIONS),
        default=DEFAULT_CONDITIONS,
    )
    synthetic.add_argument("--overwrite-output", action="store_true")
    _add_config_arguments(synthetic)

    batch = subparsers.add_parser(
        "batch", help="Score known-parent queries from batch manifests"
    )
    batch.add_argument("project", type=Path)
    batch.add_argument(
        "--manifest",
        action="append",
        type=Path,
        required=True,
        help="Existing matcher batch_manifest.json; repeat for multiple cohorts.",
    )
    batch.add_argument("--output", type=Path, required=True)
    batch.add_argument("--workers", type=int, default=1)
    batch.add_argument("--top-k", type=int, default=20)
    batch.add_argument("--overwrite-output", action="store_true")
    _add_config_arguments(batch)

    args = parser.parse_args()
    config = _config(args)
    config.validate()
    if args.command == "score":
        report = run_query_experiment(
            args.project,
            args.output,
            query_id=args.query_id,
            query_artifact_path=args.query_artifact,
            config=config,
            top_k=args.top_k,
            overwrite=args.overwrite_output,
        )
        compact = {
            "algorithm_version": report["algorithm_version"],
            "reference_count": report["reference_count"],
            "elapsed_seconds": report["elapsed_seconds"],
            "results": report["results"],
        }
    elif args.command == "synthetic":
        report = run_synthetic_experiment(
            args.project,
            args.output,
            sample_size=args.sample_size,
            seed=args.seed,
            condition_names=tuple(args.conditions),
            config=config,
            overwrite=args.overwrite_output,
        )
        compact = {
            "algorithm_version": report["algorithm_version"],
            "reference_count": report["reference_count"],
            "selected_parent_count": report["selected_parent_count"],
            "query_count": report["query_count"],
            "elapsed_seconds": report["elapsed_seconds"],
            "summary": report["summary"],
        }
    else:
        report = run_manifest_batch_experiment(
            args.project,
            args.output,
            args.manifest,
            config=config,
            workers=args.workers,
            top_k=args.top_k,
            overwrite=args.overwrite_output,
            progress=lambda message: print(message, file=sys.stderr, flush=True),
        )
        compact = {
            "algorithm_version": report["algorithm_version"],
            "reference_count": report["reference_count"],
            "query_count": report["query_count"],
            "workers": report["workers"],
            "elapsed_seconds": report["elapsed_seconds"],
            "summary": report["summary"],
        }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
