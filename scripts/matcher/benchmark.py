"""Command-line entry point for the SherdScope synthetic benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from catalog.matcher_benchmark import CONDITIONS, run_synthetic_benchmark


DEFAULT_CONDITIONS = (
    "grid_clean", "clean", "light", "moderate",
    "partial_75", "partial_50", "partial_25",
)


def comparison_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("comparison must be LABEL=PATH")
    label, raw_path = value.split("=", 1)
    if not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("comparison must be LABEL=PATH")
    return label.strip(), Path(raw_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a read-only synthetic-fragment matcher benchmark."
    )
    parser.add_argument("project", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=tuple(CONDITIONS),
        default=DEFAULT_CONDITIONS,
    )
    parser.add_argument(
        "--full-rerank-limit",
        type=int,
        default=0,
        help="Number of generated queries to send through the expensive reranker.",
    )
    parser.add_argument(
        "--comparison",
        action="append",
        type=comparison_arg,
        default=[],
        metavar="LABEL=PATH",
        help="Saved benchmark JSON to include in scaling_comparison.csv.",
    )
    parser.add_argument(
        "--require-existing-index",
        action="store_true",
        help="Fail instead of creating/replacing a missing or stale shared retrieval index.",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Allow replacement of benchmark files in the selected output directory.",
    )
    args = parser.parse_args()
    report = run_synthetic_benchmark(
        args.project,
        args.output,
        sample_size=args.sample_size,
        seed=args.seed,
        condition_names=tuple(args.conditions),
        full_rerank_limit=args.full_rerank_limit,
        require_existing_index=args.require_existing_index,
        allow_output_overwrite=args.overwrite_output,
        comparison_report_paths=dict(args.comparison),
    )
    print(json.dumps({key: report[key] for key in (
        "reference_count", "selected_parent_count", "query_count",
        "elapsed_seconds", "summary", "artifacts", "json_path", "csv_path"
    )}, indent=2))


if __name__ == "__main__":
    main()
