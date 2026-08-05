"""Command-line entry point for the SherdScope synthetic benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from catalog.matcher_benchmark import run_synthetic_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a read-only synthetic-fragment matcher benchmark."
    )
    parser.add_argument("project", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=90)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=("grid_clean", "clean", "light", "moderate"),
        default=("grid_clean", "clean", "light", "moderate"),
    )
    parser.add_argument(
        "--full-rerank-limit",
        type=int,
        default=0,
        help="Number of generated queries to send through the expensive reranker.",
    )
    args = parser.parse_args()
    report = run_synthetic_benchmark(
        args.project,
        args.output,
        sample_size=args.sample_size,
        seed=args.seed,
        condition_names=tuple(args.conditions),
        full_rerank_limit=args.full_rerank_limit,
    )
    print(json.dumps({key: report[key] for key in (
        "reference_count", "selected_parent_count", "query_count",
        "elapsed_seconds", "summary", "json_path", "csv_path"
    )}, indent=2))


if __name__ == "__main__":
    main()
