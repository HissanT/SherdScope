"""Command-line entry point for the saved 30-query matcher evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from catalog.matcher_batch import MatcherBatchError, run_saved_query_batch
from catalog.matcher_evaluation import WORKBOOK_FILENAME


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run all saved and annotated Hesban queries sequentially."
    )
    parser.add_argument("project", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=30)
    parser.add_argument(
        "--exclude-query",
        action="append",
        type=int,
        default=[],
        help="Query number to skip; repeat for multiple exclusions.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        help="Metadata manifest (defaults to project matcher/query_sets/hesban_30_metadata.json)",
    )
    args = parser.parse_args()
    try:
        run_saved_query_batch(
            args.project,
            args.output,
            expected_count=args.expected_count,
            metadata_path=args.metadata,
            excluded_queries=set(args.exclude_query),
        )
    except MatcherBatchError as exc:
        parser.error(str(exc))
    print(f"Done. Workbook: {(args.output / WORKBOOK_FILENAME).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
