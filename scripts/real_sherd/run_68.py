"""Run/resume the 68 real sherds, then build all non-expert evaluation outputs."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from catalog.matcher_evaluation import export_matcher_run_to_directory
from catalog.real_sherd_evaluation import load_evaluation


def _run(*arguments: str) -> None:
    subprocess.run([sys.executable, *arguments], check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8, choices=range(1, 9))
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Explicitly allow reuse/update of an existing output directory.",
    )
    args = parser.parse_args()
    project, output = args.project.resolve(), args.output.resolve()
    if output.exists() and not args.resume:
        raise SystemExit(
            f"Refusing to write into existing output directory: {output}. "
            "Confirm its contents, then rerun with --resume."
        )

    _run(
        "-m",
        "scripts.matcher.query_batch",
        "--project",
        str(project),
        "--output",
        str(output),
        "--set-name",
        "real_sherds_68",
        "--expected-count",
        "68",
        "--workers",
        str(args.workers),
    )
    _manifest, records = load_evaluation(output)
    final_run_id = str(records[-1]["entry"]["run_id"])
    export = export_matcher_run_to_directory(project, final_run_id, output)
    if export.get("query_count") != 68:
        raise SystemExit("Final workbook does not contain all 68 queries")
    _run(
        "-m",
        "scripts.evaluation.build_real_sherd_eval_scores",
        "--run-dir",
        str(output),
    )
    _run(
        "-m",
        "scripts.evaluation.plot_real_sherd_eval",
        "--run-dir",
        str(output),
    )
    print("Matching and non-expert evaluation outputs are complete.", flush=True)
    print("Run score_68.py next to collect the 0–3 expert grades.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
