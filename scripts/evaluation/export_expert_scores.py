"""Export real-sherd expert candidate scores to a flat CSV."""

from __future__ import annotations

import argparse
import csv
import os
import uuid
from pathlib import Path

from catalog.real_sherd_evaluation import load_evaluation, read_json


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / "outputs" / "real_sherd_68_2626_pool400_20260808"
FIELDS = [
    "query_number",
    "query_id",
    "run_id",
    "annotator",
    "no_acceptable_match",
    "query_note",
    "rank",
    "reference_id",
    "citation",
    "score",
    "candidate_note",
]


def export(run_dir: Path, scores_path: Path, output: Path) -> int:
    _manifest, records = load_evaluation(run_dir)
    scores = read_json(scores_path)
    score_queries = scores.get("queries") or {}
    rows = []
    for item in records:
        number = item["number"]
        query = score_queries.get(str(number))
        if not isinstance(query, dict):
            continue
        for candidate in query.get("candidates") or []:
            rows.append(
                {
                    "query_number": number,
                    "query_id": query.get("query_id"),
                    "run_id": query.get("run_id"),
                    "annotator": scores.get("annotator"),
                    "no_acceptable_match": bool(query.get("no_acceptable_match")),
                    "query_note": query.get("note", ""),
                    "rank": candidate.get("rank"),
                    "reference_id": candidate.get("reference_id"),
                    "citation": candidate.get("citation"),
                    "score": candidate.get("score"),
                    "candidate_note": candidate.get("note", ""),
                }
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--scores", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    scores = args.scores or args.run_dir / "expert_scores.json"
    output = args.output or args.run_dir / "expert_scores.csv"
    count = export(args.run_dir, scores, output)
    print(f"{output.resolve()} ({count} candidate rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
