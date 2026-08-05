"""Build EVAL_SCORES.md from the latest completed 30-query comparison run."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / "outputs" / "matcher_v15_final_balanced_metadata"
DEFAULT_OUTPUT = ROOT / "EVAL_SCORES.md"
EXCLUDED_QUERIES = {10: "No true parent exists in the reference catalogue."}


def read_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def is_parent(row: dict[str, Any], figure: str, item: str) -> bool:
    return str(row.get("figure") or "") == figure and str(row.get("item") or "") == item


def ranked_parent(
    rows: list[dict[str, Any]], figure: str, item: str, score_key: str
) -> tuple[int | None, dict[str, Any] | None]:
    ordered = sorted(
        rows,
        key=lambda row: (
            float(row.get(score_key, row.get("overall_score", float("inf")))),
            str(row.get("reference_id") or ""),
        ),
    )
    for rank, row in enumerate(ordered, start=1):
        if is_parent(row, figure, item):
            return rank, row
    return None, None


def stage_parent(
    run: dict[str, Any], stage_name: str, figure: str, item: str
) -> dict[str, Any] | None:
    stage = next(
        (value for value in run.get("stages", []) if value.get("name") == stage_name),
        {},
    )
    return next(
        (
            row
            for row in stage.get("candidates", [])
            if is_parent(row, figure, item)
        ),
        None,
    )


def citation(row: dict[str, Any] | None) -> str:
    if not row:
        return "—"
    figure, item = row.get("figure"), row.get("item")
    return f"{figure}.{item}" if figure and item else str(row.get("reference_id") or "—")


def number(value: Any, digits: int = 6) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def rank(value: int | None) -> str:
    return "NR" if value is None else str(value)


def flag(value: bool) -> str:
    return "Y" if value else "N"


def runtime(run: dict[str, Any]) -> dict[str, float]:
    values = run.get("runtime") or {}
    stages = values.get("stage_seconds") or {}
    return {
        "total": float(values.get("total_seconds") or 0.0),
        "retrieval": float(values.get("retrieval_seconds") or 0.0),
        "coarse": float(stages.get("coarse") or 0.0),
        "medium": float(stages.get("medium") or 0.0),
        "fine": float(stages.get("fine") or 0.0),
    }


def runtime_cell(values: dict[str, float]) -> str:
    return (
        f"{values['total']:.1f} "
        f"[{values['retrieval']:.1f}/{values['coarse']:.1f}/"
        f"{values['medium']:.1f}/{values['fine']:.1f}]"
    )


def collect(run_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = read_json(run_root / "batch_manifest.json")
    project = Path(manifest["project"])
    records = {}
    for path in (run_root / "records").glob("*.json"):
        record = read_json(path)
        records[str(record["run_id"])] = record

    output = []
    for query_number in range(1, 31):
        if query_number in EXCLUDED_QUERIES:
            continue
        entry = manifest["queries"][str(query_number)]
        figure, item = str(entry["figure"]), str(entry["item"])
        record = records[str(entry["run_id"])]
        shape_run = record["run"]
        metadata_run = read_json(
            project / "matcher" / "runs" / str(entry["metadata_run_id"]) / "result.json"
        )

        shape_rank, shape_parent = ranked_parent(
            shape_run.get("shape_candidate_pool", []), figure, item, "overall_score"
        )
        fused_rank, metadata_parent = ranked_parent(
            metadata_run.get("shape_candidate_pool", []), figure, item, "fused_score"
        )
        metadata_shape_rank, _ = ranked_parent(
            metadata_run.get("shape_candidate_pool", []), figure, item, "shape_score"
        )
        forced = ((record.get("forced_result") or {}).get("results") or [None])[0]
        parent_cost = (shape_parent or forced or {}).get("overall_score")
        shape_winner = shape_run["results"][0]
        metadata_winner = metadata_run["results"][0]
        coarse_parent = stage_parent(shape_run, "coarse", figure, item)
        retrieval = (coarse_parent or {}).get("retrieval") or (
            shape_run.get("retrieval", {}).get("target") or {}
        )
        shape_top_five = {
            str(row.get("reference_id") or "") for row in shape_run.get("results", [])
        }
        metadata_top_five = {
            str(row.get("reference_id") or "") for row in metadata_run.get("results", [])
        }
        shape_time, metadata_time = runtime(shape_run), runtime(metadata_run)
        output.append(
            {
                "query": query_number,
                "parent": f"{figure}.{item}",
                "shape_rank": shape_rank,
                "fused_rank": fused_rank,
                "shape_winner": citation(shape_winner),
                "metadata_winner": citation(metadata_winner),
                "parent_cost": parent_cost,
                "winner_cost": shape_winner.get("overall_score"),
                "gap": None
                if parent_cost is None
                else float(parent_cost) - float(shape_winner["overall_score"]),
                "retrieved": coarse_parent is not None,
                "retrieval_rank": retrieval.get("rank"),
                "outline_rank": retrieval.get("outline_rank"),
                "ribbon_rank": retrieval.get("ribbon_rank"),
                "fine_pool": shape_parent is not None,
                "metadata_shape_rank": metadata_shape_rank,
                "rank_change": None
                if metadata_shape_rank is None or fused_rank is None
                else metadata_shape_rank - fused_rank,
                "metadata_parent_shape_cost": None
                if metadata_parent is None
                else metadata_parent.get("shape_score"),
                "metadata_cost": None
                if metadata_parent is None
                else metadata_parent.get("metadata_score"),
                "fused_cost": None
                if metadata_parent is None
                else metadata_parent.get("fused_score"),
                "fields": None
                if metadata_parent is None
                else (metadata_parent.get("metadata") or {}).get("compared_fields"),
                "overlap": len(shape_top_five & metadata_top_five),
                "shape_time": shape_time,
                "metadata_time": metadata_time,
            }
        )
    return manifest, output


def accuracy(rows: list[dict[str, Any]], key: str, cutoff: int) -> tuple[int, float]:
    count = sum(row[key] is not None and row[key] <= cutoff for row in rows)
    return count, count / len(rows)


def mrr(rows: list[dict[str, Any]], key: str) -> float:
    return sum(0.0 if row[key] is None else 1.0 / row[key] for row in rows) / len(rows)


def build(run_root: Path = DEFAULT_RUN) -> str:
    manifest, rows = collect(run_root)
    shape_top1 = accuracy(rows, "shape_rank", 1)
    shape_top5 = accuracy(rows, "shape_rank", 5)
    shape_top10 = accuracy(rows, "shape_rank", 10)
    metadata_top1 = accuracy(rows, "fused_rank", 1)
    metadata_top5 = accuracy(rows, "fused_rank", 5)
    metadata_top10 = accuracy(rows, "fused_rank", 10)
    retrieval_recall = sum(row["retrieved"] for row in rows)
    shape_fine_recall = sum(row["fine_pool"] for row in rows)
    metadata_fine_recall = sum(row["fused_rank"] is not None for row in rows)
    mean_overlap = statistics.mean(row["overlap"] for row in rows)
    shape_seconds = sum(row["shape_time"]["total"] for row in rows)
    metadata_seconds = sum(row["metadata_time"]["total"] for row in rows)

    lines = [
        "# Evaluation scores",
        "",
        "## Run provenance",
        "",
        f"- Source: `{run_root.relative_to(ROOT).as_posix()}`",
        f"- Created: `{manifest.get('created_at')}`",
        f"- Shape matcher: `{manifest.get('algorithm_version')}`",
        f"- Metadata model: `{manifest.get('metadata_algorithm_version')}`",
        "- Evaluated queries: **29**. Query 10 is excluded because its stated true parent does not exist in the reference catalogue.",
        "- Stored retrieval cutoff: **300** candidates. These are the latest completed results; they predate the later code default of 400.",
        "- `NR` means the true parent did not reach that arm's fine-scored pool, so no final exact rank exists. A separately forced parent score is reported for cost diagnosis but is never converted into a fictional rank.",
        "- Runtime cells use `total [retrieval/coarse/medium/fine]` in seconds.",
        "",
        "## Aggregate results",
        "",
        "| Arm | MRR | Top-1 | Top-5 | Top-10 | Fine-pool parent recall |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Shape only | {mrr(rows, 'shape_rank'):.4f} | {shape_top1[0]}/29 ({shape_top1[1]:.1%}) | {shape_top5[0]}/29 ({shape_top5[1]:.1%}) | {shape_top10[0]}/29 ({shape_top10[1]:.1%}) | {shape_fine_recall}/29 ({shape_fine_recall/29:.1%}) |",
        f"| Shape + metadata | {mrr(rows, 'fused_rank'):.4f} | {metadata_top1[0]}/29 ({metadata_top1[1]:.1%}) | {metadata_top5[0]}/29 ({metadata_top5[1]:.1%}) | {metadata_top10[0]}/29 ({metadata_top10[1]:.1%}) | {metadata_fine_recall}/29 ({metadata_fine_recall/29:.1%}) |",
        "",
        f"The shape-only 300-candidate retrieval pool retained the true parent for **{retrieval_recall}/29 ({retrieval_recall/29:.1%})** queries. Mean Top-5 overlap between arms was **{mean_overlap:.2f}/5**. Shape-only runtime totaled **{shape_seconds/3600:.2f} h**; the independent metadata-aware arm totaled **{metadata_seconds/3600:.2f} h**.",
        "",
        "## A. Final ranking and shape-cost diagnosis",
        "",
        "| Query | True parent | Shape rank | S@1 | S@5 | S@10 | Fused rank | M@1 | M@5 | M@10 | Shape winner | Metadata winner | Parent cost | Winner cost | Gap |",
        "|---:|---|---:|:---:|:---:|:---:|---:|:---:|:---:|:---:|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| Q{row['query']} | {row['parent']} | {rank(row['shape_rank'])} | "
            f"{flag(row['shape_rank'] == 1)} | {flag(row['shape_rank'] is not None and row['shape_rank'] <= 5)} | "
            f"{flag(row['shape_rank'] is not None and row['shape_rank'] <= 10)} | {rank(row['fused_rank'])} | "
            f"{flag(row['fused_rank'] == 1)} | {flag(row['fused_rank'] is not None and row['fused_rank'] <= 5)} | "
            f"{flag(row['fused_rank'] is not None and row['fused_rank'] <= 10)} | {row['shape_winner']} | "
            f"{row['metadata_winner']} | {number(row['parent_cost'])} | {number(row['winner_cost'])} | {number(row['gap'])} |"
        )

    lines.extend(
        [
            "",
            "## B. Retrieval and cascade diagnosis (shape-only arm)",
            "",
            "| Query | In 300 pool? | Retrieval rank | Outline rank | Ribbon rank | In fine pool? |",
            "|---:|:---:|---:|---:|---:|:---:|",
        ]
    )
    for row in rows:
        retrieval_rank = (
            str(row["retrieval_rank"])
            if row["retrieval_rank"] is not None
            else ">300"
        )
        lines.append(
            f"| Q{row['query']} | {flag(row['retrieved'])} | {retrieval_rank} | "
            f"{row['outline_rank'] if row['outline_rank'] is not None else '—'} | "
            f"{row['ribbon_rank'] if row['ribbon_rank'] is not None else '—'} | {flag(row['fine_pool'])} |"
        )

    lines.extend(
        [
            "",
            "## C. Paired metadata effect",
            "",
            "`Metadata shape rank` and `fused rank` are calculated inside the same metadata-aware fine pool; positive change means metadata improved the parent's rank.",
            "",
            "| Query | Metadata shape rank | Fused rank | Change | Direction | Parent shape cost | Metadata cost | Fused cost | Fields | Top-5 overlap |",
            "|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        change = row["rank_change"]
        direction = "NR" if change is None else "improved" if change > 0 else "worsened" if change < 0 else "unchanged"
        change_label = "—" if change is None else f"{change:+d}"
        lines.append(
            f"| Q{row['query']} | {rank(row['metadata_shape_rank'])} | {rank(row['fused_rank'])} | "
            f"{change_label} | {direction} | {number(row['metadata_parent_shape_cost'])} | "
            f"{number(row['metadata_cost'])} | {number(row['fused_cost'])} | "
            f"{row['fields'] if row['fields'] is not None else '—'} | {row['overlap']}/5 |"
        )

    lines.extend(
        [
            "",
            "## D. Runtime by query",
            "",
            "| Query | Shape seconds: total [R/C/M/F] | Metadata seconds: total [R/C/M/F] | Combined total (s) |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        combined = row["shape_time"]["total"] + row["metadata_time"]["total"]
        lines.append(
            f"| Q{row['query']} | {runtime_cell(row['shape_time'])} | "
            f"{runtime_cell(row['metadata_time'])} | {combined:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Metric definitions",
            "",
            "- **Parent exact rank:** position of the true parent in the arm's final fine-scored pool. Missing parents are `NR` and contribute zero to MRR and Top-K.",
            "- **Parent/winner gap:** forced-or-in-pool true-parent shape cost minus the returned rank-1 shape cost. Zero means the parent won; a small positive value is a near miss.",
            "- **Retrieval rank:** combined cheap-retrieval rank when retained. `>300` means it was outside the stored pool; outline and ribbon ranks remain shown independently.",
            "- **Fine pool:** whether the parent survived retrieval, coarse scoring, and medium scoring to receive the final 96-sample score.",
            "- **Metadata rank change:** metadata-aware fine-pool shape rank minus fused rank. This isolates reranking direction without pretending the two independent arms shared an identical pool.",
            "- **Top-5 overlap:** number of reference IDs shared by the shape-only and metadata-aware Top 5.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    DEFAULT_OUTPUT.write_text(build(), encoding="utf-8")
    print(DEFAULT_OUTPUT)


if __name__ == "__main__":
    main()
