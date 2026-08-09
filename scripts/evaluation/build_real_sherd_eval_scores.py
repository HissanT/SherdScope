"""Build the generated real-sherd matcher evaluation Markdown report."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from catalog.real_sherd_evaluation import (
    atomic_text,
    citation,
    load_evaluation,
    runtime,
    score_summary,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / "outputs" / "real_sherd_68_2626_pool400_20260808"


def _number(value: Any, digits: int = 6) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def _runtime_cell(value: dict[str, float]) -> str:
    return (
        f"{value['total']:.1f} [{value['retrieval']:.1f}/{value['coarse']:.1f}/"
        f"{value['medium']:.1f}/{value['fine']:.1f}]"
    )


def build(run_dir: Path) -> str:
    manifest, records = load_evaluation(run_dir)
    resolved_run = Path(run_dir).resolve()
    try:
        source_label = resolved_run.relative_to(ROOT).as_posix()
    except ValueError:
        source_label = str(resolved_run)
    rows = []
    first_retrieval = records[0]["record"]["run"].get("retrieval") or {}
    for item in records:
        run = item["record"]["run"]
        results = run["results"]
        winner = results[0]
        components = winner.get("score_components") or {}
        fine = next(
            (stage for stage in run.get("stages") or [] if stage.get("name") == "fine"),
            {},
        )
        survived = any(
            str(row.get("reference_id") or "") == str(winner.get("reference_id") or "")
            for row in fine.get("candidates") or []
        )
        retrieval = winner.get("retrieval") or {}
        warnings = winner.get("warnings") or []
        rows.append(
            {
                "number": item["number"],
                "citation": citation(winner),
                "cost": winner.get("overall_score"),
                "margin": run.get("confidence_margin"),
                "retrieval_rank": retrieval.get("rank"),
                "fine": survived,
                "components": components,
                "runtime": runtime(run),
                "warnings": warnings,
            }
        )

    lines = [
        "# Real-sherd evaluation scores",
        "",
        "## Run provenance",
        "",
        f"- Source: `{source_label}`",
        f"- Created: `{manifest.get('created_at')}`",
        f"- Matcher: `{manifest.get('algorithm_version')}`",
        f"- Evaluated real sherds: **{len(rows)}**.",
        f"- Reference index: **{first_retrieval.get('input_count')}**; retrieval pool: **{first_retrieval.get('kept_count')}** candidates.",
        "- These real sherds have no catalogued parent. Top-K recall and MRR are therefore not reported.",
        "- Runtime is `total [retrieval/coarse/medium/fine]` in seconds.",
        "",
        "## Expert 0–3 scoring",
        "",
    ]
    summary = score_summary(Path(run_dir) / "expert_scores.json")
    if not summary or not summary["total"]:
        lines.append("Not yet scored.")
    else:
        total = summary["total"]
        counts = summary["counts"]
        mean = sum(score * counts[score] for score in range(4)) / total
        at_least_two = counts[2] + counts[3]
        lines.extend(
            [
                f"Top-1 candidates scored: **{total}/{len(rows)}**.",
                "",
                "| Score | Meaning | Count | Percent |",
                "|---:|---|---:|---:|",
                f"| 0 | No match | {counts[0]} | {counts[0] / total:.1%} |",
                f"| 1 | Weak/family-only | {counts[1]} | {counts[1] / total:.1%} |",
                f"| 2 | Plausible parent | {counts[2]} | {counts[2] / total:.1%} |",
                f"| 3 | Near-exact | {counts[3]} | {counts[3] / total:.1%} |",
                "",
                f"- Mean top-1 expert score: **{mean:.3f}/3**.",
                f"- Top-1 scored ≥2: **{at_least_two}/{total} ({at_least_two / total:.1%})**.",
                f"- Top-1 scored =3: **{counts[3]}/{total} ({counts[3] / total:.1%})**.",
            ]
        )

    lines.extend(
        [
            "",
            "## Per-query top-1 results",
            "",
            "| Query | Top-1 citation | Match cost ↓ | First–second margin | Retrieval rank | Fine-pool survival | Runtime total [R/C/M/F] | Warnings |",
            "|---:|---|---:|---:|---:|:---:|---:|---|",
        ]
    )
    for row in rows:
        warning = "; ".join(str(value) for value in row["warnings"]) or "—"
        lines.append(
            f"| Q{row['number']} | {row['citation']} | {_number(row['cost'])} | "
            f"{_number(row['margin'])} | {row['retrieval_rank'] or '—'} | "
            f"{'Y' if row['fine'] else 'N'} | {_runtime_cell(row['runtime'])} | {warning} |"
        )

    keys = [
        ("fgw", "FGW"),
        ("ribbon", "Ribbon"),
        ("salience", "Salience"),
        ("alignment_tail", "Alignment tail"),
        ("rim_region", "Rim region"),
        ("transform_reliability", "Transform reliability"),
        ("completeness", "Completeness"),
    ]
    lines.extend(
        [
            "",
            "## Top-1 cost components",
            "",
            "| Query | " + " | ".join(label for _, label in keys) + " |",
            "|---:|" + "---:|" * len(keys),
        ]
    )
    for row in rows:
        lines.append(
            f"| Q{row['number']} | "
            + " | ".join(_number(row["components"].get(key)) for key, _ in keys)
            + " |"
        )
    lines.extend(
        [
            "",
            "## Metric definitions",
            "",
            "- **Expert score:** 0 no match; 1 weak/family-only; 2 plausible parent; 3 near-exact.",
            "- **Match cost:** the matcher distance for the returned top-1 candidate; lower is better.",
            "- **First–second margin:** rank-2 cost minus rank-1 cost. Larger positive margins indicate a clearer winner.",
            "- **Retrieval rank:** the top-1 candidate's rank in the stored 400-candidate cheap-retrieval pool.",
            "- **Fine-pool survival:** confirms the returned top-1 candidate appears in the saved fine-stage pool.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.run_dir / "REAL_SHERD_EVAL_SCORES.md"
    atomic_text(output, build(args.run_dir))
    print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
