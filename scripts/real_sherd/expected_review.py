"""Create an aligned top-two versus expected-answer review gallery."""
from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path

from catalog.contours import contour_root, read_manifest
from catalog.matcher import load_query, load_run, query_root, score_reference_by_citation

EXPECTED = {
    1: [("4.5", "14")],
    2: [("3.53", "4"), ("3.65", "26"), ("3.76", "2")],
    3: [("3.84", "2")],
    4: [("3.88", "4")],
    5: [("3.25", "25"), ("3.56", "9")],
}


def copy_asset(source: Path | None, destination: Path) -> str | None:
    if not source or not source.is_file():
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination.relative_to(destination.parents[1]).as_posix()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/real_expected_review"))
    parser.add_argument("--batch", type=Path, default=Path("outputs/real_sherd_match_results/batch_manifest.json"))
    args = parser.parse_args()
    project, output = args.project.resolve(), args.output.resolve()
    with open(args.batch.resolve(), encoding="utf-8") as handle:
        batch = json.load(handle)
    output.mkdir(parents=True, exist_ok=True)
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    contour_manifest = read_manifest(project)
    report: dict[str, object] = {"project": str(project), "queries": {}}
    cards: list[str] = []
    for number in range(1, 6):
        entry = (batch.get("queries") or {}).get(str(number)) or {}
        run_id = str(entry.get("run_id") or "")
        # A batch worker writes result.json before its final manifest/workbook
        # update.  That result is complete and safe to review, so accept it
        # while the manifest still says "running" (useful after a long run).
        run_path = project / "matcher" / "runs" / run_id
        if entry.get("status") != "complete" and not (run_path / "result.json").exists():
            raise SystemExit(f"Query {number} is not complete yet; wait for its rematch to finish.")
        run = load_run(project, run_id)
        query_id = str(run.get("query_id") or entry.get("query_id") or "")
        query = load_query(project, query_id)
        query_image = query_root(project) / query_id / "query.png"
        query_copy = assets / f"query_{number}.png"
        shutil.copy2(query_image, query_copy)
        top = sorted(run.get("results") or [], key=lambda x: float(x.get("overall_score", 1e9)))[:2]
        top_rows = []
        for rank, result in enumerate(top, 1):
            diagnostic = Path(project) / "matcher" / "runs" / run_id / str(result.get("diagnostic") or "")
            diagnostic_copy = assets / f"q{number}_top{rank}_aligned.png"
            diagnostic_rel = copy_asset(diagnostic, diagnostic_copy)
            top_rows.append({"rank": rank, "citation": result.get("citation_label"),
                             "score": result.get("overall_score"), "diagnostic": diagnostic_rel,
                             "warnings": result.get("warnings") or []})
        expected_rows = []
        for figure, item in EXPECTED[number]:
            try:
                forced = score_reference_by_citation(project, run_id, figure=figure, item=item)
                for position, result in enumerate(forced.get("results") or [], 1):
                    diagnostic = Path(project) / "matcher" / "runs" / run_id / str(result.get("diagnostic") or "")
                    diagnostic_copy = assets / f"q{number}_expected_{figure.replace('.', '_')}_{item}_{position}_aligned.png"
                    diagnostic_rel = copy_asset(diagnostic, diagnostic_copy)
                    match = next((v for v in contour_manifest.get("references", {}).values()
                                  if v.get("source_filename") == result.get("source_filename")), {})
                    clean = contour_root(project) / str(match.get("clean_mask") or "")
                    clean_copy = assets / f"q{number}_expected_{figure.replace('.', '_')}_{item}_{position}_reference.png"
                    clean_rel = copy_asset(clean, clean_copy)
                    expected_rows.append({"citation": f"Figure {figure} Item {item}",
                                          "score": result.get("overall_score"),
                                          "diagnostic": diagnostic_rel, "reference": clean_rel,
                                          "warnings": result.get("warnings") or []})
            except Exception as exc:
                expected_rows.append({"citation": f"Figure {figure} Item {item}", "error": str(exc)})
        report["queries"][str(number)] = {"run_id": run_id, "query": query.get("source_filename"),
                                           "query_image": f"assets/query_{number}.png",
                                           "top_two": top_rows, "expected": expected_rows}
        top_html = "".join(
            f"<article><h3>Top result {row['rank']}: {html.escape(str(row.get('citation')))}</h3>"
            f"<p>Score: {row.get('score')}</p>"
            f"<img src='{html.escape(str(row.get('diagnostic') or ''))}' alt='Top result aligned diagnostic'></article>"
            for row in top_rows
        )
        expected_html = "".join(
            f"<article class='expected'><h3>Expected: {html.escape(str(row['citation']))}</h3>"
            + (f"<p>Direct score: {row.get('score')}</p><div class='pair'><img src='{html.escape(str(row.get('reference') or ''))}' alt='Expected reference'><img src='{html.escape(str(row.get('diagnostic') or ''))}' alt='Expected aligned diagnostic'></div>"
               if row.get("diagnostic") else f"<p class='error'>Could not score: {html.escape(str(row.get('error')))}</p>")
            + "</article>"
            for row in expected_rows
        )
        cards.append(
            f"<section><h2>Query {number} — {html.escape(str(query.get('source_filename') or ''))}</h2>"
            f"<img class='query' src='assets/query_{number}.png' alt='Query mask'>"
            f"<h3>Matcher top two</h3><div class='grid'>{top_html}</div>"
            f"<h3>Expected answers</h3><div class='grid'>{expected_html}</div></section>"
        )
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Real sherd expected-answer review</title>"
        "<style>body{font:15px system-ui;margin:24px;background:#eef1f5;color:#172033}section{background:white;padding:18px;margin:0 0 22px;border-radius:12px}h2{border-bottom:1px solid #d8dee8;padding-bottom:8px}.query{width:180px;height:180px;object-fit:contain;background:#000;border:1px solid #ccd4df}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px}article{border:1px solid #d8dee8;border-radius:8px;padding:10px;background:#fafcff}article.expected{background:#f4fbf6;border-color:#9ad1a9}article img{width:100%;height:280px;object-fit:contain;background:#fff;border:1px solid #e1e6ee}.pair{display:grid;grid-template-columns:1fr 1fr;gap:8px}.pair img{height:220px}.error{color:#b42318}</style>"
        + "<body><h1>Real sherd: top two versus expected answers</h1>"
        + "<p>Shape-only direct scores. This temporary gallery does not alter the main workbook.</p>"
        + "".join(cards) + "</body>", encoding="utf-8"
    )
    print(f"Review page: {output / 'index.html'}")
    print(f"JSON diagnostics: {output / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
