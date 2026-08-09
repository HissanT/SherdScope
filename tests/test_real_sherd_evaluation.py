import csv
import json
from pathlib import Path

from PIL import Image

from catalog.real_sherd_evaluation import load_evaluation
from catalog.real_sherd_scoring import create_scoring_app
from scripts.evaluation.build_real_sherd_eval_scores import build
from scripts.evaluation.export_expert_scores import export
from scripts.evaluation.plot_real_sherd_eval import plot


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    research = tmp_path / "Research"
    project = research / "SherdScope" / "projects" / "project"
    run_dir = research / "SherdScope" / "outputs" / "real"
    queries, manifest_queries = {}, {}
    for number in range(1, 69):
        query_id, run_id = f"query-{number}", f"run-{number}"
        filename = f"IMG_{number:04d}.png"
        queries[str(number)] = {
            "number": number,
            "filename": filename,
            "query_id": query_id,
            **({"horizontal_flip": True} if number == 5 else {}),
        }
        manifest_queries[str(number)] = {
            "number": number,
            "query_id": query_id,
            "run_id": run_id,
            "status": "complete",
            "result_count": 5,
            "completed_at": "2026-08-08T00:00:00+00:00",
        }
        diagnostic = project / "matcher" / "runs" / run_id / "diagnostic_1.png"
        query_image = project / "matcher" / "queries" / query_id / "preview.png"
        photo = research / "DL ArchProject" / "dataset_clean" / "train" / "images" / filename
        for path in (diagnostic, query_image, photo):
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (12, 10), (number, 30, 60)).save(path)
        candidate = {
            "rank": 1,
            "reference_id": f"reference-{number}",
            "citation_label": f"Figure X Item {number}",
            "overall_score": 0.1 + number / 1000,
            "diagnostic": "diagnostic_1.png",
            "retrieval": {"rank": number},
            "score_components": {
                "fgw": 0.01,
                "ribbon": 0.02,
                "salience": 0.03,
                "alignment_tail": 0.04,
                "rim_region": 0.05,
                "transform_reliability": 0.06,
                "completeness": 0.07,
            },
            "warnings": [],
        }
        record = {
            "schema_version": 1,
            "run_id": run_id,
            "query": {"query_id": query_id, "source_filename": filename, "metadata": {}},
            "run": {
                "run_id": run_id,
                "query_id": query_id,
                "algorithm_version": "matcher-version-from-run",
                "retrieval": {"input_count": 2626, "kept_count": 400},
                "confidence_margin": 0.01,
                "results": [candidate],
                "stages": [{"name": "fine", "candidates": [candidate]}],
                "runtime": {
                    "retrieval_seconds": 1.0,
                    "stage_seconds": {"coarse": 2.0, "medium": 3.0, "fine": 4.0},
                    "total_seconds": 10.0,
                },
            },
        }
        _write(run_dir / "records" / f"{run_id}.json", record)
    _write(project / "matcher" / "query_sets" / "real_sherds_68.json", {"queries": queries})
    _write(
        run_dir / "batch_manifest.json",
        {
            "schema_version": 1,
            "query_set": "real_sherds_68",
            "project": str(project),
            "queries": manifest_queries,
            "created_at": "2026-08-08T00:00:00+00:00",
        },
    )
    return project, run_dir


def test_loads_matcher_version_from_records(tmp_path):
    _project, run_dir = _fixture(tmp_path)
    manifest, records = load_evaluation(run_dir)

    assert manifest["algorithm_version"] == "matcher-version-from-run"
    assert len(records) == 68


def test_scoring_autosave_and_resume(tmp_path):
    project, run_dir = _fixture(tmp_path)
    output = run_dir / "expert_scores.json"
    app = create_scoring_app(project, run_dir, output, annotator="Expert A")
    client = app.test_client()
    queue = client.get("/api/queue").get_json()["queries"]
    candidate = queue[0]["candidates"][0]
    response = client.post(
        "/api/save",
        json={
            "number": 1,
            "no_acceptable_match": False,
            "note": "query note",
            "candidates": [
                {
                    "rank": candidate["rank"],
                    "reference_id": candidate["reference_id"],
                    "citation": candidate["citation"],
                    "score": 3,
                    "note": "candidate note",
                }
            ],
        },
    )

    assert response.get_json()["success"] is True
    resumed = create_scoring_app(project, run_dir, output, annotator="Expert A")
    saved = resumed.test_client().get("/api/queue").get_json()["queries"][0]
    assert saved["candidates"][0]["score"] == 3
    assert saved["note"] == "query note"
    assert saved["candidates"][0]["note"] == "candidate note"
    assert resumed.test_client().get("/asset/query/5/photo").status_code == 200


def test_generated_report_csv_and_plots(tmp_path):
    project, run_dir = _fixture(tmp_path)
    output = run_dir / "expert_scores.json"
    app = create_scoring_app(project, run_dir, output, annotator="Expert A")
    first = app.test_client().get("/api/queue").get_json()["queries"][0]["candidates"][0]
    app.test_client().post(
        "/api/save",
        json={
            "number": 1,
            "no_acceptable_match": False,
            "note": "",
            "candidates": [{**first, "score": 2, "note": ""}],
        },
    )

    report = build(run_dir)
    assert "matcher-version-from-run" in report
    assert "Top-1 candidates scored: **1/68**" in report
    assert "Figure X Item 1" in report

    csv_path = run_dir / "expert_scores.csv"
    assert export(run_dir, output, csv_path) == 68
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["score"] == "2"

    names = plot(run_dir, run_dir / "figures")
    assert len(names) == 6
    assert (run_dir / "figures" / "cost_vs_score_scatter.png").is_file()
    assert (run_dir / "figures" / "runtime_stacked_bar.svg").is_file()
