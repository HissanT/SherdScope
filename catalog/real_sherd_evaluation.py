"""Shared loading and metric helpers for the 68-query real-sherd evaluation."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any


class RealSherdEvaluationError(ValueError):
    """Raised when a real-sherd evaluation directory is incomplete or malformed."""


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RealSherdEvaluationError(f"Could not read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RealSherdEvaluationError(f"Expected a JSON object in {path}")
    return value


def atomic_text(path: Path, value: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(value, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)


def load_evaluation(
    run_dir: Path,
    *,
    expected_count: int = 68,
    expected_pool: int = 400,
    expected_index: int = 2626,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load a complete batch in query-number order and verify every saved record."""
    run_dir = Path(run_dir).resolve()
    manifest = read_json(run_dir / "batch_manifest.json")
    queries = manifest.get("queries") or {}
    expected = [str(number) for number in range(1, expected_count + 1)]
    missing = [number for number in expected if number not in queries]
    incomplete = [
        number
        for number in expected
        if number in queries and queries[number].get("status") != "complete"
    ]
    if missing or incomplete:
        details = []
        if missing:
            details.append(f"missing queries: {', '.join(missing)}")
        if incomplete:
            details.append(f"incomplete queries: {', '.join(incomplete)}")
        raise RealSherdEvaluationError("Batch is not complete (" + "; ".join(details) + ")")

    records: list[dict[str, Any]] = []
    versions: set[str] = set()
    for number in range(1, expected_count + 1):
        entry = queries[str(number)]
        run_id = str(entry.get("run_id") or "")
        path = run_dir / "records" / f"{run_id}.json"
        if not run_id or not path.is_file():
            raise RealSherdEvaluationError(f"Query {number} is missing its exported record")
        record = read_json(path)
        run = record.get("run") or {}
        if str(record.get("run_id") or "") != run_id:
            raise RealSherdEvaluationError(f"Query {number} record/run ID mismatch")
        if str(run.get("query_id") or "") != str(entry.get("query_id") or ""):
            raise RealSherdEvaluationError(f"Query {number} record/query ID mismatch")
        results = [row for row in run.get("results") or [] if isinstance(row, dict)]
        if not results:
            raise RealSherdEvaluationError(f"Query {number} has no ranked results")
        version = str(run.get("algorithm_version") or "")
        if not version:
            raise RealSherdEvaluationError(f"Query {number} has no matcher version")
        versions.add(version)
        retrieval = run.get("retrieval") or {}
        if int(retrieval.get("kept_count") or 0) != expected_pool:
            raise RealSherdEvaluationError(
                f"Query {number} kept {retrieval.get('kept_count')} retrieval candidates; expected {expected_pool}"
            )
        if int(retrieval.get("input_count") or 0) != expected_index:
            raise RealSherdEvaluationError(
                f"Query {number} used an index of {retrieval.get('input_count')}; expected {expected_index}"
            )
        records.append({"number": number, "entry": entry, "record": record})
    if len(versions) != 1:
        raise RealSherdEvaluationError(
            "The batch contains multiple matcher versions: " + ", ".join(sorted(versions))
        )
    manifest = dict(manifest)
    manifest["algorithm_version"] = next(iter(versions))
    return manifest, records


def citation(row: dict[str, Any]) -> str:
    label = str(row.get("citation_label") or "").strip()
    if label:
        return label
    figure, item = str(row.get("figure") or ""), str(row.get("item") or "")
    if figure and item:
        return f"Figure {figure} Item {item}"
    return str(row.get("reference_id") or "—")


def runtime(run: dict[str, Any]) -> dict[str, float]:
    values = run.get("runtime") or {}
    stages = values.get("stage_seconds") or {}
    return {
        "retrieval": float(values.get("retrieval_seconds") or 0.0),
        "coarse": float(stages.get("coarse") or 0.0),
        "medium": float(stages.get("medium") or 0.0),
        "fine": float(stages.get("fine") or 0.0),
        "total": float(values.get("total_seconds") or 0.0),
    }


def score_summary(path: Path) -> dict[str, Any] | None:
    if not Path(path).is_file():
        return None
    value = read_json(path)
    scores = []
    for query in (value.get("queries") or {}).values():
        candidates = query.get("candidates") or []
        top = next((row for row in candidates if int(row.get("rank") or 0) == 1), None)
        if top is not None and top.get("score") in {0, 1, 2, 3}:
            scores.append(int(top["score"]))
    counts = {score: scores.count(score) for score in range(4)}
    return {"scores": scores, "counts": counts, "total": len(scores)}
