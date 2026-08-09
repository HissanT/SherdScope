"""Run any saved matcher query set with resumable, visible progress."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from catalog.contours import _atomic_json
from catalog.matcher import (
    MATCHER_ALGORITHM_VERSION,
    MatcherError,
    run_match,
    score_reference_by_citation,
)
from catalog.matcher_evaluation import (
    MatcherEvaluationError,
    export_matcher_run_to_directory,
)
from catalog.metadata_fusion import load_reference_metadata


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not read {path}: {exc}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _human_observations(entry: dict[str, Any]) -> dict[str, Any]:
    values = dict(entry.get("metadata") or {})
    uncertainty = float(entry.get("diameter_uncertainty_cm") or 1.5)
    return {
        key: {
            "value": value,
            "source": "human",
            "reliability": 0.65,
            **({"uncertainty": uncertainty} if key == "rim_diameter_cm" else {}),
        }
        for key, value in values.items()
        if str(value).strip()
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Match a saved query set and export a resumable workbook."
    )
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--set-name", required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument(
        "--targets",
        type=Path,
        help="Optional JSON containing a queries object with figure/item gold citations.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        help="Run only a metadata-aware arm using this saved query-metadata manifest.",
    )
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 9))
    args = parser.parse_args()

    os.environ["SHERDSCOPE_MATCHER_WORKERS"] = str(args.workers)
    project = args.project.resolve()
    output = args.output.resolve()
    set_path = project / "matcher" / "query_sets" / f"{args.set_name}.json"
    query_set = _read_json(set_path)
    entries = query_set.get("queries") or {}
    numbers = sorted(int(value) for value in entries)
    expected = list(range(1, args.expected_count + 1))
    if numbers != expected:
        raise SystemExit(
            f"Expected Query 1-{args.expected_count} in {set_path}, found {numbers}"
        )

    targets = (_read_json(args.targets.resolve()).get("queries") or {}) if args.targets else {}
    if args.targets:
        missing = [number for number in expected if str(number) not in targets]
        if missing:
            raise SystemExit(f"Gold target file is missing queries: {missing}")

    metadata_path = args.metadata.resolve() if args.metadata else None
    metadata_queries = (_read_json(metadata_path).get("queries") or {}) if metadata_path else {}
    if metadata_path:
        missing = [number for number in expected if str(number) not in metadata_queries]
        if missing:
            raise SystemExit(f"Query metadata file is missing queries: {missing}")
    reference_metadata = load_reference_metadata(project) if metadata_path else None

    queries = []
    for number in numbers:
        entry = entries[str(number)]
        query_id = str(entry.get("query_id") or "")
        artifact = project / "matcher" / "queries" / query_id / "artifact.json"
        if not query_id or not artifact.is_file():
            raise SystemExit(f"Query {number} has no saved matcher artifact")
        target = targets.get(str(number)) or {}
        queries.append((number, query_id, _sha256(artifact), target))

    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "batch_manifest.json"
    manifest = _read_json(manifest_path) if manifest_path.is_file() else {
        "schema_version": 1,
        "algorithm_version": MATCHER_ALGORITHM_VERSION,
        "query_set": args.set_name,
        "arm": "shape_plus_metadata" if metadata_path else "shape_only",
        "project": str(project),
        "workers": args.workers,
        "metadata_manifest": str(metadata_path) if metadata_path else None,
        "metadata_manifest_sha256": _sha256(metadata_path) if metadata_path else None,
        "queries": {},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if manifest.get("algorithm_version") != MATCHER_ALGORITHM_VERSION:
        raise SystemExit("Output folder belongs to a different matcher version")
    if manifest.get("query_set") != args.set_name:
        raise SystemExit("Output folder belongs to a different query set")
    expected_arm = "shape_plus_metadata" if metadata_path else "shape_only"
    if manifest.get("arm", "shape_only") != expected_arm:
        raise SystemExit("Output folder belongs to a different evaluation arm")
    if metadata_path and manifest.get("metadata_manifest_sha256") != _sha256(metadata_path):
        raise SystemExit("Query metadata changed after this batch began; choose a new output folder")

    failures = []
    for position, (number, query_id, artifact_hash, target) in enumerate(queries, start=1):
        key = str(number)
        prior = (manifest.get("queries") or {}).get(key) or {}
        if prior.get("status") == "complete":
            if prior.get("artifact_sha256") != artifact_hash:
                raise SystemExit(f"Query {number} changed after this batch began")
            print(f"[{position}/{len(queries)}] Query {number}: already complete", flush=True)
            continue

        print(
            f"[{position}/{len(queries)}] Query {number}: matching with {args.workers} workers",
            flush=True,
        )
        run_id = uuid.uuid4().hex
        entry = {
            "number": number,
            "query_id": query_id,
            "artifact_sha256": artifact_hash,
            "run_id": run_id,
            "status": "running",
            **({"known_target": target} if target else {}),
        }
        manifest["queries"][key] = entry
        _atomic_json(manifest_path, manifest)

        last_progress = {"stage": None, "done": -1}

        def progress(stage: int, done: int, total: int, message: str) -> None:
            if stage != last_progress["stage"] or done == total or done - last_progress["done"] >= 25:
                print(f"    {message}: {done}/{total}", flush=True)
                last_progress.update(stage=stage, done=done)

        try:
            known_target = (
                {"figure": str(target["figure"]), "item": str(target["item"])}
                if target else None
            )
            run = run_match(
                project,
                query_id,
                run_id=run_id,
                progress=progress,
                known_target=known_target,
                query_metadata=(
                    _human_observations(metadata_queries[key])
                    if metadata_path else None
                ),
                reference_metadata=reference_metadata,
            )
            if known_target and not any(
                str(row.get("figure") or "") == known_target["figure"]
                and str(row.get("item") or "") == known_target["item"]
                for row in (run.get("results") or [])
            ):
                try:
                    score_reference_by_citation(
                        project,
                        run_id,
                        figure=known_target["figure"],
                        item=known_target["item"],
                    )
                    entry["known_target_status"] = "scored_separately"
                except MatcherError as exc:
                    entry["known_target_status"] = "unavailable"
                    entry["known_target_error"] = str(exc)
            elif known_target:
                entry["known_target_status"] = "in_top_five"
            try:
                export = export_matcher_run_to_directory(project, run_id, output)
                entry["result_count"] = export.get("result_count")
            except MatcherEvaluationError as exc:
                print(f"    Result saved; workbook export deferred: {exc}", flush=True)
                entry["result_count"] = len(run.get("results") or [])
            entry["status"] = "complete"
            entry["completed_at"] = datetime.now(timezone.utc).isoformat()
        except Exception as exc:
            entry["status"] = "failed"
            entry["error"] = f"{type(exc).__name__}: {exc}"
            failures.append(number)
            print(f"    FAILED: {entry['error']}", flush=True)
        _atomic_json(manifest_path, manifest)

    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(manifest_path, manifest)
    if failures:
        raise SystemExit(f"Failed queries (rerun the same command to retry): {failures}")
    print(f"Done. Workbook: {output / 'SherdScope_matcher_evaluation.xlsx'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
