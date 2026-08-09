"""Resumable sequential evaluation of the saved Hesban query set."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from catalog.contours import _atomic_json
from catalog.matcher import (
    MATCHER_ALGORITHM_VERSION,
    MatcherError,
    load_query,
    query_root,
    run_root,
    run_match,
    score_reference_by_citation,
)
from catalog.matcher_evaluation import (
    _read_json,
    evaluation_root,
    export_matcher_run_to_directory,
)
from catalog.metadata_fusion import VERSION as METADATA_VERSION
from catalog.metadata_fusion import fuse_shape_results, load_reference_metadata
from catalog.matcher_query_set import HESBAN_TARGETS


class MatcherBatchError(ValueError):
    pass


def _query_number(filename: str) -> int | None:
    match = re.search(r"query\s*0*(\d+)", filename, re.IGNORECASE)
    return int(match.group(1)) if match else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_reviewed_queries(project_path: Path) -> list[dict[str, Any]]:
    """Return the exact prepared queries represented by the current 30-run workbook."""
    query_set_path = query_root(project_path).parent / "query_sets" / "hesban_30.json"
    query_set = _read_json(query_set_path)
    if query_set:
        prepared = query_set.get("queries") or {}
        results = []
        for number in range(1, 31):
            saved = prepared.get(str(number)) or {}
            query_id = str(saved.get("query_id") or "")
            if not query_id:
                continue
            artifact_path = query_root(project_path) / query_id / "artifact.json"
            artifact = load_query(project_path, query_id)
            results.append({
                "number": number, "label": f"Query {number}", "query_id": query_id,
                "source_filename": str(artifact.get("source_filename") or saved.get("filename") or ""),
                "figure": HESBAN_TARGETS[number - 1][0], "item": HESBAN_TARGETS[number - 1][1],
                "artifact_sha256": _sha256(artifact_path),
            })
        return results
    records_root = evaluation_root(project_path) / "records"
    records = []
    for path in sorted(records_root.glob("*.json")):
        record = _read_json(path)
        if record:
            records.append(record)
    by_number: dict[int, dict[str, Any]] = {}
    for record in records:
        query = record.get("query") or {}
        filename = str(query.get("source_filename") or "")
        number = _query_number(filename)
        query_id = str(query.get("query_id") or "")
        if number is None or not query_id:
            continue
        artifact_path = query_root(project_path) / query_id / "artifact.json"
        if not artifact_path.is_file():
            raise MatcherBatchError(f"Saved artifact is missing for {filename}")
        artifact = load_query(project_path, query_id)
        boundary = artifact.get("query_master_boundary") or {}
        fracture = (artifact.get("curves") or {}).get("fracture") or []
        if len(boundary.get("points") or []) < 4 or len(fracture) < 2:
            raise MatcherBatchError(
                f"{filename} does not have a saved rim point and fracture trace"
            )
        candidate = {
            "number": number,
            "label": f"Query {number}",
            "query_id": query_id,
            "source_filename": filename,
            "figure": HESBAN_TARGETS[number - 1][0] if 1 <= number <= 30 else None,
            "item": HESBAN_TARGETS[number - 1][1] if 1 <= number <= 30 else None,
            "artifact_sha256": _sha256(artifact_path),
        }
        prior = by_number.get(number)
        if prior and prior["query_id"] != query_id:
            raise MatcherBatchError(
                f"The evaluation set contains two different saved versions of Query {number}"
            )
        by_number[number] = candidate
    return [by_number[number] for number in sorted(by_number)]


def _target_in_top_five(run: dict[str, Any], figure: str, item: str) -> bool:
    return any(
        str(result.get("figure") or "") == figure
        and str(result.get("item") or "") == item
        for result in (run.get("results") or [])
    )


def _load_query_metadata(project_path: Path, metadata_path: Path | None) -> tuple[dict[str, Any], Path]:
    path = metadata_path or (
        query_root(project_path).parent / "query_sets" / "hesban_30_metadata.json"
    )
    value = _read_json(Path(path))
    if not value:
        raise MatcherBatchError(
            "Query metadata has not been saved. Run "
            "python -m scripts.matcher.metadata first."
        )
    return value, Path(path)


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


def _save_metadata_rerank(
    project_path: Path,
    run: dict[str, Any],
    query: dict[str, Any],
    metadata_entry: dict[str, Any],
    reference_metadata: dict[str, dict[str, str]],
) -> dict[str, Any]:
    pool = list(run.get("shape_candidate_pool") or [])
    if not pool:
        raise MatcherBatchError("Matcher run did not preserve a shape candidate pool")
    fused = pool if run.get("metadata_used") else fuse_shape_results(
        pool, _human_observations(metadata_entry), reference_metadata
    )
    top_five = fused[:5]
    target = next(
        (
            row for row in fused
            if str(row.get("figure") or "") == str(query.get("figure") or "")
            and str(row.get("item") or "") == str(query.get("item") or "")
        ),
        None,
    )
    payload = {
        "schema_version": 1,
        "algorithm_version": METADATA_VERSION,
        "experiment_arm": "shape_plus_metadata",
        "candidate_source": "metadata_aware_full_pipeline",
        "metadata_retrieval_allowed": True,
        "run_id": run.get("run_id"),
        "query_id": run.get("query_id"),
        "candidate_count": len(fused),
        "results": top_five,
        "known_target": {
            "figure": query.get("figure"),
            "item": query.get("item"),
            "rank": target.get("fused_rank") if target else None,
            "in_candidate_pool": target is not None,
            "result": target,
        },
    }
    _atomic_json(
        run_root(project_path) / str(run.get("run_id")) / "metadata_result.json",
        payload,
    )
    return payload


def run_saved_query_batch(
    project_path: Path,
    output_root: Path,
    *,
    expected_count: int = 30,
    metadata_path: Path | None = None,
    excluded_queries: set[int] | None = None,
) -> dict[str, Any]:
    project_path, output_root = Path(project_path), Path(output_root)
    queries = discover_reviewed_queries(project_path)
    metadata_manifest, resolved_metadata_path = _load_query_metadata(
        project_path, metadata_path
    )
    metadata_queries = metadata_manifest.get("queries") or {}
    if len(queries) != expected_count or [q["number"] for q in queries] != list(
        range(1, expected_count + 1)
    ):
        raise MatcherBatchError(
            f"Expected saved Queries 1-{expected_count}, but found "
            f"{[q['number'] for q in queries]}"
        )
    excluded = {int(number) for number in (excluded_queries or set())}
    invalid_exclusions = sorted(
        number for number in excluded if number < 1 or number > expected_count
    )
    if invalid_exclusions:
        raise MatcherBatchError(
            f"Excluded query numbers are outside 1-{expected_count}: "
            f"{invalid_exclusions}"
        )
    active_queries = [query for query in queries if query["number"] not in excluded]
    missing_metadata = [number for number in range(1, expected_count + 1) if str(number) not in metadata_queries]
    if missing_metadata:
        raise MatcherBatchError(
            "Save or explicitly leave blank the metadata form for every query; "
            f"missing: {missing_metadata}"
        )
    reference_metadata = load_reference_metadata(project_path)
    metadata_sha256 = _sha256(resolved_metadata_path)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "batch_manifest.json"
    manifest = _read_json(manifest_path) or {
        "schema_version": 1,
        "algorithm_version": MATCHER_ALGORITHM_VERSION,
        "project": str(project_path.resolve()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metadata_algorithm_version": METADATA_VERSION,
        "metadata_manifest_sha256": metadata_sha256,
        "excluded_queries": sorted(excluded),
        "queries": {},
    }
    if manifest.get("algorithm_version") != MATCHER_ALGORITHM_VERSION:
        raise MatcherBatchError(
            "This output folder belongs to a different matcher version; choose a new output folder."
        )
    if manifest.get("metadata_manifest_sha256") != metadata_sha256:
        raise MatcherBatchError(
            "The query metadata changed after this batch began; choose a new output folder."
        )
    if sorted(manifest.get("excluded_queries") or []) != sorted(excluded):
        raise MatcherBatchError(
            "This output folder was started with a different query exclusion set; "
            "choose a new output folder."
        )
    failures = []
    active_count = len(active_queries)
    for position, query in enumerate(active_queries, start=1):
        key = str(query["number"])
        saved = (manifest.get("queries") or {}).get(key) or {}
        if saved.get("status") == "complete":
            if saved.get("artifact_sha256") != query["artifact_sha256"]:
                raise MatcherBatchError(f"{query['label']} was edited after this batch began")
            print(f"[{position}/{active_count}] {query['label']}: already complete")
            continue
        print(f"[{position}/{active_count}] {query['label']}: matching {query['source_filename']}")
        run_id = uuid.uuid4().hex
        entry = {**query, "run_id": run_id, "status": "running"}
        manifest["queries"][key] = entry
        _atomic_json(manifest_path, manifest)
        try:
            run = run_match(
                project_path,
                query["query_id"],
                run_id=run_id,
                known_target={"figure": query["figure"], "item": query["item"]},
            )
            metadata_run_id = uuid.uuid4().hex
            metadata_run = run_match(
                project_path,
                query["query_id"],
                run_id=metadata_run_id,
                known_target={"figure": query["figure"], "item": query["item"]},
                query_metadata=_human_observations(metadata_queries[key]),
                reference_metadata=reference_metadata,
            )
            entry["metadata_run_id"] = metadata_run_id
            metadata_result = _save_metadata_rerank(
                project_path,
                metadata_run,
                query,
                metadata_queries[key],
                reference_metadata,
            )
            # Keep the comparison workbook contract: the metadata result lives
            # beside the corresponding shape-only run.
            for item in metadata_result.get("results") or []:
                diagnostic = str(item.get("diagnostic") or "")
                source = run_root(project_path) / metadata_run_id / diagnostic
                if diagnostic and source.is_file():
                    copied_name = f"metadata_{diagnostic}"
                    shutil.copy2(
                        source,
                        run_root(project_path) / str(run.get("run_id")) / copied_name,
                    )
                    item["diagnostic"] = copied_name
            metadata_result["run_id"] = run.get("run_id")
            metadata_result["metadata_pipeline_run_id"] = metadata_run_id
            _atomic_json(
                run_root(project_path) / str(run.get("run_id")) / "metadata_result.json",
                metadata_result,
            )
            entry["shape_plus_metadata_target_rank"] = (
                metadata_result.get("known_target") or {}
            ).get("rank")
            if _target_in_top_five(run, query["figure"], query["item"]):
                entry["known_target"] = "in_top_five"
            else:
                try:
                    score_reference_by_citation(
                        project_path,
                        run_id,
                        figure=query["figure"],
                        item=query["item"],
                    )
                    entry["known_target"] = "scored_separately"
                except MatcherError as exc:
                    entry["known_target"] = "unavailable"
                    entry["known_target_error"] = str(exc)
            export_matcher_run_to_directory(project_path, run_id, output_root)
            entry["status"] = "complete"
            entry["completed_at"] = datetime.now(timezone.utc).isoformat()
        except Exception as exc:
            entry["status"] = "failed"
            entry["error"] = str(exc)
            failures.append(query["label"])
        _atomic_json(manifest_path, manifest)
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(manifest_path, manifest)
    if failures:
        raise MatcherBatchError(
            "Batch finished with failures (rerun the same command to retry): "
            + ", ".join(failures)
        )
    return manifest
