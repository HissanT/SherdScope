"""Apply metadata fusion to the five completed real-sherd shape runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from catalog.contours import _atomic_json
from catalog.matcher import load_run, run_root
from catalog.matcher_evaluation import export_matcher_run_to_directory
from catalog.metadata_fusion import VERSION as METADATA_VERSION
from catalog.metadata_fusion import fuse_shape_results, load_reference_metadata


def read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def observations(entry: dict) -> dict:
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
        description="Rerank the completed five real-sherd shape pools with metadata"
    )
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--shape-batch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    args = parser.parse_args()

    project = args.project.resolve()
    shape_batch = read_json(args.shape_batch.resolve())
    metadata_path = args.metadata.resolve() if args.metadata else (
        project / "matcher" / "query_sets" / "real_sherds_5_metadata.json"
    )
    metadata = read_json(metadata_path)
    saved_metadata = metadata.get("queries") or {}
    missing = [number for number in range(1, 6) if str(number) not in saved_metadata]
    if missing:
        raise SystemExit(f"Save the metadata form for every query first; missing: {missing}")

    references = load_reference_metadata(project)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for number in range(1, 6):
        batch_entry = (shape_batch.get("queries") or {}).get(str(number)) or {}
        if batch_entry.get("status") != "complete":
            raise SystemExit(f"Query {number} does not have a completed shape run")
        run_id = str(batch_entry.get("run_id") or "")
        run = load_run(project, run_id)
        pool = list(run.get("shape_candidate_pool") or [])
        if not pool:
            raise SystemExit(f"Query {number} has no saved shape candidate pool")
        fused = fuse_shape_results(pool, observations(saved_metadata[str(number)]), references)
        payload = {
            "schema_version": 1,
            "algorithm_version": METADATA_VERSION,
            "experiment_arm": "shape_plus_metadata",
            "candidate_source": "same_fine_scored_shape_pool",
            "metadata_retrieval_allowed": False,
            "run_id": run_id,
            "query_id": run.get("query_id"),
            "candidate_count": len(fused),
            "results": fused[:5],
        }
        _atomic_json(run_root(project) / run_id / "metadata_result.json", payload)
        export_matcher_run_to_directory(project, run_id, output)
        print(f"[{number}/5] Query {number}: metadata rerank saved", flush=True)
    print(f"Workbook: {output / 'SherdScope_matcher_evaluation.xlsx'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
