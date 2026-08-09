"""Run the five manually annotated real-sherd queries and export top-five Excel."""
from __future__ import annotations

import argparse
import hashlib
import json
import uuid
import os
from datetime import datetime, timezone
from pathlib import Path

from catalog.contours import _atomic_json
from catalog.matcher import MATCHER_ALGORITHM_VERSION, load_query, run_match
from catalog.contours import auto_query_wall_curves_from_fracture
from catalog.matcher import preprocess_query
from catalog.matcher_evaluation import MatcherEvaluationError, export_matcher_run_to_directory
from PIL import Image

# The workbook/query-set order follows the camera filenames exactly.
PILOT_IMAGE_BY_QUERY = {number: 6449 + number for number in range(1, 6)}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Match five real sherd queries and export top-five Excel.")
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--set-name", default="real_sherds_5")
    parser.add_argument("--query", type=int, help="Rematch only this query number")
    parser.add_argument("--refresh-from-pilot", action="store_true",
                        help="Rebuild the selected query from the latest manual pilot mask")
    parser.add_argument("--source-mask", type=Path,
                        help="Explicit mask/blob to use when refreshing one query")
    args = parser.parse_args()
    project = args.project.resolve()
    output = args.output.resolve()
    set_path = project / "matcher" / "query_sets" / f"{args.set_name}.json"
    with open(set_path, encoding="utf-8") as handle:
        query_set = json.load(handle)
    entries = query_set.get("queries") or {}
    queries = []
    for number in sorted(entries, key=lambda value: int(value)):
        entry = entries[number]
        query_id = str(entry.get("query_id") or "")
        artifact = project / "matcher" / "queries" / query_id / "artifact.json"
        if not query_id or not artifact.is_file():
            raise SystemExit(f"Missing saved query artifact for Query {number}")
        queries.append((int(number), query_id, sha256(artifact)))
    if len(queries) != 5:
        raise SystemExit(f"Expected five saved real-sherd queries, found {len(queries)}")
    if args.query is not None:
        if args.query not in range(1, 6):
            raise SystemExit("--query must be between 1 and 5")
        queries = [item for item in queries if item[0] == args.query]
        if args.refresh_from_pilot:
            number = args.query
            entry = entries[str(number)]
            old_query_id = str(entry.get("query_id") or "")
            old_artifact = project / "matcher" / "queries" / old_query_id / "artifact.json"
            fracture = entry.get("fracture") or []
            rim_point = entry.get("rim_point")
            if len(fracture) < 3 or not rim_point:
                raise SystemExit(f"Query {number} has no saved fracture/rim annotation")
            # Prefer the newest saved blob.  The annotation editor can save a
            # replacement under annotation_queries, while the pilot editor
            # saves the corresponding manual_gold_queries file.
            pilot_root = project.parents[1] / "outputs" / "real_sherd_pilot_5"
            candidates = [
                args.source_mask.resolve() if args.source_mask else None,
                pilot_root / "annotation_queries" / f"Query{number}.png",
                pilot_root / "annotation_queries" / f"query_{number}.png",
                pilot_root / "exports" / "manual_gold_queries" / f"IMG_{PILOT_IMAGE_BY_QUERY[number]}_manual.png",
                pilot_root / "gold" / "smooth_masks" / f"IMG_{PILOT_IMAGE_BY_QUERY[number]}.png",
            ]
            candidates = [path for path in candidates if path and path.is_file()]
            if not candidates:
                raise SystemExit(f"No saved Query {number} blob was found under {pilot_root}")
            pilot_mask = candidates[0] if args.source_mask else max(candidates, key=lambda path: path.stat().st_mtime)
            if (not args.source_mask and old_artifact.is_file()
                    and old_artifact.stat().st_mtime >= pilot_mask.stat().st_mtime):
                print(f"Query {number}: using the already-saved query artifact {old_query_id}")
                queries = [(number, old_query_id, sha256(old_artifact))]
                pilot_mask = None
            if pilot_mask is None:
                pass
            else:
                print(f"Query {number}: using saved blob {pilot_mask}")
            if pilot_mask is not None:
                with Image.open(pilot_mask) as source:
                    image_value = source.convert("RGBA")
                curves = auto_query_wall_curves_from_fracture(
                    image_value, {"fracture": fracture, "rim_point": rim_point}
                )
                refreshed = preprocess_query(
                    project, image_value, original_filename=f"Query{number}.png",
                    metadata={"query_id": f"Query {number}", "rim_diameter_cm": "", "fabric": "", "surface": "", "notes": ""},
                    manual_curves=curves,
                )
                entry["query_id"] = refreshed["query_id"]
                query_set["queries"][str(number)] = entry
                _atomic_json(set_path, query_set)
                for record_path in (output / "records").glob("*.json") if (output / "records").is_dir() else []:
                    try:
                        record = json.loads(record_path.read_text(encoding="utf-8"))
                        if (record.get("query") or {}).get("source_filename") == f"Query{number}.png":
                            record_path.unlink()
                    except (OSError, json.JSONDecodeError):
                        pass
                queries = [(number, refreshed["query_id"], sha256(project / "matcher" / "queries" / refreshed["query_id"] / "artifact.json"))]
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "batch_manifest.json"
    if manifest_path.is_file():
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
    else:
        manifest = {
            "schema_version": 1,
            "algorithm_version": MATCHER_ALGORITHM_VERSION,
            "query_set": args.set_name,
            "project": str(project),
            "queries": {},
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    for number, query_id, artifact_hash in queries:
        key = str(number)
        prior = manifest["queries"].get(key) or {}
        if args.refresh_from_pilot and args.query == number:
            prior = {}
        if prior.get("status") == "complete" and prior.get("artifact_sha256") == artifact_hash:
            print(f"Query {number}: already complete")
            continue
        print(f"Query {number}: matching")
        run_id = uuid.uuid4().hex
        manifest["queries"][key] = {
            "number": number, "query_id": query_id, "artifact_sha256": artifact_hash,
            "run_id": run_id, "status": "running",
        }
        _atomic_json(manifest_path, manifest)
        run = run_match(project, query_id, run_id=run_id)
        try:
            export = export_matcher_run_to_directory(project, run_id, output)
        except MatcherEvaluationError as exc:
            # Excel can lock the workbook while the matcher result itself is
            # already safely stored in records/.  Keep the run complete; a
            # fresh workbook can be rebuilt later without rerunning matching.
            print(f"Query {number}: result saved, workbook export deferred ({exc})")
            export = {"result_count": len(run.get("results") or [])}
        manifest["queries"][key].update({
            "status": "complete",
            "result_count": export.get("result_count"),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        })
        _atomic_json(manifest_path, manifest)
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(manifest_path, manifest)
    print(f"Done. Workbook: {output / 'SherdScope_matcher_evaluation.xlsx'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
