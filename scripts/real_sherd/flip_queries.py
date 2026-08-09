"""Mirror selected saved real-sherd query masks and their final smoothed geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from catalog.contours import _atomic_json, matcher_root
from catalog.matcher import preprocess_query


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _mirror_points(points: Any, width: int) -> list[list[float]]:
    values = np.asarray(points, dtype=float).copy()
    if values.ndim != 2 or values.shape[1] != 2 or not len(values):
        raise ValueError("Saved curve is missing or malformed")
    values[:, 0] = float(width - 1) - values[:, 0]
    return values.tolist()


def _source_wall(artifact: dict[str, Any], name: str) -> list[list[float]]:
    normalization = artifact.get("normalization") or {}
    centroid = np.asarray(normalization.get("centroid") or [], dtype=float)
    scale = float(normalization.get("scale") or 0.0)
    points = np.asarray((artifact.get("curves") or {}).get(name) or [], dtype=float)
    if centroid.shape != (2,) or scale <= 0 or points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Saved {name} curve cannot be restored to source coordinates")
    return (points * scale + centroid).tolist()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--set-name", default="real_sherds_68")
    parser.add_argument("--query", type=int, action="append", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    project = args.project.resolve()
    masks_root = args.masks.resolve()
    set_path = matcher_root(project) / "query_sets" / f"{args.set_name}.json"
    value = _read_json(set_path)
    entries = value.get("queries") or {}
    selected = sorted(set(args.query))
    invalid = [number for number in selected if str(number) not in entries]
    if invalid:
        raise SystemExit(f"Queries are not saved in {args.set_name}: {invalid}")

    for number in selected:
        entry = entries[str(number)]
        if entry.get("horizontal_flip"):
            print(f"Query {number}: already horizontally flipped; skipped", flush=True)
            continue
        mask_path = masks_root / str(entry.get("filename") or "")
        if not mask_path.is_file():
            raise SystemExit(f"Query {number}: source mask is missing: {mask_path}")
        query_id = str(entry.get("query_id") or "")
        artifact_path = matcher_root(project) / "queries" / query_id / "artifact.json"
        if not artifact_path.is_file():
            raise SystemExit(f"Query {number}: saved artifact is missing")
        artifact = _read_json(artifact_path)
        with Image.open(mask_path) as source:
            width = source.width
            prepared = ImageOps.mirror(
                ImageOps.invert(source.convert("L")).convert("RGBA")
            )
        exterior = _mirror_points(_source_wall(artifact, "wall_a"), width)
        interior = _mirror_points(_source_wall(artifact, "wall_b"), width)
        fracture = _mirror_points(entry.get("fracture") or [], width)
        rim = _mirror_points([entry.get("rim_point")], width)[0]
        print(
            f"Query {number}: {entry['filename']} — mirror mask and saved smoothed walls",
            flush=True,
        )
        if args.dry_run:
            continue
        result = preprocess_query(
            project,
            prepared,
            original_filename=entry["filename"],
            metadata={
                "query_id": f"Real sherd {number}",
                "rim_diameter_cm": "",
                "fabric": "",
                "surface": "",
                "notes": "",
            },
            manual_curves={
                "exterior": exterior,
                "interior": interior,
                "fracture": fracture,
                "rim_point": rim,
            },
        )
        entry.update({
            "query_id": result["query_id"],
            "fracture": fracture,
            "rim_point": rim,
            "horizontal_flip": True,
            "horizontal_flip_axis": "x",
            "pre_flip_query_id": query_id,
        })
        value["queries"][str(number)] = entry
        _atomic_json(set_path, value)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
