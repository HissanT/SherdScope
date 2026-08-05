"""Reproducible synthetic-to-real diagnostics for the contour matcher.

The benchmark is deliberately read-only with respect to a SherdScope project.
It creates partial query contours in memory and writes results to a caller-
selected output directory. Reviewed masks, contour artifacts, query records,
saved matcher runs, and the retrieval index are never modified.
"""

from __future__ import annotations

import copy
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.ndimage import gaussian_filter1d

from catalog.contours import load_ready_artifacts
from catalog.matcher import (
    COARSE_LEVELS,
    MATCHER_ALGORITHM_VERSION,
    RETRIEVAL_ALGORITHM_VERSION,
    RETRIEVAL_KEEP,
    _master_boundary,
    _polyline_interval,
    _score_candidates_parallel,
    _split_continuous_boundary,
    retrieve_candidates,
)


BENCHMARK_SCHEMA_VERSION = 1
DEFAULT_TOP_K = (1, 5, 10, 50, 150, 300, 400)


@dataclass(frozen=True)
class SyntheticCondition:
    name: str
    point_noise: float
    smooth_noise: float


CONDITIONS = {
    "grid_clean": SyntheticCondition("grid_clean", 0.0, 0.0),
    "clean": SyntheticCondition("clean", 0.0, 0.0),
    "light": SyntheticCondition("light", 0.0008, 0.0015),
    "moderate": SyntheticCondition("moderate", 0.0018, 0.0035),
}


def _arc_fraction(points: np.ndarray, index: int) -> float:
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total = max(float(lengths.sum()), 1e-12)
    return float(lengths[:index].sum() / total)


def _point_at_arc_fraction(points: np.ndarray, fraction: float) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    total = max(float(cumulative[-1]), 1e-12)
    target = float(np.clip(fraction, 0.0, 1.0)) * total
    upper = min(int(np.searchsorted(cumulative, target, side="right")), len(points) - 1)
    lower = max(0, upper - 1)
    span = max(float(cumulative[upper] - cumulative[lower]), 1e-12)
    weight = (target - cumulative[lower]) / span
    return points[lower] * (1.0 - weight) + points[upper] * weight


def _rim_oriented_boundary(
    reference: dict[str, Any],
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Choose the upper publication cap as the archaeological rim.

    Reference masks are cropped from upright publication pages. Their stored
    nominal seam is a geometric end-cap hypothesis, not a reviewed rim label.
    Compare that seam with the opposite cap in original pixel coordinates,
    then reverse the master representation when the opposite cap is higher.
    """
    points, seam = _master_boundary(reference)
    master = reference.get("reference_master_boundary") or {}
    source = np.asarray(master.get("source_points", []), dtype=float)
    if source.ndim != 2 or source.shape[1:] != (2,) or len(source) < 8:
        raise ValueError(
            f"Reference {reference.get('reference_id')} lacks source coordinates "
            "required for synthetic rim validation"
        )
    source_seam = float(master.get("nominal_seam_fraction", seam))
    nominal_cap = _point_at_arc_fraction(source, source_seam)
    opposite_cap = 0.5 * (source[0] + source[-1])
    use_nominal = bool(nominal_cap[1] <= opposite_cap[1])
    diagnostics = {
        "rim_selection": "nominal_seam" if use_nominal else "opposite_cap",
        "source_nominal_cap_y": float(nominal_cap[1]),
        "source_opposite_cap_y": float(opposite_cap[1]),
        "selection_rule": "smaller_source_y_on_upright_publication_crop",
    }
    if use_nominal:
        return points, seam, diagnostics

    walls = _split_continuous_boundary(points, seam, 0.0, 384)
    alternate = np.vstack((walls["wall_a"], walls["wall_b"][-2::-1]))
    length_a = float(np.linalg.norm(np.diff(walls["wall_a"], axis=0), axis=1).sum())
    length_b = float(np.linalg.norm(np.diff(walls["wall_b"], axis=0), axis=1).sum())
    alternate_seam = length_a / max(length_a + length_b, 1e-12)
    return alternate, float(alternate_seam), diagnostics


def synthetic_query_from_reference(
    reference: dict[str, Any],
    *,
    rng: np.random.Generator,
    condition: SyntheticCondition,
    coverage: float | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create a rim-centred partial query while retaining exact provenance."""
    points, seam, rim_diagnostics = _rim_oriented_boundary(reference)
    if condition.name == "grid_clean":
        coverage = float(coverage or rng.choice((0.30, 0.45, 0.60)))
        asymmetry = float(rng.choice((0.75, 1.00, 1.25)))
    else:
        coverage = float(coverage or rng.uniform(0.28, 0.62))
        asymmetry = float(rng.uniform(0.72, 1.28))
    left_coverage = min(0.95, coverage * asymmetry)
    right_coverage = min(0.95, coverage / asymmetry)
    start = seam * (1.0 - left_coverage)
    end = seam + (1.0 - seam) * right_coverage
    partial = _polyline_interval(points, start, end, 192)

    # The source interval is sampled uniformly by arc length. Locate the rim
    # on that sampled curve, then retain its measured arc fraction after noise.
    nominal_seam = float(np.clip((seam - start) / (end - start), 0.08, 0.92))
    rim_index = int(round(nominal_seam * (len(partial) - 1)))

    # Draw both noise arrays even for the clean condition. This deliberately
    # keeps all later random values identical in paired clean/noisy trials.
    correlated = gaussian_filter1d(
        rng.normal(size=partial.shape), sigma=5.0, axis=0, mode="nearest"
    )
    independent = rng.normal(size=partial.shape)
    partial = (
        partial
        + condition.smooth_noise * correlated
        + condition.point_noise * independent
    )

    angle = math.radians(float(rng.uniform(-18.0, 18.0)))
    rotation = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=float,
    )
    scale = float(rng.uniform(0.72, 1.32))
    translation = rng.uniform(-0.18, 0.18, size=2)
    partial = scale * (partial @ rotation.T) + translation
    measured_seam = _arc_fraction(partial, rim_index)

    query = copy.deepcopy(reference)
    query_id = f"synthetic-{condition.name}-{reference['reference_id']}"
    query["reference_id"] = query_id
    query["query_id"] = query_id
    query["source_filename"] = f"{query_id}.synthetic"
    query["query_master_boundary"] = {
        "points": partial.tolist(),
        "source_points": partial.tolist(),
        "nominal_seam_fraction": measured_seam,
        "annotated_seam_fraction": measured_seam,
        "semantics": "synthetic_partial_fracture_to_rim_to_fracture_boundary",
        "split_before_wall_resampling": True,
    }
    provenance = {
        "parent_reference_id": reference["reference_id"],
        "condition": condition.name,
        "coverage": coverage,
        "left_coverage": left_coverage,
        "right_coverage": right_coverage,
        "rotation_degrees": math.degrees(angle),
        "scale": scale,
        "translation": translation.tolist(),
        "point_noise": condition.point_noise,
        "smooth_noise": condition.smooth_noise,
        **rim_diagnostics,
    }
    query["synthetic_provenance"] = provenance
    return query, provenance


def _rank_for_parent(
    candidates: Iterable[dict[str, Any]], parent_reference_id: str
) -> int | None:
    for rank, candidate in enumerate(candidates, start=1):
        artifact = candidate.get("artifact", candidate)
        if artifact.get("reference_id") == parent_reference_id:
            return rank
    return None


def _full_rerank(
    query: dict[str, Any], candidates: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    active = list(candidates)
    for level_index, (samples, keep) in enumerate(COARSE_LEVELS):
        scored = _score_candidates_parallel(
            query,
            active,
            samples,
            exact_transport=level_index > 0,
            level_index=level_index,
            workers=(None if level_index == 0 else 1),
        )
        if not scored:
            return []
        scored.sort(
            key=lambda item: (item["overall_score"], item["reference_id"])
        )
        retained_ids = {item["reference_id"] for item in scored[:keep]}
        by_id = {candidate["artifact"]["reference_id"]: candidate for candidate in active}
        active = []
        for result in scored:
            if result["reference_id"] not in retained_ids:
                continue
            candidate = by_id[result["reference_id"]]
            candidate["previous"] = result
            active.append(candidate)
    return scored[: COARSE_LEVELS[-1][1]]


def _summary(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    ranks = [row.get(field) for row in rows]
    present = [int(rank) for rank in ranks if isinstance(rank, int)]
    total = len(rows)
    summary = {
        "total": total,
        "missing": total - len(present),
        "mean_reciprocal_rank": (
            float(np.mean([1.0 / rank for rank in present])) if present else 0.0
        ),
        "median_rank": float(np.median(present)) if present else None,
    }
    for k in DEFAULT_TOP_K:
        summary[f"top_{k}"] = sum(rank <= k for rank in present)
        summary[f"top_{k}_accuracy"] = (
            summary[f"top_{k}"] / total if total else 0.0
        )
    return summary


def run_synthetic_benchmark(
    project_path: Path,
    output_dir: Path,
    *,
    sample_size: int = 90,
    seed: int = 20260801,
    condition_names: tuple[str, ...] = ("grid_clean", "clean", "light", "moderate"),
    full_rerank_limit: int = 0,
) -> dict[str, Any]:
    """Run a deterministic benchmark and persist machine-readable results."""
    project_path = Path(project_path)
    output_dir = Path(output_dir)
    references = load_ready_artifacts(project_path)
    if not references:
        raise ValueError("The project has no approved contour references")
    invalid = [name for name in condition_names if name not in CONDITIONS]
    if invalid:
        raise ValueError(f"Unknown synthetic condition(s): {', '.join(invalid)}")

    rng = np.random.default_rng(seed)
    selected_count = min(max(1, int(sample_size)), len(references))
    selected_indices = rng.choice(
        len(references), size=selected_count, replace=False
    )
    selected = [references[int(index)] for index in selected_indices]
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    full_remaining = max(0, int(full_rerank_limit))

    for reference in selected:
        paired_seed = int(rng.integers(0, np.iinfo(np.int64).max))
        for condition_name in condition_names:
            query, provenance = synthetic_query_from_reference(
                reference,
                rng=np.random.default_rng(paired_seed),
                condition=CONDITIONS[condition_name],
            )
            retrieval_started = time.perf_counter()
            candidates, diagnostics = retrieve_candidates(
                project_path, query, references, keep=RETRIEVAL_KEEP
            )
            retrieval_seconds = time.perf_counter() - retrieval_started
            parent_id = provenance["parent_reference_id"]
            retrieval_rank = _rank_for_parent(candidates, parent_id)
            final_rank = None
            rerank_seconds = None
            if full_remaining > 0:
                rerank_started = time.perf_counter()
                final = _full_rerank(query, candidates)
                rerank_seconds = time.perf_counter() - rerank_started
                final_rank = _rank_for_parent(final, parent_id)
                full_remaining -= 1
            rows.append(
                {
                    **provenance,
                    "retrieval_rank": retrieval_rank,
                    "final_rank": final_rank,
                    "retrieval_seconds": retrieval_seconds,
                    "rerank_seconds": rerank_seconds,
                    "retrieval_cache_hit": diagnostics.get("cache_hit"),
                    "reference_count": len(references),
                }
            )

    summaries = {}
    for condition_name in condition_names:
        condition_rows = [
            row for row in rows if row["condition"] == condition_name
        ]
        summaries[condition_name] = {
            "retrieval": _summary(condition_rows, "retrieval_rank"),
            "final": _summary(
                [row for row in condition_rows if row["rerank_seconds"] is not None],
                "final_rank",
            ),
        }
    report = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "matcher_algorithm_version": MATCHER_ALGORITHM_VERSION,
        "retrieval_algorithm_version": RETRIEVAL_ALGORITHM_VERSION,
        "seed": seed,
        "reference_count": len(references),
        "selected_parent_count": len(selected),
        "query_count": len(rows),
        "conditions": list(condition_names),
        "full_rerank_requested": int(full_rerank_limit),
        "elapsed_seconds": time.perf_counter() - started,
        "summary": summaries,
        "rows": rows,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "synthetic_benchmark.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    csv_path = output_dir / "synthetic_benchmark.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report["json_path"] = str(json_path)
    report["csv_path"] = str(csv_path)
    return report
