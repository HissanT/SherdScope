"""Partial three-curve matching and diagnostic rendering for SherdScope."""

from __future__ import annotations

import json
import math
import os
import uuid
import csv
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import cdist

from catalog.contours import (
    CONTOUR_ALGORITHM_VERSION,
    ContourError,
    _atomic_json,
    _resample_curve,
    build_manual_query_artifact,
    card_stem,
    find_contours_by_citation,
    load_ready_artifacts,
    matcher_root,
    render_previews,
)
from catalog.metadata_fusion import fuse_shape_results


MATCHER_SCHEMA_VERSION = 1
MATCHER_ALGORITHM_VERSION = "two-wall-joint-fgw-v15-final-balanced-metadata"
RETRIEVAL_ALGORITHM_VERSION = "dual-outline-ribbon-metadata-v3"
CURVE_NAMES = ("wall_a", "wall_b")
COARSE_LEVELS = ((24, 40), (48, 12), (96, 5))
RETRIEVAL_SAMPLES = 20
RETRIEVAL_WALL_SAMPLES = 12
RETRIEVAL_KEEP = 400
RETRIEVAL_METHOD_KEEP = 150
# Channel champions are appended to the score shortlist. They never displace a
# candidate that earned its place in the ordinary top-K shape ranking.
CASCADE_CHANNEL_APPEND = (2, 1, 0)
DEFAULT_MATCHER_WORKERS = min(4, max(1, os.cpu_count() or 1))
MAX_MATCHER_WORKERS = 8
RETRIEVAL_WINDOW_COVERAGES = (0.20, 0.30, 0.45, 0.60, 0.80, 1.00)
RETRIEVAL_WINDOW_ASYMMETRIES = (0.75, 1.00, 1.25)
FGW_ALPHA = 0.55
MAX_ALIGNMENT_ROTATION_DEGREES = 45.0
MIN_REFERENCE_INTERVAL_FRACTION = 0.20
RIM_ANCHOR_TOLERANCE_FRACTION = 0.05
RIM_SEAM_INITIAL_OFFSET_FRACTION = 0.125
RIM_SEAM_EXPANSION_STEP_FRACTION = 0.0625
RIM_SEAM_MAX_OFFSET_FRACTION = 0.25
RIM_STABLE_GUARD_FRACTION = 0.0625
RIM_REGION_FRACTION = 0.125
REFERENCE_RIM_SPLIT_FRACTION = 0.125
REFERENCE_RIM_SPLITS = (
    ("A", -REFERENCE_RIM_SPLIT_FRACTION),
    ("B", 0.0),
    ("C", REFERENCE_RIM_SPLIT_FRACTION),
)
MAX_INTERVAL_REFINEMENTS = 2
MAX_JOINT_ALIGNMENT_ITERATIONS = 7
MIN_ALIGNMENT_SCALE = 0.10
MAX_ALIGNMENT_SCALE = 10.0
JOINT_PROGRESS_TOLERANCE = 0.75
JOINT_ROTATION_TOLERANCE_DEGREES = 0.05
JOINT_SCALE_TOLERANCE = 1e-3
SALIENCE_LOCAL_WINDOW_FRACTION = 1.0 / 24.0
SALIENCE_ENDPOINT_GUARD_FRACTION = 1.0 / 24.0
RANK_WEIGHTS = {
    "fgw": 0.31,
    "ribbon": 0.26,
    "salience": 0.13,
    "alignment_tail": 0.08,
    "rim_region": 0.10,
    "transform_reliability": 0.06,
    "completeness": 0.06,
}
# Raw metrics have different natural units. Convert each to a bounded 0..1
# mismatch before applying percentage weights so a nominal 15% term cannot
# silently become most of the final score.
SCORE_SCALES = {
    "fgw": 0.10,
    "ribbon": 0.08,
    "salience": 0.50,
    "alignment_tail": 0.12,
    "rim_region": 0.06,
}
DESCRIPTOR_WEIGHTS = {
    "radius": 0.25,
    "tangent": 0.25,
    "curvature": 0.50,
}


class MatcherError(ValueError):
    """Raised for invalid matcher inputs or unavailable matcher state."""


def query_root(project_path: Path) -> Path:
    return matcher_root(project_path) / "queries"


def run_root(project_path: Path) -> Path:
    return matcher_root(project_path) / "runs"


def matcher_config() -> dict[str, Any]:
    return {
        "schema_version": MATCHER_SCHEMA_VERSION,
        "algorithm_version": MATCHER_ALGORITHM_VERSION,
        "contour_algorithm_version": CONTOUR_ALGORITHM_VERSION,
        "coarse_levels": [list(level) for level in COARSE_LEVELS],
        "retrieval": {
            "algorithm_version": RETRIEVAL_ALGORITHM_VERSION,
            "samples": RETRIEVAL_SAMPLES,
            "wall_samples": RETRIEVAL_WALL_SAMPLES,
            "keep": RETRIEVAL_KEEP,
            "per_method_keep": RETRIEVAL_METHOD_KEEP,
            "window_coverages": list(RETRIEVAL_WINDOW_COVERAGES),
            "window_asymmetries": list(RETRIEVAL_WINDOW_ASYMMETRIES),
            "reference_end_caps_tested": 2,
            "fixed_split_required": False,
            "outline_split_independent": True,
            "methods": ["continuous_outline", "split_flexible_ribbon"],
            "metadata_used": False,
        },
        "fgw_alpha": FGW_ALPHA,
        "max_alignment_rotation_degrees": MAX_ALIGNMENT_ROTATION_DEGREES,
        "min_reference_interval_fraction": MIN_REFERENCE_INTERVAL_FRACTION,
        "rim_anchor_tolerance_fraction": RIM_ANCHOR_TOLERANCE_FRACTION,
        "rim_seam_max_offset_fraction": RIM_SEAM_MAX_OFFSET_FRACTION,
        "rim_seam_initial_offset_fraction": RIM_SEAM_INITIAL_OFFSET_FRACTION,
        "rim_seam_expansion_step_fraction": RIM_SEAM_EXPANSION_STEP_FRACTION,
        "rim_stable_guard_fraction": RIM_STABLE_GUARD_FRACTION,
        "rim_region_fraction": RIM_REGION_FRACTION,
        "reference_rim_splits": [
            {"label": label, "offset_fraction": offset}
            for label, offset in REFERENCE_RIM_SPLITS
        ],
        "rim_seam_candidates": {
            "coarse": 5,
            "medium": 5,
            "fine": 5,
            "adaptive_extra_per_direction": 2,
        },
        "exact_hypotheses": {"medium": 2, "fine": 3},
        "max_interval_refinements": MAX_INTERVAL_REFINEMENTS,
        "max_joint_alignment_iterations": MAX_JOINT_ALIGNMENT_ITERATIONS,
        "alignment_scale_range": [MIN_ALIGNMENT_SCALE, MAX_ALIGNMENT_SCALE],
        "rank_weights": dict(RANK_WEIGHTS),
        "candidate_workers": matcher_worker_count(),
        "parallel_levels": ["coarse_descriptor_alignment"],
        "exact_solver_parallel": False,
        "completeness_penalty": "squared_unmatched_reference_fraction",
        "score_scales": dict(SCORE_SCALES),
        "descriptor_weights": dict(DESCRIPTOR_WEIGHTS),
        "reflection_allowed": False,
        "metadata_used": False,
        "pot_required": True,
        "ordered_shared_correspondence": True,
        "query_master_boundary_split_before_resampling": True,
        "reference_master_boundary_split_before_resampling": True,
        "noise_handling": (
            "v9-compatible multiscale shape scoring; experimental persistent "
            "local-shape retrieval is disabled until held-out validation"
        ),
        "cascade_channel_append": list(CASCADE_CHANNEL_APPEND),
        "confidence_policy": "deferred_not_used_for_ranking_or_ui",
    }


def matcher_worker_count() -> int:
    """Return a bounded worker count without changing process-global state."""
    raw = os.environ.get("SHERDSCOPE_MATCHER_WORKERS", "").strip()
    if not raw:
        return DEFAULT_MATCHER_WORKERS
    try:
        requested = int(raw)
    except ValueError:
        return DEFAULT_MATCHER_WORKERS
    return min(MAX_MATCHER_WORKERS, max(1, requested))


def pot_runtime_status() -> dict[str, Any]:
    """Report whether the required POT solver is available to this process."""
    try:
        import ot
    except Exception as exc:
        return {
            "available": False,
            "version": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "available": True,
        "version": str(getattr(ot, "__version__", "unknown")),
        "error": None,
    }


def _retrieval_index_path(project_path: Path) -> Path:
    return matcher_root(project_path) / "retrieval_index.npz"


def _master_boundary(
    artifact: dict[str, Any],
) -> tuple[np.ndarray, float]:
    """Return one continuous boundary and its approximate rim position."""
    for field in ("query_master_boundary", "reference_master_boundary"):
        master = artifact.get(field)
        if not isinstance(master, dict):
            continue
        try:
            points = np.asarray(master["points"], dtype=float)
            seam = float(
                master.get(
                    "annotated_seam_fraction",
                    master.get("nominal_seam_fraction", 0.5),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
        if (
            points.ndim == 2
            and points.shape[1:] == (2,)
            and len(points) >= 8
            and np.all(np.isfinite(points))
        ):
            return points, float(np.clip(seam, 0.05, 0.95))

    count = max(32, int(artifact.get("sample_count", 96)))
    curves = _normalised_curves(artifact, count)
    wall_a = curves["wall_a"]
    wall_b = curves["wall_b"]
    points = np.vstack((wall_a[::-1], wall_b[1:]))
    length_a = float(np.linalg.norm(np.diff(wall_a, axis=0), axis=1).sum())
    length_b = float(np.linalg.norm(np.diff(wall_b, axis=0), axis=1).sum())
    seam = length_a / max(length_a + length_b, 1e-12)
    return points, float(np.clip(seam, 0.05, 0.95))


def _polyline_interval(
    points: np.ndarray,
    start_fraction: float,
    end_fraction: float,
    samples: int,
) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    keep = np.concatenate(
        (
            np.array([True]),
            np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-10,
        )
    )
    points = points[keep]
    if len(points) < 3:
        raise MatcherError("Retrieval boundary is too short")
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate((np.array([0.0]), np.cumsum(lengths)))
    total = float(cumulative[-1])
    if total <= 1e-10:
        raise MatcherError("Retrieval boundary has zero length")
    start = float(np.clip(start_fraction, 0.0, 1.0)) * total
    end = float(np.clip(end_fraction, 0.0, 1.0)) * total
    if end - start <= total * 0.02:
        raise MatcherError("Retrieval boundary interval is too short")
    targets = np.linspace(start, end, int(samples))
    return np.column_stack(
        (
            np.interp(targets, cumulative, points[:, 0]),
            np.interp(targets, cumulative, points[:, 1]),
        )
    )


def _retrieval_descriptor(points: np.ndarray) -> np.ndarray:
    """Rotation/translation/scale-invariant descriptor for one open outline."""
    points = np.asarray(points, dtype=float)
    pairwise = cdist(points, points)
    upper = np.triu_indices(len(points), 1)
    distances = pairwise[upper]
    distance_scale = max(float(np.quantile(distances, 0.90)), 1e-8)
    distances = distances / distance_scale

    segments = np.diff(points, axis=0)
    segments /= np.maximum(
        np.linalg.norm(segments, axis=1, keepdims=True), 1e-8
    )
    turn_cos = np.sum(segments[:-1] * segments[1:], axis=1)
    turn_sin = (
        segments[:-1, 0] * segments[1:, 1]
        - segments[:-1, 1] * segments[1:, 0]
    )
    turns = np.column_stack((turn_cos, turn_sin)).ravel()
    # Scaling by component count makes squared Euclidean distance equal to a
    # weighted mean rather than favouring the longer distance block.
    distance_block = distances * math.sqrt(0.75 / max(len(distances), 1))
    turn_block = turns * math.sqrt(0.25 / max(len(turns), 1))
    return np.concatenate((distance_block, turn_block)).astype(np.float32)


def _query_retrieval_descriptors(
    artifact: dict[str, Any],
) -> np.ndarray:
    points, _ = _master_boundary(artifact)
    sampled = _polyline_interval(
        points, 0.0, 1.0, RETRIEVAL_SAMPLES
    )
    return np.vstack(
        (
            _retrieval_descriptor(sampled),
            _retrieval_descriptor(sampled[::-1]),
        )
    )


def _local_shape_retrieval_descriptor(points: np.ndarray) -> np.ndarray:
    """Compact descriptor which preserves cross-scale local shape events."""
    sampled = _resample_curve(np.asarray(points, dtype=float), 32)
    curvature = _persistent_curvature_features(sampled)
    tangent = np.gradient(sampled, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-8)
    # Remove global rotation while preserving ordered local turning.
    angles = np.unwrap(np.arctan2(tangent[:, 1], tangent[:, 0]))
    angles -= angles[0]
    turns = np.column_stack((np.cos(angles), np.sin(angles)))
    return np.concatenate(
        (
            curvature.ravel() * math.sqrt(0.70 / curvature.size),
            turns.ravel() * math.sqrt(0.30 / turns.size),
        )
    ).astype(np.float32)


def _query_local_shape_descriptors(artifact: dict[str, Any]) -> np.ndarray:
    points, _ = _master_boundary(artifact)
    return np.vstack(
        (
            _local_shape_retrieval_descriptor(points),
            _local_shape_retrieval_descriptor(points[::-1]),
        )
    )


def _reference_local_shape_descriptors(artifact: dict[str, Any]) -> np.ndarray:
    points, seam = _master_boundary(artifact)
    variants = []
    for coverage in RETRIEVAL_WINDOW_COVERAGES:
        for asymmetry in RETRIEVAL_WINDOW_ASYMMETRIES:
            left = min(1.0, coverage * asymmetry)
            right = min(1.0, coverage / asymmetry)
            start = seam * (1.0 - left)
            end = seam + (1.0 - seam) * right
            window = _polyline_interval(points, start, end, 32)
            variants.append(_local_shape_retrieval_descriptor(window))
            variants.append(_local_shape_retrieval_descriptor(window[::-1]))
    return np.asarray(variants, dtype=np.float32)


def _reference_retrieval_descriptors(
    artifact: dict[str, Any],
) -> np.ndarray:
    points, seam = _master_boundary(artifact)
    variants = []
    walls = _split_continuous_boundary(points, seam, 0.0, 96)
    alternate = np.vstack(
        (walls["wall_a"], walls["wall_b"][-2::-1])
    )
    alternate_a = float(
        np.linalg.norm(np.diff(walls["wall_a"], axis=0), axis=1).sum()
    )
    alternate_b = float(
        np.linalg.norm(np.diff(walls["wall_b"], axis=0), axis=1).sum()
    )
    alternate_seam = alternate_a / max(alternate_a + alternate_b, 1e-12)
    # PCA gives two possible end caps and does not tell us which is the rim.
    # Search windows around both. This keeps retrieval independent of a
    # possibly inverted automatic reference orientation.
    for master_points, master_seam in (
        (points, seam),
        (alternate, alternate_seam),
    ):
        for coverage in RETRIEVAL_WINDOW_COVERAGES:
            for asymmetry in RETRIEVAL_WINDOW_ASYMMETRIES:
                left_coverage = min(1.0, coverage * asymmetry)
                right_coverage = min(1.0, coverage / asymmetry)
                start = master_seam * (1.0 - left_coverage)
                end = master_seam + (1.0 - master_seam) * right_coverage
                sampled = _polyline_interval(
                    master_points, start, end, RETRIEVAL_SAMPLES
                )
                variants.append(_retrieval_descriptor(sampled))
    return np.asarray(variants, dtype=np.float32)


def _ribbon_retrieval_descriptor(
    curves: dict[str, np.ndarray],
) -> np.ndarray:
    wall_a = np.asarray(curves["wall_a"], dtype=float)
    wall_b = np.asarray(curves["wall_b"], dtype=float)
    points = np.vstack((wall_a, wall_b))
    pairwise = cdist(points, points)
    upper = np.triu_indices(len(points), 1)
    distances = pairwise[upper]
    distances /= max(float(np.quantile(distances, 0.90)), 1e-8)

    thickness = np.linalg.norm(wall_a - wall_b, axis=1)
    thickness /= max(float(np.median(thickness)), 1e-8)
    tangent_a = np.diff(wall_a, axis=0)
    tangent_b = np.diff(wall_b, axis=0)
    tangent_a /= np.maximum(
        np.linalg.norm(tangent_a, axis=1, keepdims=True), 1e-8
    )
    tangent_b /= np.maximum(
        np.linalg.norm(tangent_b, axis=1, keepdims=True), 1e-8
    )
    tangent_agreement = np.sum(tangent_a * tangent_b, axis=1)
    return np.concatenate(
        (
            distances * math.sqrt(0.70 / max(len(distances), 1)),
            thickness * math.sqrt(0.20 / max(len(thickness), 1)),
            tangent_agreement
            * math.sqrt(0.10 / max(len(tangent_agreement), 1)),
        )
    ).astype(np.float32)


def _query_ribbon_retrieval_descriptors(
    artifact: dict[str, Any],
) -> np.ndarray:
    variants = []
    for offset in _rim_seam_offsets(RETRIEVAL_SAMPLES):
        curves = _query_curves_for_rim_seam(
            artifact, offset, RETRIEVAL_WALL_SAMPLES
        )
        variants.append(_ribbon_retrieval_descriptor(curves))
        variants.append(
            _ribbon_retrieval_descriptor(
                {
                    "wall_a": curves["wall_b"],
                    "wall_b": curves["wall_a"],
                }
            )
        )
    return np.asarray(variants, dtype=np.float32)


def _reference_ribbon_retrieval_descriptors(
    artifact: dict[str, Any],
) -> np.ndarray:
    variants = []
    source_samples = max(96, RETRIEVAL_WALL_SAMPLES * 4)
    for _, split_offset in REFERENCE_RIM_SPLITS:
        for reverse in (False, True):
            curves = _reference_curves_for_rim_split(
                artifact,
                split_offset,
                source_samples,
                reverse=reverse,
            )
            for coverage in RETRIEVAL_WINDOW_COVERAGES:
                count = max(
                    3, int(round(source_samples * float(coverage)))
                )
                variants.append(
                    _ribbon_retrieval_descriptor(
                        {
                            "wall_a": _resample_curve(
                                curves["wall_a"][:count],
                                RETRIEVAL_WALL_SAMPLES,
                            ),
                            "wall_b": _resample_curve(
                                curves["wall_b"][:count],
                                RETRIEVAL_WALL_SAMPLES,
                            ),
                        }
                    )
                )
    return np.asarray(variants, dtype=np.float32)


def _descriptor_scores(
    descriptors: np.ndarray,
    query_descriptors: np.ndarray,
    *,
    chunk_size: int = 256,
) -> np.ndarray:
    scores = np.full(len(descriptors), np.inf, dtype=float)
    for start in range(0, len(descriptors), chunk_size):
        end = min(len(descriptors), start + chunk_size)
        block = descriptors[start:end]
        local = np.full(end - start, np.inf, dtype=float)
        for query_descriptor in query_descriptors:
            differences = block - query_descriptor[None, None, :]
            variant_scores = np.sum(differences * differences, axis=2)
            local = np.minimum(local, np.min(variant_scores, axis=1))
        scores[start:end] = local
    return scores


def _retrieval_signature(references: list[dict[str, Any]]) -> str:
    records = [
        {
            "id": str(reference.get("reference_id") or ""),
            "source": reference.get("source_fingerprint"),
            "algorithm": reference.get("algorithm_version"),
            "smoothing": reference.get("smoothing_mode"),
            "samples": reference.get("sample_count"),
        }
        for reference in references
    ]
    payload = json.dumps(
        {
            "version": RETRIEVAL_ALGORITHM_VERSION,
            "samples": RETRIEVAL_SAMPLES,
            "wall_samples": RETRIEVAL_WALL_SAMPLES,
            "method_keep": RETRIEVAL_METHOD_KEEP,
            "coverages": RETRIEVAL_WINDOW_COVERAGES,
            "asymmetries": RETRIEVAL_WINDOW_ASYMMETRIES,
            "references": records,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_or_build_retrieval_index(
    project_path: Path,
    references: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, list[str], bool, float]:
    started = time.perf_counter()
    signature = _retrieval_signature(references)
    expected_ids = [
        str(reference.get("reference_id") or "") for reference in references
    ]
    path = _retrieval_index_path(project_path)
    if path.exists():
        try:
            with np.load(path, allow_pickle=False) as stored:
                stored_signature = str(stored["signature"].item())
                identifiers = [str(value) for value in stored["reference_ids"]]
                outline_descriptors = np.asarray(
                    stored["outline_descriptors"], dtype=np.float32
                )
                ribbon_descriptors = np.asarray(
                    stored["ribbon_descriptors"], dtype=np.float32
                )
            if (
                stored_signature == signature
                and identifiers == expected_ids
                and outline_descriptors.shape[0] == len(references)
                and ribbon_descriptors.shape[0] == len(references)
            ):
                return (
                    outline_descriptors,
                    ribbon_descriptors,
                    identifiers,
                    True,
                    time.perf_counter() - started,
                )
        except (OSError, ValueError, KeyError):
            pass

    outline_descriptors = np.stack(
        [
            _reference_retrieval_descriptors(reference)
            for reference in references
        ]
    ).astype(np.float32)
    ribbon_descriptors = np.stack(
        [
            _reference_ribbon_retrieval_descriptors(reference)
            for reference in references
        ]
    ).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with open(temporary, "wb") as handle:
        np.savez_compressed(
            handle,
            signature=np.asarray(signature),
            reference_ids=np.asarray(expected_ids),
            outline_descriptors=outline_descriptors,
            ribbon_descriptors=ribbon_descriptors,
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return (
        outline_descriptors,
        ribbon_descriptors,
        expected_ids,
        False,
        time.perf_counter() - started,
    )


def retrieve_candidates(
    project_path: Path,
    query: dict[str, Any],
    references: list[dict[str, Any]],
    *,
    keep: int = RETRIEVAL_KEEP,
    known_target: dict[str, str] | None = None,
    query_metadata: dict[str, Any] | None = None,
    reference_metadata: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return a broad shortlist, optionally using metadata before top-K pruning."""
    if not references:
        return [], {
            "algorithm_version": RETRIEVAL_ALGORITHM_VERSION,
            "input_count": 0,
            "kept_count": 0,
        }
    (
        outline_descriptors,
        ribbon_descriptors,
        identifiers,
        cache_hit,
        index_seconds,
    ) = (
        _load_or_build_retrieval_index(project_path, references)
    )
    search_started = time.perf_counter()
    outline_scores = _descriptor_scores(
        outline_descriptors,
        _query_retrieval_descriptors(query),
    )
    ribbon_scores = _descriptor_scores(
        ribbon_descriptors,
        _query_ribbon_retrieval_descriptors(query),
    )
    outline_order = sorted(
        range(len(references)),
        key=lambda index: (float(outline_scores[index]), identifiers[index]),
    )
    ribbon_order = sorted(
        range(len(references)),
        key=lambda index: (float(ribbon_scores[index]), identifiers[index]),
    )
    outline_ranks = np.empty(len(references), dtype=int)
    ribbon_ranks = np.empty(len(references), dtype=int)
    for rank, index in enumerate(outline_order, start=1):
        outline_ranks[index] = rank
    for rank, index in enumerate(ribbon_order, start=1):
        ribbon_ranks[index] = rank

    target = min(max(1, int(keep)), len(references))
    method_keep = min(RETRIEVAL_METHOD_KEEP, target, len(references))

    def combined_key(index: int) -> tuple[int, int, str]:
        return (
            int(min(outline_ranks[index], ribbon_ranks[index])),
            int(outline_ranks[index] + ribbon_ranks[index]),
            identifiers[index],
        )

    shape_order = sorted(range(len(references)), key=combined_key)
    shape_rank = {index: rank for rank, index in enumerate(shape_order, start=1)}
    metadata_enabled = bool(query_metadata and reference_metadata)
    retrieval_rows = []
    for index, reference in enumerate(references):
        # Rank percentiles put the two descriptor channels on a stable 0..1
        # scale. The best channel remains primary while agreement between both
        # channels supplies the remaining evidence.
        best = min(outline_ranks[index], ribbon_ranks[index]) / len(references)
        mean = (outline_ranks[index] + ribbon_ranks[index]) / (2 * len(references))
        retrieval_rows.append({
            "reference_id": identifiers[index],
            "source_filename": reference.get("source_filename", ""),
            "shape_score": float(.75 * best + .25 * mean),
            "reference_index": index,
        })
    if metadata_enabled:
        retrieval_rows = fuse_shape_results(
            retrieval_rows, query_metadata or {}, reference_metadata or {}
        )
        kept = [int(row["reference_index"]) for row in retrieval_rows[:target]]
        metadata_by_index = {
            int(row["reference_index"]): row for row in retrieval_rows
        }
    else:
        kept = shape_order[:target]
        metadata_by_index = {}
    selected = []
    for retrieval_rank, index in enumerate(kept, start=1):
        selected_by = []
        if outline_ranks[index] <= method_keep:
            selected_by.append("continuous_outline")
        if ribbon_ranks[index] <= method_keep:
            selected_by.append("split_flexible_ribbon")
        selected.append(
            {
                "artifact": references[index],
                "retrieval": {
                    "rank": retrieval_rank,
                    "score": float(
                        min(outline_scores[index], ribbon_scores[index])
                    ),
                    "outline_rank": int(outline_ranks[index]),
                    "outline_score": float(outline_scores[index]),
                    "ribbon_rank": int(ribbon_ranks[index]),
                    "ribbon_score": float(ribbon_scores[index]),
                    "selected_by": selected_by or ["combined_fill"],
                    "shape_rank": int(shape_rank[index]),
                    **({
                        "metadata_rank": int(metadata_by_index[index]["fused_rank"]),
                        "metadata_adjustment": metadata_by_index[index]["metadata_adjustment"],
                        "metadata_score": metadata_by_index[index]["metadata_score"],
                    } if metadata_enabled else {}),
                },
            }
        )
    diagnostics = {
        "algorithm_version": RETRIEVAL_ALGORITHM_VERSION,
        "input_count": len(references),
        "kept_count": len(selected),
        "keep_limit": int(keep),
        "per_method_keep": int(method_keep),
        "cache_hit": cache_hit,
        "index_seconds": float(index_seconds),
        "search_seconds": float(time.perf_counter() - search_started),
        "fixed_split_required": False,
        "outline_split_independent": True,
        "methods": ["continuous_outline", "split_flexible_ribbon"],
        "metadata_used": metadata_enabled,
    }
    if known_target:
        matching = [
            index for index, reference in enumerate(references)
            if str(reference.get("figure", "")) == str(known_target.get("figure", ""))
            and str(reference.get("item", "")) == str(known_target.get("item", ""))
        ]
        if matching:
            index = min(matching, key=combined_key)
            diagnostics["target"] = {
                "reference_id": identifiers[index],
                "selected": index in set(kept),
                "combined_rank": 1 + sorted(
                    range(len(references)), key=combined_key
                ).index(index),
                "metadata_rank": (
                    int(metadata_by_index[index]["fused_rank"])
                    if metadata_enabled else None
                ),
                "outline_rank": int(outline_ranks[index]),
                "ribbon_rank": int(ribbon_ranks[index]),
            }
        else:
            diagnostics["target"] = None
    return selected, diagnostics


def _curvature(points: np.ndarray, sigma: float) -> np.ndarray:
    x = gaussian_filter1d(points[:, 0], sigma=sigma, mode="nearest")
    y = gaussian_filter1d(points[:, 1], sigma=sigma, mode="nearest")
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    denominator = np.maximum((dx * dx + dy * dy) ** 1.5, 1e-8)
    return (dx * ddy - dy * ddx) / denominator


def _persistent_curvature_features(points: np.ndarray) -> np.ndarray:
    """Describe fine geometry only when its direction persists across scales.

    This is deliberately not blanket smoothing.  The fine response remains in
    the descriptor when neighbouring scales confirm its sign and location, as
    a real lip, rib or small bump normally does.  One-pixel stair steps tend to
    change sign or vanish immediately and therefore receive less influence.
    """
    responses = np.column_stack(
        [_curvature(points, sigma) for sigma in (0.75, 1.5, 3.0, 6.0)]
    )
    scales = np.maximum(
        np.quantile(np.abs(responses), 0.90, axis=0), 1e-5
    )
    normalised = np.clip(responses / scales, -4.0, 4.0)
    fine, local, medium, broad = normalised.T
    sign_support = (
        (np.sign(fine) == np.sign(local)).astype(float)
        + (np.sign(fine) == np.sign(medium)).astype(float)
    ) / 2.0
    magnitude_support = np.clip(
        (np.abs(responses[:, 1]) + 0.5 * np.abs(responses[:, 2]))
        / np.maximum(np.abs(responses[:, 0]), 1e-8),
        0.0,
        1.0,
    )
    persistence = 0.05 + 0.95 * sign_support * magnitude_support
    persistent_detail = fine * persistence
    # Derivative estimates at open-curve endpoints are inherently one-sided.
    # Do not trust their finest channel; rim/lip geometry remains represented
    # by the medium and broad channels below.
    endpoint_guard = max(2, int(round(len(points) * 0.04)))
    persistent_detail[:endpoint_guard] = 0.0
    persistent_detail[-endpoint_guard:] = 0.0
    return np.column_stack((persistent_detail, medium, broad))


def _normalised_angle_degrees(angle: float) -> float:
    return ((float(angle) + 180.0) % 360.0) - 180.0


def _alignment_allowed(alignment: dict[str, Any]) -> bool:
    return abs(_normalised_angle_degrees(alignment["rotation_degrees"])) <= MAX_ALIGNMENT_ROTATION_DEGREES


def _normalised_curves(
    artifact: dict[str, Any],
    samples: int,
    reverse: bool = False,
    swap_walls: bool = False,
):
    curves = {}
    source_names = {
        "wall_a": "wall_b" if swap_walls else "wall_a",
        "wall_b": "wall_a" if swap_walls else "wall_b",
        "centreline": "centreline",
    }
    for name in CURVE_NAMES:
        points = _resample_curve(
            np.asarray(artifact["curves"][source_names[name]], dtype=float),
            samples,
        )
        curves[name] = points[::-1].copy() if reverse else points
    return curves


def _rim_seam_offsets(
    samples: int, hint: float | None = None
) -> list[float]:
    """Return a bounded five-candidate coarse-to-fine rim seam search.

    The first pass spans the complete allowed neighbourhood so a badly placed
    gold hint cannot be trapped behind an initially narrow local search.
    Later resolutions refine around the best physical split found so far.
    """
    if hint is None or samples <= 24:
        offsets = (-0.25, -0.125, 0.0, 0.125, 0.25)
    elif samples <= 48:
        offsets = tuple(float(hint) + delta for delta in (-0.04, -0.02, 0.0, 0.02, 0.04))
    else:
        offsets = tuple(float(hint) + delta for delta in (-0.02, -0.01, 0.0, 0.01, 0.02))
    clipped = {
        round(
            float(
                np.clip(
                    value,
                    -RIM_SEAM_MAX_OFFSET_FRACTION,
                    RIM_SEAM_MAX_OFFSET_FRACTION,
                )
            ),
            6,
        )
        for value in offsets
    }
    return sorted(clipped)


def _shift_rim_seam(
    curves: dict[str, np.ndarray],
    offset_fraction: float,
    samples: int,
) -> dict[str, np.ndarray]:
    """Move only the conventional wall split while preserving the rim outline.

    Positive offsets move the seam down wall A and transfer that short rim arc
    to wall B. Negative offsets do the opposite. The physical boundary is
    unchanged; only its two-wall parameterisation changes.
    """
    wall_a = np.asarray(curves["wall_a"], dtype=float)
    wall_b = np.asarray(curves["wall_b"], dtype=float)
    maximum = max(1, int(round((samples - 1) * RIM_SEAM_MAX_OFFSET_FRACTION)))
    steps = min(maximum, int(round(abs(float(offset_fraction)) * (samples - 1))))
    if steps == 0:
        return {name: np.asarray(curves[name], dtype=float).copy() for name in CURVE_NAMES}
    if steps >= min(len(wall_a), len(wall_b)) - 2:
        raise MatcherError("Rim seam candidate leaves a wall too short")
    if offset_fraction > 0:
        shifted_a = wall_a[steps:]
        shifted_b = np.vstack((wall_a[steps::-1], wall_b[1:]))
    else:
        shifted_b = wall_b[steps:]
        shifted_a = np.vstack((wall_b[steps::-1], wall_a[1:]))
    return {
        "wall_a": _resample_curve(shifted_a, samples),
        "wall_b": _resample_curve(shifted_b, samples),
    }


def _split_continuous_boundary(
    points: np.ndarray,
    seam_fraction: float,
    offset_fraction: float,
    samples: int,
) -> dict[str, np.ndarray]:
    """Split one fracture-to-rim-to-fracture boundary before resampling."""
    points = np.asarray(points, dtype=float)
    if (
        points.ndim != 2
        or points.shape[1:] != (2,)
        or len(points) < 4
        or not np.all(np.isfinite(points))
    ):
        raise MatcherError("Query master boundary is invalid")
    keep = np.concatenate(
        (
            np.array([True]),
            np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-10,
        )
    )
    points = points[keep]
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate((np.array([0.0]), np.cumsum(segment_lengths)))
    total = float(cumulative[-1])
    if len(points) < 4 or total <= 1e-10:
        raise MatcherError("Query master boundary is too short")

    nominal = float(np.clip(seam_fraction, 0.0, 1.0)) * total
    available = min(nominal, total - nominal)
    if available <= total * 0.02:
        raise MatcherError("Query rim hint leaves one wall too short")
    split_distance = nominal - float(offset_fraction) * available
    guard = max(total * 0.01, total / max(len(points) - 1, 1))
    split_distance = float(np.clip(split_distance, guard, total - guard))

    upper = int(np.searchsorted(cumulative, split_distance, side="right"))
    upper = min(max(upper, 1), len(points) - 1)
    lower = upper - 1
    span = max(float(cumulative[upper] - cumulative[lower]), 1e-12)
    weight = float((split_distance - cumulative[lower]) / span)
    split_point = points[lower] * (1.0 - weight) + points[upper] * weight
    left = np.vstack((points[:upper], split_point))
    right = np.vstack((split_point, points[upper:]))
    if min(len(left), len(right)) < 3:
        raise MatcherError("Rim seam candidate leaves a wall too short")
    return {
        "wall_a": _resample_curve(left[::-1], samples),
        "wall_b": _resample_curve(right, samples),
    }


def _query_curves_for_rim_seam(
    artifact: dict[str, Any],
    offset_fraction: float,
    samples: int,
) -> dict[str, np.ndarray]:
    """Cut a query seam from its continuous master boundary, then resample."""
    master = artifact.get("query_master_boundary")
    if not isinstance(master, dict):
        return _shift_rim_seam(
            _normalised_curves(artifact, samples),
            offset_fraction,
            samples,
        )
    try:
        points = np.asarray(master["points"], dtype=float)
        seam_fraction = float(master["nominal_seam_fraction"])
    except (KeyError, TypeError, ValueError):
        return _shift_rim_seam(
            _normalised_curves(artifact, samples),
            offset_fraction,
            samples,
        )
    return _split_continuous_boundary(
        points, seam_fraction, offset_fraction, samples
    )


def _reference_curves_for_rim_split(
    artifact: dict[str, Any],
    offset_fraction: float,
    samples: int,
    *,
    reverse: bool = False,
    swap_walls: bool = False,
) -> dict[str, np.ndarray]:
    """Create one of the three reference rim splits from a shared boundary."""
    master = artifact.get("reference_master_boundary")
    if isinstance(master, dict):
        try:
            points = np.asarray(master["points"], dtype=float)
            seam_fraction = float(master["nominal_seam_fraction"])
            curves = _split_continuous_boundary(
                points, seam_fraction, offset_fraction, samples
            )
        except (KeyError, TypeError, ValueError):
            master = None
    if not isinstance(master, dict):
        # Existing contour libraries remain usable. Reconstruct one continuous
        # boundary from their stored 96-point walls, then cut A/B/C from it.
        base = _normalised_curves(
            artifact, max(samples, int(artifact.get("sample_count", 96)))
        )
        wall_a = base["wall_a"]
        wall_b = base["wall_b"]
        points = np.vstack((wall_a[::-1], wall_b[1:]))
        length_a = float(
            np.linalg.norm(np.diff(wall_a, axis=0), axis=1).sum()
        )
        length_b = float(
            np.linalg.norm(np.diff(wall_b, axis=0), axis=1).sum()
        )
        seam_fraction = length_a / max(length_a + length_b, 1e-12)
        curves = _split_continuous_boundary(
            points, seam_fraction, offset_fraction, samples
        )
    if swap_walls:
        curves = {
            "wall_a": curves["wall_b"].copy(),
            "wall_b": curves["wall_a"].copy(),
        }
    if reverse:
        curves = {
            name: points[::-1].copy() for name, points in curves.items()
        }
    return curves


def _rim_geometry_limit(curve: np.ndarray) -> float:
    """Estimate where a curved rim cap settles into a sustained wall tangent."""
    points = np.asarray(curve, dtype=float)
    samples = len(points)
    if samples < 12:
        return RIM_SEAM_INITIAL_OFFSET_FRACTION
    smoothed = np.column_stack(
        (
            gaussian_filter1d(points[:, 0], 1.25, mode="nearest"),
            gaussian_filter1d(points[:, 1], 1.25, mode="nearest"),
        )
    )
    tangent = np.gradient(smoothed, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-8)
    body_start = max(3, int(round(samples * 0.30)))
    body_end = max(body_start + 3, int(round(samples * 0.60)))
    body = np.median(tangent[body_start:body_end], axis=0)
    body /= max(float(np.linalg.norm(body)), 1e-8)
    agreement = np.abs(tangent @ body)
    threshold = math.cos(math.radians(25.0))
    run = max(3, int(round(samples * 0.04)))
    initial_index = int(round((samples - 1) * RIM_SEAM_INITIAL_OFFSET_FRACTION))
    maximum_index = int(round((samples - 1) * RIM_SEAM_MAX_OFFSET_FRACTION))
    for index in range(initial_index, min(maximum_index + 1, samples - run)):
        if np.all(agreement[index : index + run] >= threshold):
            return float(
                np.clip(
                    index / max(samples - 1, 1),
                    RIM_SEAM_INITIAL_OFFSET_FRACTION,
                    RIM_SEAM_MAX_OFFSET_FRACTION,
                )
            )
    return RIM_SEAM_MAX_OFFSET_FRACTION


def _centreline_confidence(
    artifact: dict[str, Any], samples: int, *, reverse: bool = False
) -> np.ndarray:
    values = np.asarray(
        artifact.get(
            "centreline_confidence",
            np.ones(int(artifact.get("sample_count", samples)), dtype=float),
        ),
        dtype=float,
    )
    if len(values) < 2:
        result = np.ones(samples, dtype=float)
    else:
        result = np.interp(
            np.linspace(0.0, 1.0, samples),
            np.linspace(0.0, 1.0, len(values)),
            values,
        )
    result = np.clip(result, 0.0, 1.0)
    return result[::-1].copy() if reverse else result


def _query_point_mass(centre_confidence: np.ndarray) -> np.ndarray:
    samples = len(centre_confidence)
    walls = np.full(samples * 2, 0.4 / samples, dtype=float)
    confidence = np.asarray(centre_confidence, dtype=float)
    centre = (
        0.2 * confidence / confidence.sum()
        if float(confidence.sum()) > 1e-8
        else np.zeros(samples, dtype=float)
    )
    mass = np.concatenate((walls, centre))
    return mass / max(float(mass.sum()), 1e-12)


def _curve_descriptors(curves: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    thickness = np.linalg.norm(curves["wall_a"] - curves["wall_b"], axis=1)
    thickness_scale = max(float(np.median(thickness)), 1e-8)
    wall_tangents = {}
    for name in CURVE_NAMES:
        tangent = np.gradient(curves[name], axis=0)
        wall_tangents[name] = tangent / np.maximum(
            np.linalg.norm(tangent, axis=1, keepdims=True), 1e-8
        )
    tangent_cos = np.sum(
        wall_tangents["wall_a"] * wall_tangents["wall_b"], axis=1
    )
    tangent_sin = (
        wall_tangents["wall_a"][:, 0] * wall_tangents["wall_b"][:, 1]
        - wall_tangents["wall_a"][:, 1] * wall_tangents["wall_b"][:, 0]
    )
    points: list[np.ndarray] = []
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for curve_index, name in enumerate(CURVE_NAMES):
        curve = curves[name]
        radius = thickness / thickness_scale
        curvature = _persistent_curvature_features(curve)
        one_hot = np.zeros((len(curve), len(CURVE_NAMES)), dtype=float)
        one_hot[:, curve_index] = 1.0
        feature = np.column_stack((radius, tangent_cos, tangent_sin, curvature, one_hot))
        points.append(curve)
        features.append(feature)
        labels.append(np.full(len(curve), curve_index, dtype=int))
    return np.vstack(points), np.vstack(features), np.concatenate(labels)


def _feature_cost(
    query_features: np.ndarray,
    reference_features: np.ndarray,
    query_labels: np.ndarray,
    reference_labels: np.ndarray,
) -> np.ndarray:
    radius = cdist(query_features[:, :1], reference_features[:, :1], metric="sqeuclidean")
    tangent = cdist(query_features[:, 1:3], reference_features[:, 1:3], metric="sqeuclidean")
    curvature = cdist(query_features[:, 3:6], reference_features[:, 3:6], metric="sqeuclidean") / 3.0
    mismatch = (query_labels[:, None] != reference_labels[None, :]).astype(float) * 8.0
    return (
        DESCRIPTOR_WEIGHTS["radius"] * radius
        + DESCRIPTOR_WEIGHTS["tangent"] * tangent
        + DESCRIPTOR_WEIGHTS["curvature"] * curvature
        + mismatch
    )


def _structure(
    points: np.ndarray, point_weights: np.ndarray | None = None
) -> np.ndarray:
    matrix = cdist(points, points)
    if point_weights is not None:
        active = np.asarray(point_weights, dtype=float) > 1e-10
        measured = matrix[np.ix_(active, active)]
    else:
        measured = matrix
    scale = float(np.quantile(measured, 0.9))
    return matrix / max(scale, 1e-8)


def _gw_structural_cost(
    query_structure: np.ndarray,
    reference_structure: np.ndarray,
    transport: np.ndarray,
) -> float:
    query_mass = transport.sum(axis=1)
    reference_mass = transport.sum(axis=0)
    first = float(query_mass @ (query_structure**2) @ query_mass)
    second = float(reference_mass @ (reference_structure**2) @ reference_mass)
    cross = float(np.sum((query_structure @ transport @ reference_structure.T) * transport))
    return max(0.0, first + second - 2.0 * cross)


def _srfgw(
    feature_cost: np.ndarray,
    query_structure: np.ndarray,
    reference_structure: np.ndarray,
    query_labels: np.ndarray,
    reference_labels: np.ndarray,
    query_mass: np.ndarray | None = None,
) -> tuple[np.ndarray, float, list[str]]:
    try:
        import ot
    except Exception as exc:
        raise MatcherError(
            "POT is required for matching but is unavailable in the running "
            "SherdScope process. Install POT>=0.9.6,<0.10 and restart SherdScope."
        ) from exc
    query_mass = (
        np.asarray(query_mass, dtype=float)
        if query_mass is not None
        else np.full(len(query_structure), 1.0 / len(query_structure), dtype=float)
    )
    query_mass /= max(float(query_mass.sum()), 1e-12)
    try:
        transport, log = ot.gromov.semirelaxed_fused_gromov_wasserstein(
            feature_cost,
            query_structure,
            reference_structure,
            p=query_mass,
            loss_fun="square_loss",
            alpha=FGW_ALPHA,
            log=True,
            max_iter=250,
            tol_rel=1e-7,
            tol_abs=1e-7,
        )
    except TypeError:
        transport, log = ot.gromov.semirelaxed_fused_gromov_wasserstein(
            feature_cost,
            query_structure,
            reference_structure,
            query_mass,
            "square_loss",
            alpha=FGW_ALPHA,
            log=True,
            max_iter=250,
        )
    distance = float(
        log.get("srfgw_dist", log.get("fgw_dist", np.sum(transport * feature_cost)))
    )
    transport = np.asarray(transport, dtype=float)
    if transport.shape != feature_cost.shape or not np.all(np.isfinite(transport)):
        raise MatcherError("POT returned an invalid transport matrix")
    warnings = [] if bool(log.get("converged", True)) else ["srfgw_not_converged"]
    return transport, distance, warnings


def _shared_monotonic_transport(
    soft_transport: np.ndarray,
    query_labels: np.ndarray,
    reference_labels: np.ndarray,
    query_mass: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Project POT's soft map onto one ordered progress map shared by both walls."""
    query_by_wall = [
        np.flatnonzero(query_labels == wall) for wall in range(len(CURVE_NAMES))
    ]
    reference_by_wall = [
        np.flatnonzero(reference_labels == wall) for wall in range(len(CURVE_NAMES))
    ]
    query_count = min((len(indices) for indices in query_by_wall), default=0)
    if query_count < 2 or any(len(indices) < 2 for indices in reference_by_wall):
        raise MatcherError("Transport projection requires two usable ordered walls")

    wall_progress = []
    for wall, q_indices in enumerate(query_by_wall):
        q_indices = q_indices[:query_count]
        r_indices = reference_by_wall[wall]
        block = soft_transport[np.ix_(q_indices, r_indices)]
        row_mass = block.sum(axis=1)
        positions = np.linspace(0.0, 1.0, len(r_indices))
        barycentric = block @ positions / np.maximum(row_mass, 1e-12)
        missing = row_mass <= 1e-12
        if np.any(missing):
            barycentric[missing] = np.linspace(0.0, 1.0, query_count)[missing]
        wall_progress.append(np.clip(barycentric, 0.0, 1.0))

    # Both walls describe the same physical progress from rim to fracture.
    # Averaging before isotonic projection prevents either wall from jumping
    # independently to an unrelated reference feature.
    shared = _monotonic(np.mean(np.vstack(wall_progress), axis=0))
    uniform = np.linspace(0.0, 1.0, query_count)
    shared = _monotonic(0.75 * shared + 0.25 * uniform)
    span = float(shared[-1] - shared[0])
    if span > 1e-8:
        shared = (shared - shared[0]) / span
    else:
        shared = uniform
    shared[0] = 0.0
    shared[-1] = 1.0

    projected = _transport_from_shared_progress(
        shared,
        query_labels,
        reference_labels,
        query_mass,
        shape=soft_transport.shape,
    )

    row_mass = projected.sum(axis=1)
    covered = row_mass > 1e-12
    backward_steps = int(np.sum(np.diff(shared) < -1e-12))
    stationary_steps = int(np.sum(np.diff(shared) <= 1e-6))
    coverage = float(min(1.0, max(0.0, query_mass[covered].sum())))
    if abs(coverage - 1.0) < 1e-10:
        coverage = 1.0
    return projected, {
        "query_mass_coverage": coverage,
        "shared_progress_span": float(shared[-1] - shared[0]),
        "backward_steps": backward_steps,
        "stationary_steps": stationary_steps,
        "ordered": backward_steps == 0,
        "shared_across_walls": True,
        "progress": shared.tolist(),
    }


def _transport_from_shared_progress(
    progress: np.ndarray | list[float],
    query_labels: np.ndarray,
    reference_labels: np.ndarray,
    query_mass: np.ndarray,
    *,
    shape: tuple[int, int],
) -> np.ndarray:
    """Build a mass-preserving two-wall transport from shared ordered progress."""
    progress = np.asarray(progress, dtype=float)
    projected = np.zeros(shape, dtype=float)
    for wall in range(len(CURVE_NAMES)):
        q_indices = np.flatnonzero(query_labels == wall)
        r_indices = np.flatnonzero(reference_labels == wall)
        if len(q_indices) != len(progress) or len(r_indices) < 2:
            raise MatcherError("Shared progress does not match the two wall curves")
        for position, query_index in zip(progress, q_indices):
            floating = float(np.clip(position, 0.0, 1.0)) * (len(r_indices) - 1)
            left = int(math.floor(floating))
            right = min(left + 1, len(r_indices) - 1)
            right_weight = floating - left
            mass = float(query_mass[query_index])
            projected[query_index, r_indices[left]] += mass * (1.0 - right_weight)
            projected[query_index, r_indices[right]] += mass * right_weight
    return projected


def _refined_intervals_from_soft_transport(
    soft_transport: np.ndarray,
    query_labels: np.ndarray,
    reference_labels: np.ndarray,
    intervals: dict[str, tuple[int, int]],
    full_samples: int,
) -> dict[str, tuple[int, int]]:
    """Shrink a provisional interval to POT's shared robust support."""
    progress_maps = []
    for wall in range(len(CURVE_NAMES)):
        q_indices = np.flatnonzero(query_labels == wall)
        r_indices = np.flatnonzero(reference_labels == wall)
        if len(q_indices) < 2 or len(r_indices) < 2:
            return intervals
        block = soft_transport[np.ix_(q_indices, r_indices)]
        row_mass = block.sum(axis=1)
        positions = np.linspace(0.0, 1.0, len(r_indices))
        barycentric = block @ positions / np.maximum(row_mass, 1e-12)
        progress_maps.append(np.clip(barycentric, 0.0, 1.0))
    shared = _monotonic(np.mean(np.vstack(progress_maps), axis=0))
    low = max(0.0, float(np.quantile(shared, 0.02)) - 0.03)
    high = min(1.0, float(np.quantile(shared, 0.98)) + 0.03)
    if high - low < MIN_REFERENCE_INTERVAL_FRACTION:
        return intervals

    minimum_width = max(6, int(round(full_samples * MIN_REFERENCE_INTERVAL_FRACTION)))
    rim_tolerance = max(1, int(round(full_samples * RIM_ANCHOR_TOLERANCE_FRACTION)))
    refined = {}
    for name in CURVE_NAMES:
        lower, upper = intervals[name]
        width = upper - lower
        proposed_lower = max(0, lower + int(math.floor(low * max(width - 1, 1))))
        proposed_upper = min(
            full_samples,
            lower + int(math.ceil(high * max(width - 1, 1))) + 1,
        )
        # Matching remains anchored to the user-specified rim end. Do not let a
        # transport solution silently move to a body-only interior patch.
        proposed_lower = min(proposed_lower, rim_tolerance)
        if proposed_upper - proposed_lower < minimum_width:
            proposed_upper = min(full_samples, proposed_lower + minimum_width)
        refined[name] = (proposed_lower, proposed_upper)
    return refined


def _monotonic(values: np.ndarray) -> np.ndarray:
    """Small dependency-free isotonic projection for non-decreasing indices."""
    result = np.asarray(values, dtype=float).copy()
    blocks = [[index, index, result[index], 1.0] for index in range(len(result))]
    cursor = 0
    while cursor < len(blocks) - 1:
        if blocks[cursor][2] <= blocks[cursor + 1][2]:
            cursor += 1
            continue
        left, right = blocks[cursor], blocks[cursor + 1]
        weight = left[3] + right[3]
        merged = [left[0], right[1], (left[2] * left[3] + right[2] * right[3]) / weight, weight]
        blocks[cursor : cursor + 2] = [merged]
        cursor = max(0, cursor - 1)
    for start, end, value, _ in blocks:
        result[int(start) : int(end) + 1] = value
    return result


def _transport_interval(
    transport: np.ndarray,
    query_samples: int,
    reference_samples: int | None = None,
) -> tuple[int, int, np.ndarray]:
    reference_samples = reference_samples or query_samples
    mapped = []
    for curve_index in range(len(CURVE_NAMES)):
        q_slice = slice(
            curve_index * query_samples, (curve_index + 1) * query_samples
        )
        r_slice = slice(
            curve_index * reference_samples,
            (curve_index + 1) * reference_samples,
        )
        block = transport[q_slice, r_slice]
        row_mass = block.sum(axis=1)
        barycentric = (
            block @ np.arange(reference_samples, dtype=float)
            / np.maximum(row_mass, 1e-12)
        )
        mapped.append(_monotonic(barycentric))
    mapped_array = np.vstack(mapped)
    lower = int(max(0, math.floor(np.quantile(mapped_array, 0.02)) - 2))
    upper = int(
        min(reference_samples, math.ceil(np.quantile(mapped_array, 0.98)) + 3)
    )
    if upper - lower < 5:
        midpoint = int(round(float(np.median(mapped_array))))
        lower = max(0, midpoint - 3)
        upper = min(reference_samples, lower + 6)
    return lower, upper, mapped_array


def _restrict_reference(
    points: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    samples: int,
    lower: int,
    upper: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    indices = np.concatenate(
        [
            np.arange(curve * samples + lower, curve * samples + upper)
            for curve in range(len(CURVE_NAMES))
        ]
    )
    return points[indices], features[indices], labels[indices], indices


def _weighted_similarity(source: np.ndarray, target: np.ndarray, weights: np.ndarray):
    weights = np.asarray(weights, dtype=float)
    weights /= max(float(weights.sum()), 1e-12)
    source_mean = np.sum(source * weights[:, None], axis=0)
    target_mean = np.sum(target * weights[:, None], axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = (target_centered * weights[:, None]).T @ source_centered
    u, singular, vt = np.linalg.svd(covariance)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    denominator = float(np.sum(weights * np.sum(source_centered**2, axis=1)))
    scale = float(singular.sum() / max(denominator, 1e-12))
    translation = target_mean - scale * (source_mean @ rotation.T)
    return scale, rotation, translation


def _align(
    query_points: np.ndarray,
    reference_points: np.ndarray,
    query_labels: np.ndarray,
    reference_labels: np.ndarray,
    transport: np.ndarray,
    iterations: int = 8,
) -> dict[str, Any]:
    row_mass = transport.sum(axis=1)
    targets = transport @ reference_points / np.maximum(row_mass[:, None], 1e-12)
    weights = np.maximum(row_mass, 1e-12)
    scale, rotation, translation = _weighted_similarity(query_points, targets, weights)
    previous_rms = float("inf")
    converged = False
    matched_indices = np.zeros(len(query_points), dtype=int)
    for iteration in range(iterations):
        transformed = scale * (query_points @ rotation.T) + translation
        target_points = []
        distances = []
        for index, point in enumerate(transformed):
            candidates = np.flatnonzero(reference_labels == query_labels[index])
            candidate_distances = np.linalg.norm(reference_points[candidates] - point, axis=1)
            nearest_position = int(np.argmin(candidate_distances))
            matched_indices[index] = int(candidates[nearest_position])
            target_points.append(reference_points[matched_indices[index]])
            distances.append(float(candidate_distances[nearest_position]))
        target_points_array = np.asarray(target_points)
        distances_array = np.asarray(distances)
        huber_scale = max(float(np.median(distances_array) * 1.4826), 1e-5)
        robust_weights = np.minimum(1.0, (1.5 * huber_scale) / np.maximum(distances_array, 1e-8))
        scale, rotation, translation = _weighted_similarity(
            query_points, target_points_array, robust_weights
        )
        rms = float(np.sqrt(np.mean(distances_array**2)))
        if abs(previous_rms - rms) < 1e-6:
            converged = True
            break
        previous_rms = rms
    transformed = scale * (query_points @ rotation.T) + translation
    distances = np.linalg.norm(reference_points[matched_indices] - transformed, axis=1)
    reverse_distances = np.min(cdist(reference_points, transformed), axis=1)
    angle = _normalised_angle_degrees(
        math.degrees(math.atan2(float(rotation[1, 0]), float(rotation[0, 0])))
    )
    return {
        "points": transformed,
        "matched_indices": matched_indices,
        "distances": distances,
        "median": float(np.median(distances)),
        "rms": float(np.sqrt(np.mean(distances**2))),
        "p95": float(np.quantile(distances, 0.95)),
        "chamfer": float((np.mean(distances) + np.mean(reverse_distances)) / 2.0),
        "hausdorff95": float(max(np.quantile(distances, 0.95), np.quantile(reverse_distances, 0.95))),
        "scale": float(scale),
        "rotation_degrees": angle,
        "translation": translation.tolist(),
        "iterations": iteration + 1,
        "converged": converged,
    }


def _two_wall_alignment(
    query_curves: dict[str, np.ndarray],
    reference_curves: dict[str, np.ndarray],
    intervals: dict[str, tuple[int, int]],
) -> dict[str, Any]:
    samples = len(query_curves["wall_a"])
    query_points = np.vstack([query_curves[name] for name in CURVE_NAMES])
    target_curves = {
        name: _resample_curve(
            reference_curves[name][intervals[name][0] : intervals[name][1]],
            samples,
        )
        for name in CURVE_NAMES
    }
    target_points = np.vstack([target_curves[name] for name in CURVE_NAMES])
    rim_guard = max(2, int(round(samples * RIM_STABLE_GUARD_FRACTION)))
    wall_weights = np.ones(samples, dtype=float)
    wall_weights[:rim_guard] = np.linspace(0.30, 1.0, rim_guard)
    weights = np.tile(wall_weights, len(CURVE_NAMES))
    weights /= weights.sum()
    procrustes_scale, rotation, _ = _weighted_similarity(
        query_points, target_points, weights
    )
    scale, median_scale, scale_disagreement = _robust_scale_consensus(
        query_points, target_points, samples, procrustes_scale
    )
    source_mean = np.sum(query_points * weights[:, None], axis=0)
    target_mean = np.sum(target_points * weights[:, None], axis=0)
    translation = target_mean - scale * (source_mean @ rotation.T)
    transformed = scale * (query_points @ rotation.T) + translation
    distances = np.linalg.norm(transformed - target_points, axis=1)
    per_curve = {}
    for index, name in enumerate(CURVE_NAMES):
        values = distances[index * samples : (index + 1) * samples]
        per_curve[name] = {
            "median": float(np.median(values)),
            "rms": float(np.sqrt(np.mean(values**2))),
            "p95": float(np.quantile(values, 0.95)),
        }
    query_thickness = np.linalg.norm(
        transformed[:samples] - transformed[samples:], axis=1
    )
    reference_thickness = np.linalg.norm(
        target_points[:samples] - target_points[samples:], axis=1
    )
    thickness_error = np.abs(query_thickness - reference_thickness)
    stable_wall_rms = []
    for index, name in enumerate(CURVE_NAMES):
        values = distances[
            index * samples + rim_guard : (index + 1) * samples
        ]
        stable_wall_rms.append(float(np.sqrt(np.mean(values**2))))
        per_curve[name]["stable_rms"] = stable_wall_rms[-1]
    wall_rms = [per_curve["wall_a"]["rms"], per_curve["wall_b"]["rms"]]
    ribbon_cost = float(
        0.45 * np.mean(stable_wall_rms)
        + 0.35 * max(stable_wall_rms)
        + 0.20 * np.sqrt(np.mean(thickness_error[rim_guard:] ** 2))
    )
    rim_count = max(3, int(round(samples * RIM_REGION_FRACTION)))
    rim_indices = np.concatenate(
        (
            np.arange(0, rim_count),
            np.arange(samples, samples + rim_count),
        )
    )
    query_rim = transformed[rim_indices]
    target_rim = target_points[rim_indices]
    rim_region_cost = float(
        (
            np.mean(np.min(cdist(query_rim, target_rim), axis=1))
            + np.mean(np.min(cdist(target_rim, query_rim), axis=1))
        )
        / 2.0
    )
    reverse_distances = np.min(cdist(target_points, transformed), axis=1)
    angle = _normalised_angle_degrees(
        math.degrees(math.atan2(float(rotation[1, 0]), float(rotation[0, 0])))
    )
    return {
        "points": transformed,
        "target_points": target_points,
        "distances": distances,
        "median": float(np.median(distances)),
        "rms": float(np.sqrt(np.mean(distances**2))),
        "p95": float(np.quantile(distances, 0.95)),
        "chamfer": float((np.mean(distances) + np.mean(reverse_distances)) / 2.0),
        "hausdorff95": float(
            max(np.quantile(distances, 0.95), np.quantile(reverse_distances, 0.95))
        ),
        "scale": float(scale),
        "procrustes_scale": float(procrustes_scale),
        "median_ratio_scale": float(median_scale),
        "scale_log_disagreement": float(scale_disagreement),
        "rotation_degrees": angle,
        "translation": translation.tolist(),
        "iterations": 1,
        "converged": True,
        "per_curve": per_curve,
        "wall_rms_max": float(max(wall_rms)),
        "wall_rms_mean": float(np.mean(wall_rms)),
        "stable_wall_rms_max": float(max(stable_wall_rms)),
        "stable_wall_rms_mean": float(np.mean(stable_wall_rms)),
        "rim_stable_guard_samples": rim_guard,
        "rim_region_samples": rim_count,
        "rim_region_cost": rim_region_cost,
        "thickness_rms": float(np.sqrt(np.mean(thickness_error**2))),
        "ribbon_cost": ribbon_cost,
    }


def _robust_scale_consensus(
    source: np.ndarray,
    target: np.ndarray,
    samples: int,
    procrustes_scale: float,
) -> tuple[float, float, float]:
    """Blend global least-squares scale with robust wall/thickness ratios."""
    ratios = []
    for wall in range(len(CURVE_NAMES)):
        start = wall * samples
        source_steps = np.linalg.norm(
            np.diff(source[start : start + samples], axis=0), axis=1
        )
        target_steps = np.linalg.norm(
            np.diff(target[start : start + samples], axis=0), axis=1
        )
        valid = source_steps > 1e-8
        ratios.extend((target_steps[valid] / source_steps[valid]).tolist())
    source_thickness = np.linalg.norm(
        source[:samples] - source[samples : samples * 2], axis=1
    )
    target_thickness = np.linalg.norm(
        target[:samples] - target[samples : samples * 2], axis=1
    )
    valid_thickness = source_thickness > 1e-8
    ratios.extend(
        (target_thickness[valid_thickness] / source_thickness[valid_thickness]).tolist()
    )
    usable = np.asarray(
        [
            value
            for value in ratios
            if np.isfinite(value)
            and MIN_ALIGNMENT_SCALE <= value <= MAX_ALIGNMENT_SCALE
        ],
        dtype=float,
    )
    median_scale = (
        float(np.median(usable)) if len(usable) else float(procrustes_scale)
    )
    consensus = float(
        math.exp(
            0.65 * math.log(max(procrustes_scale, 1e-12))
            + 0.35 * math.log(max(median_scale, 1e-12))
        )
    )
    disagreement = abs(
        math.log(
            max(procrustes_scale, 1e-12) / max(median_scale, 1e-12)
        )
    )
    return consensus, median_scale, disagreement


def _ordered_transport_alignment(
    query_points: np.ndarray,
    interval_points: np.ndarray,
    transport: np.ndarray,
    samples: int,
    *,
    iterations: int = 6,
    salience_weights: np.ndarray | None = None,
) -> dict[str, Any]:
    """Recover an overlay from the exact ordered correspondence used in scoring."""
    row_mass = transport.sum(axis=1)
    if np.any(row_mass <= 1e-12):
        raise MatcherError("Ordered transport left query points unmatched")
    targets = transport @ interval_points / row_mass[:, None]
    rim_guard = max(2, int(round(samples * RIM_STABLE_GUARD_FRACTION)))
    wall_stability = np.ones(samples, dtype=float)
    wall_stability[:rim_guard] = np.linspace(0.30, 1.0, rim_guard)
    stability_weights = np.tile(wall_stability, len(CURVE_NAMES))
    robust_weights = row_mass * stability_weights
    salience = (
        np.clip(np.asarray(salience_weights, dtype=float), 0.0, 1.0)
        if salience_weights is not None
        else np.zeros(len(query_points), dtype=float)
    )
    if len(salience) != len(query_points):
        raise MatcherError("Salience weights do not match the query points")
    previous_rms = float("inf")
    converged = False
    scale = 1.0
    rotation = np.eye(2)
    translation = np.zeros(2)
    procrustes_scale = 1.0
    median_scale = 1.0
    scale_disagreement = 0.0
    for iteration in range(iterations):
        procrustes_scale, rotation, _ = _weighted_similarity(
            query_points, targets, robust_weights
        )
        scale, median_scale, scale_disagreement = _robust_scale_consensus(
            query_points, targets, samples, procrustes_scale
        )
        if not MIN_ALIGNMENT_SCALE <= scale <= MAX_ALIGNMENT_SCALE:
            raise MatcherError(
                "Alignment scale left the permitted "
                f"{MIN_ALIGNMENT_SCALE:g}–{MAX_ALIGNMENT_SCALE:g} range"
            )
        normalised_weights = robust_weights / max(
            float(robust_weights.sum()), 1e-12
        )
        source_mean = np.sum(
            query_points * normalised_weights[:, None], axis=0
        )
        target_mean = np.sum(
            targets * normalised_weights[:, None], axis=0
        )
        translation = target_mean - scale * (source_mean @ rotation.T)
        transformed = scale * (query_points @ rotation.T) + translation
        distances = np.linalg.norm(transformed - targets, axis=1)
        rms = float(np.sqrt(np.average(distances**2, weights=row_mass)))
        if abs(previous_rms - rms) < 1e-7:
            converged = True
            break
        huber_scale = max(float(np.median(distances) * 1.4826), 1e-5)
        huber_weights = np.minimum(
            1.0, (1.5 * huber_scale) / np.maximum(distances, 1e-8)
        )
        # Persistent bends, lips and shoulders retain at least 45% influence.
        # This prevents a poor initialization from labelling the most
        # diagnostic points as disposable outliers.
        robust_floor = 0.15 + 0.30 * salience
        robust_weights = (
            row_mass
            * stability_weights
            * np.maximum(huber_weights, robust_floor)
        )
        previous_rms = rms

    transformed = scale * (query_points @ rotation.T) + translation
    distances = np.linalg.norm(transformed - targets, axis=1)
    reverse_distances = np.min(cdist(interval_points, transformed), axis=1)
    per_curve = {}
    for index, name in enumerate(CURVE_NAMES):
        values = distances[index * samples : (index + 1) * samples]
        per_curve[name] = {
            "median": float(np.median(values)),
            "rms": float(np.sqrt(np.mean(values**2))),
            "p95": float(np.quantile(values, 0.95)),
        }
    query_thickness = np.linalg.norm(
        transformed[:samples] - transformed[samples : samples * 2], axis=1
    )
    target_thickness = np.linalg.norm(
        targets[:samples] - targets[samples : samples * 2], axis=1
    )
    thickness_error = np.abs(query_thickness - target_thickness)
    stable_wall_rms = []
    for index, name in enumerate(CURVE_NAMES):
        values = distances[
            index * samples + rim_guard : (index + 1) * samples
        ]
        stable_wall_rms.append(float(np.sqrt(np.mean(values**2))))
        per_curve[name]["stable_rms"] = stable_wall_rms[-1]
    wall_rms = [per_curve["wall_a"]["rms"], per_curve["wall_b"]["rms"]]
    stable_thickness_error = thickness_error[rim_guard:]
    ribbon_cost = float(
        0.45 * np.mean(stable_wall_rms)
        + 0.35 * max(stable_wall_rms)
        + 0.20 * np.sqrt(np.mean(stable_thickness_error**2))
    )
    rim_count = max(3, int(round(samples * RIM_REGION_FRACTION)))
    rim_indices = np.concatenate(
        (
            np.arange(0, rim_count),
            np.arange(samples, samples + rim_count),
        )
    )
    query_rim = transformed[rim_indices]
    target_rim = targets[rim_indices]
    rim_forward = np.min(cdist(query_rim, target_rim), axis=1)
    rim_reverse = np.min(cdist(target_rim, query_rim), axis=1)
    rim_region_cost = float(
        (np.mean(rim_forward) + np.mean(rim_reverse)) / 2.0
    )
    angle = _normalised_angle_degrees(
        math.degrees(math.atan2(float(rotation[1, 0]), float(rotation[0, 0])))
    )
    return {
        "points": transformed,
        "target_points": targets,
        "distances": distances,
        "median": float(np.median(distances)),
        "rms": float(np.sqrt(np.mean(distances**2))),
        "p95": float(np.quantile(distances, 0.95)),
        "chamfer": float((np.mean(distances) + np.mean(reverse_distances)) / 2.0),
        "hausdorff95": float(
            max(np.quantile(distances, 0.95), np.quantile(reverse_distances, 0.95))
        ),
        "scale": float(scale),
        "procrustes_scale": float(procrustes_scale),
        "median_ratio_scale": float(median_scale),
        "scale_log_disagreement": float(scale_disagreement),
        "rotation_degrees": angle,
        "translation": translation.tolist(),
        "iterations": iteration + 1,
        "converged": converged,
        "per_curve": per_curve,
        "wall_rms_max": float(max(wall_rms)),
        "wall_rms_mean": float(np.mean(wall_rms)),
        "stable_wall_rms_max": float(max(stable_wall_rms)),
        "stable_wall_rms_mean": float(np.mean(stable_wall_rms)),
        "rim_stable_guard_samples": rim_guard,
        "rim_region_samples": rim_count,
        "rim_region_cost": rim_region_cost,
        "thickness_rms": float(np.sqrt(np.mean(thickness_error**2))),
        "ribbon_cost": ribbon_cost,
    }


def _query_salience_weights(features: np.ndarray, samples: int) -> np.ndarray:
    weights = np.zeros(len(features), dtype=float)
    guard = max(4, int(round(samples * SALIENCE_ENDPOINT_GUARD_FRACTION)))
    for wall in range(len(CURVE_NAMES)):
        start = wall * samples
        strength = np.mean(np.abs(features[start : start + samples, 3:6]), axis=1)
        threshold = max(float(np.quantile(strength, 0.80)), 1e-6)
        values = np.clip(strength / (2.0 * threshold), 0.0, 1.0)
        taper = np.ones(samples, dtype=float)
        if guard:
            ramp = np.linspace(0.2, 1.0, guard, endpoint=False)
            taper[:guard] = ramp
            taper[-guard:] = ramp[::-1]
        weights[start : start + samples] = values * taper
    return weights


def _resample_feature_rows(values: np.ndarray, count: int) -> np.ndarray:
    source = np.asarray(values, dtype=float)
    if len(source) == count:
        return source.copy()
    old = np.linspace(0.0, 1.0, len(source))
    new = np.linspace(0.0, 1.0, count)
    return np.column_stack(
        [np.interp(new, old, source[:, column]) for column in range(source.shape[1])]
    )


def _normalise_cost_matrix(values: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=float), 0.0)
    positive = values[values > 1e-12]
    scale = float(np.quantile(positive, 0.65)) if len(positive) else 1.0
    return np.clip(values / max(scale, 1e-8), 0.0, 12.0)


def _coupled_dtw_progress(
    query_points: np.ndarray,
    interval_points: np.ndarray,
    query_features: np.ndarray,
    interval_features: np.ndarray,
    query_labels: np.ndarray,
    reference_labels: np.ndarray,
    alignment: dict[str, Any],
    current_progress: np.ndarray,
    samples: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Update one shared wall progression under the current similarity transform."""
    angle = math.radians(float(alignment["rotation_degrees"]))
    rotation = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=float,
    )
    transformed = (
        float(alignment["scale"]) * (query_points @ rotation.T)
        + np.asarray(alignment["translation"], dtype=float)
    )
    query_walls = []
    reference_walls = []
    query_feature_walls = []
    reference_feature_walls = []
    for wall in range(len(CURVE_NAMES)):
        q_indices = np.flatnonzero(query_labels == wall)
        r_indices = np.flatnonzero(reference_labels == wall)
        query_walls.append(transformed[q_indices])
        reference_walls.append(_resample_curve(interval_points[r_indices], samples))
        query_feature_walls.append(query_features[q_indices, :6])
        reference_feature_walls.append(
            _resample_feature_rows(interval_features[r_indices, :6], samples)
        )

    geometry = np.zeros((samples, samples), dtype=float)
    feature = np.zeros((samples, samples), dtype=float)
    for wall in range(len(CURVE_NAMES)):
        geometry += cdist(
            query_walls[wall], reference_walls[wall], metric="sqeuclidean"
        )
        feature += (
            cdist(
                query_feature_walls[wall],
                reference_feature_walls[wall],
                metric="sqeuclidean",
            )
            / query_feature_walls[wall].shape[1]
        )
    geometry /= len(CURVE_NAMES)
    feature /= len(CURVE_NAMES)
    query_thickness = np.linalg.norm(query_walls[0] - query_walls[1], axis=1)
    reference_thickness = np.linalg.norm(
        reference_walls[0] - reference_walls[1], axis=1
    )
    thickness = (
        query_thickness[:, None] - reference_thickness[None, :]
    ) ** 2
    grid = np.linspace(0.0, 1.0, samples)
    prior = (grid[None, :] - current_progress[:, None]) ** 2
    local_cost = (
        0.52 * _normalise_cost_matrix(geometry)
        + 0.25 * _normalise_cost_matrix(feature)
        + 0.18 * _normalise_cost_matrix(thickness)
        + 0.05 * _normalise_cost_matrix(prior)
    )

    band = max(6, int(round(samples * 0.18)))
    dp = np.full((samples, samples), np.inf, dtype=float)
    parent = np.full((samples, samples, 2), -1, dtype=int)
    dp[0, 0] = local_cost[0, 0]
    warp_penalty = 0.025
    for i in range(samples):
        centre = int(round(float(current_progress[i]) * (samples - 1)))
        lower = max(0, centre - band)
        upper = min(samples, centre + band + 1)
        if i == 0:
            lower = 0
        if i == samples - 1:
            upper = samples
        for j in range(lower, upper):
            if i == 0 and j == 0:
                continue
            choices = []
            if i > 0 and j > 0:
                choices.append((dp[i - 1, j - 1], i - 1, j - 1))
            if i > 0:
                choices.append((dp[i - 1, j] + warp_penalty, i - 1, j))
            if j > 0:
                choices.append((dp[i, j - 1] + warp_penalty, i, j - 1))
            best = min(choices, key=lambda item: item[0])
            if np.isfinite(best[0]):
                dp[i, j] = best[0] + local_cost[i, j]
                parent[i, j] = (best[1], best[2])
    if not np.isfinite(dp[-1, -1]):
        return current_progress.copy(), {
            "objective": None,
            "path_length": 0,
            "updated": False,
            "warning": "joint_dtw_path_unavailable",
        }

    path = []
    i = samples - 1
    j = samples - 1
    while i >= 0 and j >= 0:
        path.append((i, j))
        if i == 0 and j == 0:
            break
        next_i, next_j = parent[i, j]
        if next_i < 0 or next_j < 0:
            return current_progress.copy(), {
                "objective": None,
                "path_length": len(path),
                "updated": False,
                "warning": "joint_dtw_backtrack_failed",
            }
        i, j = int(next_i), int(next_j)
    path.reverse()
    mapped = np.zeros(samples, dtype=float)
    for query_index in range(samples):
        positions = [ref_index for q_index, ref_index in path if q_index == query_index]
        mapped[query_index] = (
            float(np.mean(positions)) / max(samples - 1, 1)
            if positions
            else current_progress[query_index]
        )
    mapped = _monotonic(np.clip(mapped, 0.0, 1.0))
    mapped[0] = 0.0
    mapped[-1] = 1.0
    return mapped, {
        "objective": float(dp[-1, -1] / max(len(path), 1)),
        "path_length": len(path),
        "updated": True,
        "warning": None,
    }


def _joint_alignment_objective(
    fgw_cost: float, alignment: dict[str, Any]
) -> float:
    return float(
        0.40 * _bounded_score(fgw_cost, SCORE_SCALES["fgw"])
        + 0.35
        * _bounded_score(alignment["ribbon_cost"], SCORE_SCALES["ribbon"])
        + 0.15
        * _bounded_score(
            alignment["hausdorff95"], SCORE_SCALES["alignment_tail"]
        )
        + 0.10
        * _bounded_score(
            alignment["rim_region_cost"], SCORE_SCALES["rim_region"]
        )
    )


def _joint_correspondence_alignment(
    initial_transport: np.ndarray,
    initial_progress: np.ndarray,
    query_points: np.ndarray,
    interval_points: np.ndarray,
    query_features: np.ndarray,
    interval_features: np.ndarray,
    query_labels: np.ndarray,
    reference_labels: np.ndarray,
    query_mass: np.ndarray,
    query_structure: np.ndarray,
    interval_structure: np.ndarray,
    interval_feature_cost: np.ndarray,
    samples: int,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any], float, float, float]:
    """Alternate shared ordered correspondence and robust similarity alignment."""
    salience_weights = _query_salience_weights(query_features, samples)
    transport = initial_transport
    progress = np.asarray(initial_progress, dtype=float)
    feature_cost = float(np.sum(transport * interval_feature_cost))
    structural_cost = _gw_structural_cost(
        query_structure, interval_structure, transport
    )
    fgw_cost = float(
        (1.0 - FGW_ALPHA) * feature_cost + FGW_ALPHA * structural_cost
    )
    alignment = _ordered_transport_alignment(
        query_points,
        interval_points,
        transport,
        samples,
        salience_weights=salience_weights,
    )
    best = (
        _joint_alignment_objective(fgw_cost, alignment),
        transport,
        progress,
        alignment,
        fgw_cost,
        feature_cost,
        structural_cost,
    )
    best_iteration = 0
    history = [
        {
            "iteration": 0,
            "objective": best[0],
            "fgw_cost": fgw_cost,
            "ribbon_cost": alignment["ribbon_cost"],
            "rotation_degrees": alignment["rotation_degrees"],
            "scale": alignment["scale"],
            "progress_delta_samples": 0.0,
        }
    ]
    converged = False
    warnings = []
    for iteration in range(1, MAX_JOINT_ALIGNMENT_ITERATIONS + 1):
        updated_progress, dtw = _coupled_dtw_progress(
            query_points,
            interval_points,
            query_features,
            interval_features,
            query_labels,
            reference_labels,
            alignment,
            progress,
            samples,
        )
        if dtw.get("warning"):
            warnings.append(str(dtw["warning"]))
        if not dtw["updated"]:
            break
        updated_transport = _transport_from_shared_progress(
            updated_progress,
            query_labels,
            reference_labels,
            query_mass,
            shape=initial_transport.shape,
        )
        updated_feature = float(
            np.sum(updated_transport * interval_feature_cost)
        )
        updated_structural = _gw_structural_cost(
            query_structure, interval_structure, updated_transport
        )
        updated_fgw = float(
            (1.0 - FGW_ALPHA) * updated_feature
            + FGW_ALPHA * updated_structural
        )
        try:
            updated_alignment = _ordered_transport_alignment(
                query_points,
                interval_points,
                updated_transport,
                samples,
                salience_weights=salience_weights,
            )
        except MatcherError as exc:
            warnings.append(str(exc))
            break
        if not _alignment_allowed(updated_alignment):
            warnings.append("joint_rotation_cap_reached")
            break
        objective = _joint_alignment_objective(
            updated_fgw, updated_alignment
        )
        progress_delta = float(
            np.max(np.abs(updated_progress - progress)) * (samples - 1)
        )
        rotation_delta = abs(
            _normalised_angle_degrees(
                updated_alignment["rotation_degrees"]
                - alignment["rotation_degrees"]
            )
        )
        scale_delta = abs(
            math.log(
                max(updated_alignment["scale"], 1e-12)
                / max(alignment["scale"], 1e-12)
            )
        )
        translation_delta = float(
            np.linalg.norm(
                np.asarray(updated_alignment["translation"])
                - np.asarray(alignment["translation"])
            )
        )
        history.append(
            {
                "iteration": iteration,
                "objective": objective,
                "fgw_cost": updated_fgw,
                "ribbon_cost": updated_alignment["ribbon_cost"],
                "rotation_degrees": updated_alignment["rotation_degrees"],
                "scale": updated_alignment["scale"],
                "progress_delta_samples": progress_delta,
                "rotation_delta_degrees": rotation_delta,
                "scale_log_delta": scale_delta,
                "translation_delta": translation_delta,
                "dtw_objective": dtw["objective"],
            }
        )
        # Retain the best joint state. A tiny tolerance avoids rejecting a
        # geometrically equivalent update due to floating-point noise.
        if objective <= best[0] + 1e-8:
            best = (
                objective,
                updated_transport,
                updated_progress,
                updated_alignment,
                updated_fgw,
                updated_feature,
                updated_structural,
            )
            best_iteration = iteration
        elif objective > best[0] * 1.02:
            warnings.append("joint_update_rejected")
            break

        transport = updated_transport
        progress = updated_progress
        alignment = updated_alignment
        if (
            progress_delta <= JOINT_PROGRESS_TOLERANCE
            and rotation_delta <= JOINT_ROTATION_TOLERANCE_DEGREES
            and scale_delta <= JOINT_SCALE_TOLERANCE
            and translation_delta <= 1e-4
        ):
            converged = True
            break

    (
        best_objective,
        best_transport,
        best_progress,
        best_alignment,
        best_fgw,
        best_feature,
        best_structural,
    ) = best
    diagnostics = {
        "iterations": len(history) - 1,
        "converged": bool(
            converged and best_iteration == len(history) - 1
        ),
        "terminal_converged": converged,
        "best_iteration": best_iteration,
        "returned_best_differs_from_terminal": bool(
            best_iteration != len(history) - 1
        ),
        "objective": best_objective,
        "history": history,
        "warnings": sorted(set(warnings)),
        "progress": best_progress.tolist(),
    }
    return (
        best_transport,
        best_alignment,
        diagnostics,
        best_fgw,
        best_feature,
        best_structural,
    )


def _select_two_wall_intervals(
    query_curves: dict[str, np.ndarray],
    reference_curves: dict[str, np.ndarray],
    *,
    selection_strategy: str = "ribbon",
) -> tuple[dict[str, tuple[int, int]], dict[str, Any]]:
    """Select coupled wall intervals while allowing a diagonal fracture skew."""
    samples = len(reference_curves["wall_a"])
    minimum_width = max(6, int(round(samples * MIN_REFERENCE_INTERVAL_FRACTION)))
    rim_tolerance = max(1, int(round(samples * RIM_ANCHOR_TOLERANCE_FRACTION)))
    stride = 3 if samples <= 48 else 4
    skew = max(stride, int(round(samples * 0.08)))
    shifts = (-skew, 0, skew)
    widths = list(range(minimum_width, samples + 1, stride))
    if widths[-1] != samples:
        widths.append(samples)

    def alignment_key(
        alignment: dict[str, Any],
        candidate_intervals: dict[str, tuple[int, int]],
    ) -> tuple[float, float, int, int]:
        coverage = float(
            np.mean(
                [
                    (upper - lower) / samples
                    for lower, upper in candidate_intervals.values()
                ]
            )
        )
        ribbon = _bounded_score(
            alignment["ribbon_cost"], SCORE_SCALES["ribbon"]
        )
        rim = _bounded_score(
            alignment["rim_region_cost"], SCORE_SCALES["rim_region"]
        )
        scale = _bounded_score(
            alignment["scale_log_disagreement"], 0.15
        )
        rotation = min(
            1.0,
            abs(alignment["rotation_degrees"])
            / MAX_ALIGNMENT_ROTATION_DEGREES,
        )
        composite = (
            0.55 * ribbon + 0.20 * rim + 0.20 * scale + 0.05 * rotation
        )
        if selection_strategy == "transform":
            score = 0.55 * ribbon + 0.35 * scale + 0.10 * rotation
        elif selection_strategy == "broad":
            score = composite + 0.08 * (1.0 - coverage)
        elif selection_strategy == "composite":
            score = composite
        else:
            score = float(alignment["ribbon_cost"])
        first_lower = candidate_intervals["wall_a"][0]
        second_lower = candidate_intervals["wall_b"][0]
        return (float(score), -coverage, first_lower, second_lower)

    best_shared = None
    for width in widths:
        lowers = list(range(0, min(rim_tolerance, samples - width) + 1, stride))
        if not lowers:
            lowers = [0]
        for lower_a in lowers:
            upper_a = lower_a + width
            intervals = {
                "wall_a": (lower_a, upper_a),
                "wall_b": (lower_a, upper_a),
            }
            alignment = _two_wall_alignment(
                query_curves, reference_curves, intervals
            )
            if not _alignment_allowed(alignment):
                continue
            key = alignment_key(alignment, intervals)
            if best_shared is None or key < best_shared[0]:
                best_shared = (key, intervals, alignment)
    if best_shared is None:
        raise MatcherError("No coupled wall intervals could be evaluated")

    lower_a, upper_a = best_shared[1]["wall_a"]
    best = best_shared
    for start_shift in shifts:
        for end_shift in shifts:
            lower_b = max(0, lower_a + start_shift)
            upper_b = min(samples, upper_a + end_shift)
            if upper_b - lower_b < minimum_width:
                continue
            if lower_b > rim_tolerance:
                continue
            intervals = {
                "wall_a": (lower_a, upper_a),
                "wall_b": (lower_b, upper_b),
            }
            alignment = _two_wall_alignment(
                query_curves, reference_curves, intervals
            )
            if not _alignment_allowed(alignment):
                continue
            key = alignment_key(alignment, intervals)
            if key < best[0]:
                best = (key, intervals, alignment)
    return best[1], best[2]


def _restrict_wall_intervals(
    points: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    samples: int,
    intervals: dict[str, tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    indices = np.concatenate(
        [
            np.arange(
                curve_index * samples + intervals[name][0],
                curve_index * samples + intervals[name][1],
            )
            for curve_index, name in enumerate(CURVE_NAMES)
        ]
    )
    return points[indices], features[indices], labels[indices], indices


def _salience_penalty(
    query_features: np.ndarray,
    reference_features: np.ndarray,
    query_artifact: dict[str, Any],
    reference_artifact: dict[str, Any],
    samples: int,
    intervals: dict[str, tuple[int, int]],
    aligned_distances: np.ndarray | list[float] | None = None,
    shared_progress: np.ndarray | list[float] | None = None,
    reverse: bool = False,
    swap_walls: bool = False,
) -> dict[str, Any]:
    """Compare persistent curvature locally and symmetrically on ordered walls."""
    endpoint_guard = max(
        4, int(round(samples * SALIENCE_ENDPOINT_GUARD_FRACTION))
    )
    local_window = max(
        2, int(round(samples * SALIENCE_LOCAL_WINDOW_FRACTION))
    )
    progress = np.asarray(
        shared_progress
        if shared_progress is not None
        else np.linspace(0.0, 1.0, samples),
        dtype=float,
    )
    if len(progress) != samples:
        progress = np.interp(
            np.linspace(0.0, 1.0, samples),
            np.linspace(0.0, 1.0, len(progress)),
            progress,
        )
    reference_penalties: list[float] = []
    reference_weights: list[float] = []
    query_penalties: list[float] = []
    query_weights: list[float] = []
    salient_alignment_distances: list[float] = []
    salient_alignment_weights: list[float] = []
    aligned_values = (
        np.asarray(aligned_distances, dtype=float)
        if aligned_distances is not None
        else np.empty(0, dtype=float)
    )

    for curve_index, name in enumerate(CURVE_NAMES):
        lower, upper = intervals[name]
        width = upper - lower
        if width < 2:
            continue
        preceding = sum(
            intervals[prior][1] - intervals[prior][0]
            for prior in CURVE_NAMES[:curve_index]
        )
        query_offset = curve_index * samples
        q_curvature = query_features[
            query_offset : query_offset + samples, 3:6
        ]
        r_curvature = reference_features[
            preceding : preceding + width, 3:6
        ]
        q_strength = np.mean(np.abs(q_curvature), axis=1)
        r_strength = np.mean(np.abs(r_curvature), axis=1)
        q_threshold = max(float(np.quantile(q_strength, 0.80)), 1e-6)
        r_threshold = max(float(np.quantile(r_strength, 0.80)), 1e-6)

        # Query-to-reference: every persistent high-curvature query location
        # must agree near the corresponding ordered reference position.
        for query_index in range(endpoint_guard, samples - endpoint_guard):
            relative = float(progress[query_index])
            expected = int(round(relative * (width - 1)))
            if max(q_strength[query_index] / q_threshold, r_strength[expected] / r_threshold) < 1.0:
                continue
            r_lower = max(0, expected - local_window)
            r_upper = min(width, expected + local_window + 1)
            differences = np.linalg.norm(
                r_curvature[r_lower:r_upper] - q_curvature[query_index],
                axis=1,
            )
            query_penalties.append(
                min(2.0, float(np.min(differences)) / 2.0)
            )
            query_weights.append(
                1.0
                + min(
                    3.0,
                    max(
                        q_strength[query_index] / q_threshold,
                        r_strength[expected] / r_threshold,
                    ),
                )
            )
            if len(aligned_values) == len(query_features):
                salient_alignment_distances.append(
                    float(aligned_values[query_offset + query_index])
                )
                salient_alignment_weights.append(query_weights[-1])

        # Reference-to-query catches a diagnostic reference feature that the
        # query lacks. Only the selected interval is examined.
        for reference_index in range(width):
            full_index = lower + reference_index
            if full_index < endpoint_guard or full_index >= samples - endpoint_guard:
                continue
            relative = reference_index / max(width - 1, 1)
            expected = int(np.argmin(np.abs(progress - relative)))
            if max(r_strength[reference_index] / r_threshold, q_strength[expected] / q_threshold) < 1.0:
                continue
            q_lower = max(endpoint_guard, expected - local_window)
            q_upper = min(samples - endpoint_guard, expected + local_window + 1)
            if q_upper <= q_lower:
                continue
            differences = np.linalg.norm(
                q_curvature[q_lower:q_upper] - r_curvature[reference_index],
                axis=1,
            )
            reference_penalties.append(
                min(2.0, float(np.min(differences)) / 2.0)
            )
            reference_weights.append(
                1.0
                + min(
                    3.0,
                    max(
                        r_strength[reference_index] / r_threshold,
                        q_strength[expected] / q_threshold,
                    ),
                )
            )

    reference_mean = (
        float(np.average(reference_penalties, weights=reference_weights))
        if reference_penalties
        else 0.0
    )
    query_mean = (
        float(np.average(query_penalties, weights=query_weights))
        if query_penalties
        else 0.0
    )
    directional = []
    if reference_penalties:
        directional.append(reference_mean)
    if query_penalties:
        directional.append(query_mean)
    curvature_mean = float(np.mean(directional)) if directional else 0.0
    positional_rms = (
        float(
            np.sqrt(
                np.average(
                    np.square(salient_alignment_distances),
                    weights=salient_alignment_weights,
                )
            )
        )
        if salient_alignment_distances
        else 0.0
    )
    positional_penalty = _bounded_score(positional_rms, 0.05)
    return {
        "overall": float(0.5 * curvature_mean + 0.5 * positional_penalty),
        "curvature_correspondence": curvature_mean,
        "salient_alignment_rms": positional_rms,
        "salient_alignment_penalty": positional_penalty,
        "reference_to_query": reference_mean,
        "query_to_reference": query_mean,
        "reference_feature_count": len(reference_penalties),
        "query_feature_count": len(query_penalties),
        "endpoint_guard_samples": endpoint_guard,
        "local_window_samples": local_window,
    }


def _bounded_score(value: float, scale: float) -> float:
    value = max(0.0, float(value))
    return float(value / (value + max(float(scale), 1e-12)))


def _completeness_penalty(matched_reference_fraction: float) -> float:
    """Mildly penalize candidates that explain only a tiny easy interval."""
    fraction = float(np.clip(matched_reference_fraction, 0.0, 1.0))
    return float((1.0 - fraction) ** 2)


def _match_one_single(
    query: dict[str, Any],
    reference: dict[str, Any],
    samples: int,
    *,
    exact_transport: bool = True,
    rim_seam_hint: float | None = None,
    hypothesis_strategy: str = "composite",
) -> dict[str, Any]:
    base_query_curves = _query_curves_for_rim_seam(query, 0.0, samples)
    seam_offsets = _rim_seam_offsets(samples, rim_seam_hint)
    orientation_candidates: list[dict[str, Any]] = []
    evaluated_offsets: set[float] = set()
    reference_variants = []
    for reference_split_label, reference_split_offset in REFERENCE_RIM_SPLITS:
        for reverse in (False, True):
            for swap_walls in (False, True):
                reference_variants.append(
                    {
                        "label": reference_split_label,
                        "offset": reference_split_offset,
                        "reverse": reverse,
                        "swap_walls": swap_walls,
                        "curves": _reference_curves_for_rim_split(
                            reference,
                            reference_split_offset,
                            samples,
                            reverse=reverse,
                            swap_walls=swap_walls,
                        ),
                    }
                )

    def evaluate_offset(seam_offset: float) -> None:
        seam_offset = round(float(seam_offset), 6)
        if seam_offset in evaluated_offsets:
            return
        evaluated_offsets.add(seam_offset)
        query_curves = _query_curves_for_rim_seam(
            query, seam_offset, samples
        )
        for variant in reference_variants:
            reference_split_label = variant["label"]
            reference_split_offset = variant["offset"]
            reverse = variant["reverse"]
            swap_walls = variant["swap_walls"]
            reference_curves = variant["curves"]
            try:
                intervals, ordered_alignment = _select_two_wall_intervals(
                    query_curves,
                    reference_curves,
                    selection_strategy=hypothesis_strategy,
                )
            except MatcherError:
                continue
            coverage = float(
                np.mean(
                    [
                        (upper - lower) / samples
                        for lower, upper in intervals.values()
                    ]
                )
            )
            ribbon_term = _bounded_score(
                ordered_alignment["ribbon_cost"],
                SCORE_SCALES["ribbon"],
            )
            rim_term = _bounded_score(
                ordered_alignment["rim_region_cost"],
                SCORE_SCALES["rim_region"],
            )
            scale_term = _bounded_score(
                ordered_alignment["scale_log_disagreement"], 0.15
            )
            rotation_term = min(
                1.0,
                abs(ordered_alignment["rotation_degrees"])
                / MAX_ALIGNMENT_ROTATION_DEGREES,
            )
            composite = float(
                0.55 * ribbon_term
                + 0.20 * rim_term
                + 0.20 * scale_term
                + 0.05 * rotation_term
            )
            transform_score = float(
                0.55 * ribbon_term
                + 0.35 * scale_term
                + 0.10 * rotation_term
            )
            broad_score = float(composite + 0.08 * (1.0 - coverage))
            orientation_candidates.append(
                {
                    "seam_offset": seam_offset,
                    "reference_split_label": reference_split_label,
                    "reference_split_offset": reference_split_offset,
                    "reverse": reverse,
                    "swap_walls": swap_walls,
                    "intervals": intervals,
                    "query_curves": query_curves,
                    "reference_curves": reference_curves,
                    "alignment": ordered_alignment,
                    "coverage": coverage,
                    "ribbon_score": float(
                        ordered_alignment["ribbon_cost"]
                    ),
                    "composite_score": composite,
                    "transform_score": transform_score,
                    "broad_score": broad_score,
                }
            )

    for offset in seam_offsets:
        evaluate_offset(offset)

    geometry_limits = {
        -1: _rim_geometry_limit(base_query_curves["wall_b"]),
        1: _rim_geometry_limit(base_query_curves["wall_a"]),
    }
    expansion_history: list[dict[str, Any]] = []
    for _ in range(2):
        if not orientation_candidates:
            break
        by_offset = {}
        for offset in sorted(evaluated_offsets):
            values = [
                candidate["composite_score"]
                for candidate in orientation_candidates
                if candidate["seam_offset"] == offset
            ]
            if values:
                by_offset[offset] = min(values)
        if not by_offset:
            break
        ordered_offsets = sorted(by_offset)
        if len(ordered_offsets) < 2:
            break
        best_offset = min(
            ordered_offsets, key=lambda offset: (by_offset[offset], abs(offset))
        )
        direction = 0
        if best_offset == ordered_offsets[0] and best_offset < 0:
            direction = -1
            neighbour = ordered_offsets[1]
        elif best_offset == ordered_offsets[-1] and best_offset > 0:
            direction = 1
            neighbour = ordered_offsets[-2]
        else:
            break
        improving = (
            by_offset[best_offset] <= by_offset[neighbour] * 0.985
            or by_offset[neighbour] - by_offset[best_offset] >= 0.005
        )
        proposed = best_offset + direction * RIM_SEAM_EXPANSION_STEP_FRACTION
        permitted = min(
            RIM_SEAM_MAX_OFFSET_FRACTION, geometry_limits[direction]
        )
        if not improving or abs(proposed) > permitted + 1e-9:
            break
        evaluate_offset(proposed)
        expansion_history.append(
            {
                "from": best_offset,
                "to": round(proposed, 6),
                "direction": direction,
                "geometry_limit": permitted,
            }
        )

    if not orientation_candidates:
        raise MatcherError("No rim-anchored candidate passed the rotation cap")
    strategy_field = {
        "ribbon": "ribbon_score",
        "transform": "transform_score",
        "broad": "broad_score",
        "composite": "composite_score",
    }.get(hypothesis_strategy, "composite_score")
    chosen_candidate = min(
        orientation_candidates,
        key=lambda candidate: (
            candidate[strategy_field],
            candidate["composite_score"],
            abs(candidate["reference_split_offset"]),
            abs(candidate["seam_offset"]),
            candidate["reverse"],
            candidate["swap_walls"],
        ),
    )
    seam_offset = chosen_candidate["seam_offset"]
    reference_split_label = chosen_candidate["reference_split_label"]
    reference_split_offset = chosen_candidate["reference_split_offset"]
    reverse = chosen_candidate["reverse"]
    swap_walls = chosen_candidate["swap_walls"]
    intervals = chosen_candidate["intervals"]
    query_curves = chosen_candidate["query_curves"]
    reference_curves = chosen_candidate["reference_curves"]
    alignment = chosen_candidate["alignment"]
    query_points, query_features, query_labels = _curve_descriptors(query_curves)
    query_mass = np.full(len(query_points), 1.0 / len(query_points), dtype=float)
    query_structure = _structure(query_points, query_mass)
    reference_points, reference_features, reference_labels = _curve_descriptors(
        reference_curves
    )
    interval_iterations = 0
    interval_converged: bool | None = None
    unprojected_fgw_distance = None
    transport_diagnostics = {
        "query_mass_coverage": 1.0,
        "shared_progress_span": 1.0,
        "backward_steps": 0,
        "stationary_steps": 0,
        "ordered": True,
        "shared_across_walls": True,
    }
    joint_diagnostics = {
        "iterations": 0,
        "converged": None,
        "objective": None,
        "history": [],
        "warnings": [],
    }
    if exact_transport:
        solver_warnings = []
        final_interval_ready = False
        for refinement in range(MAX_INTERVAL_REFINEMENTS):
            interval_iterations = refinement + 1
            (
                interval_points,
                interval_features,
                interval_labels,
                full_indices,
            ) = _restrict_wall_intervals(
                reference_points,
                reference_features,
                reference_labels,
                samples,
                intervals,
            )
            interval_feature_cost = _feature_cost(
                query_features,
                interval_features,
                query_labels,
                interval_labels,
            )
            interval_reference_mass = np.full(
                len(interval_points), 1.0 / len(interval_points), dtype=float
            )
            interval_structure = _structure(
                interval_points, interval_reference_mass
            )
            soft_transport, unprojected_fgw_distance, warnings = _srfgw(
                interval_feature_cost,
                query_structure,
                interval_structure,
                query_labels,
                interval_labels,
                query_mass=query_mass,
            )
            solver_warnings.extend(warnings)
            refined = _refined_intervals_from_soft_transport(
                soft_transport,
                query_labels,
                interval_labels,
                intervals,
                samples,
            )
            if refined == intervals:
                interval_converged = True
                final_interval_ready = True
                break
            intervals = refined
        else:
            interval_converged = False

        if not final_interval_ready:
            # The final permitted refinement changed the interval, so solve
            # once on that final geometry before projecting correspondence.
            (
                interval_points,
                interval_features,
                interval_labels,
                full_indices,
            ) = _restrict_wall_intervals(
                reference_points,
                reference_features,
                reference_labels,
                samples,
                intervals,
            )
            interval_feature_cost = _feature_cost(
                query_features, interval_features, query_labels, interval_labels
            )
            interval_reference_mass = np.full(
                len(interval_points), 1.0 / len(interval_points), dtype=float
            )
            interval_structure = _structure(
                interval_points, interval_reference_mass
            )
            soft_transport, unprojected_fgw_distance, warnings = _srfgw(
                interval_feature_cost,
                query_structure,
                interval_structure,
                query_labels,
                interval_labels,
                query_mass=query_mass,
            )
            solver_warnings.extend(warnings)
        interval_transport, transport_diagnostics = _shared_monotonic_transport(
            soft_transport,
            query_labels,
            interval_labels,
            query_mass,
        )
        (
            interval_transport,
            alignment,
            joint_diagnostics,
            fgw_distance,
            feature_distance,
            structural_distance,
        ) = _joint_correspondence_alignment(
            interval_transport,
            np.asarray(transport_diagnostics["progress"], dtype=float),
            query_points,
            interval_points,
            query_features,
            interval_features,
            query_labels,
            interval_labels,
            query_mass,
            query_structure,
            interval_structure,
            interval_feature_cost,
            samples,
        )
        if not _alignment_allowed(alignment):
            raise MatcherError(
                "Ordered transport alignment exceeded the 45 degree rotation cap"
            )
        final_progress = np.asarray(joint_diagnostics["progress"], dtype=float)
        transport_diagnostics.update(
            {
                "shared_progress_span": float(
                    final_progress[-1] - final_progress[0]
                ),
                "backward_steps": int(
                    np.sum(np.diff(final_progress) < -1e-12)
                ),
                "stationary_steps": int(
                    np.sum(np.diff(final_progress) <= 1e-6)
                ),
                "ordered": bool(
                    np.all(np.diff(final_progress) >= -1e-12)
                ),
                "progress": final_progress.tolist(),
            }
        )
        solver_warnings.extend(joint_diagnostics["warnings"])
        solver_warnings.append("joint_ordered_similarity_refinement")
    else:
        (
            interval_points,
            interval_features,
            interval_labels,
            full_indices,
        ) = _restrict_wall_intervals(
            reference_points,
            reference_features,
            reference_labels,
            samples,
            intervals,
        )
        paired_curves = {
            name: alignment["target_points"][
                index * samples : (index + 1) * samples
            ]
            for index, name in enumerate(CURVE_NAMES)
        }
        paired_points, paired_features, paired_labels = _curve_descriptors(
            paired_curves
        )
        paired_feature_cost = _feature_cost(
            query_features, paired_features, query_labels, paired_labels
        )
        feature_distance = float(
            np.sum(query_mass * np.diag(paired_feature_cost))
        )
        paired_structure = _structure(
            paired_points,
            np.full(len(paired_points), 1.0 / len(paired_points), dtype=float),
        )
        structural_distance = float(
            np.sum(
                query_mass[:, None]
                * query_mass[None, :]
                * (query_structure - paired_structure) ** 2
            )
        )
        fgw_distance = float(
            (1.0 - FGW_ALPHA) * feature_distance
            + FGW_ALPHA * structural_distance
        )
        solver_warnings = ["coarse_monotonic_transport"]
    salience_metrics = _salience_penalty(
        query_features,
        interval_features,
        query,
        reference,
        samples,
        intervals,
        aligned_distances=alignment["distances"],
        shared_progress=transport_diagnostics.get("progress"),
        reverse=reverse,
        swap_walls=swap_walls,
    )
    salience = float(salience_metrics["overall"])
    scale_reliability_penalty = _bounded_score(
        alignment.get("scale_log_disagreement", 0.0), 0.15
    )
    nonconvergence_penalty = float(
        bool(
            exact_transport
            and joint_diagnostics.get("terminal_converged") is False
        )
    )
    initialization_penalty = float(
        bool(
            exact_transport
            and joint_diagnostics.get("iterations", 0) > 0
            and joint_diagnostics.get("best_iteration", 0) == 0
        )
    )
    transform_reliability = float(
        0.75 * scale_reliability_penalty
        + 0.15 * nonconvergence_penalty
        + 0.10 * initialization_penalty
    )
    matched_reference_fraction = float(
        np.mean(
            [
                (upper - lower) / samples
                for lower, upper in intervals.values()
            ]
        )
    )
    normalized_terms = {
        "fgw": _bounded_score(fgw_distance, SCORE_SCALES["fgw"]),
        "ribbon": _bounded_score(
            alignment["ribbon_cost"], SCORE_SCALES["ribbon"]
        ),
        "salience": _bounded_score(salience, SCORE_SCALES["salience"]),
        "alignment_tail": _bounded_score(
            alignment["hausdorff95"], SCORE_SCALES["alignment_tail"]
        ),
        "rim_region": _bounded_score(
            alignment["rim_region_cost"], SCORE_SCALES["rim_region"]
        ),
        "transform_reliability": transform_reliability,
        "completeness": _completeness_penalty(
            matched_reference_fraction
        ),
    }
    score_components = {
        name: RANK_WEIGHTS[name] * normalized_terms[name]
        for name in RANK_WEIGHTS
    }
    overall = float(sum(score_components.values()))
    diagnostic_warnings = (
        solver_warnings
        + query.get("qc", {}).get("warnings", [])
        + reference.get("qc", {}).get("warnings", [])
    )
    if alignment["wall_rms_max"] > 0.08:
        diagnostic_warnings.append("poor_wall_curve_agreement")
    if alignment["thickness_rms"] > 0.06:
        diagnostic_warnings.append("poor_thickness_agreement")
    orientation_scores = np.asarray(
        [
            candidate["composite_score"]
            for candidate in orientation_candidates
        ],
        dtype=float,
    )
    evaluated_seams = sorted(evaluated_offsets)
    seam_saturated = bool(
        evaluated_seams
        and (
            seam_offset == evaluated_seams[0] < 0
            or seam_offset == evaluated_seams[-1] > 0
        )
        and abs(seam_offset) >= RIM_SEAM_INITIAL_OFFSET_FRACTION
    )
    return {
        "reference_id": reference["reference_id"],
        "source_filename": reference["source_filename"],
        "figure": reference.get("figure", ""),
        "item": reference.get("item", ""),
        "citation_label": reference.get(
            "citation_label", reference["reference_id"]
        ),
        "samples": samples,
        "reverse_traversal": reverse,
        "wall_swap": swap_walls,
        "rim_seam": {
            "query_offset_fraction": float(seam_offset),
            "query_offset_samples": int(
                round(float(seam_offset) * (samples - 1))
            ),
            "candidate_count": len(evaluated_seams),
            "candidate_offsets": evaluated_seams,
            "search_hint": (
                float(rim_seam_hint) if rim_seam_hint is not None else None
            ),
            "initial_offset_fraction": RIM_SEAM_INITIAL_OFFSET_FRACTION,
            "max_offset_fraction": RIM_SEAM_MAX_OFFSET_FRACTION,
            "adaptive_expansion_used": bool(expansion_history),
            "expansion_history": expansion_history,
            "boundary_saturated": seam_saturated,
            "geometry_limits": {
                "negative": geometry_limits[-1],
                "positive": geometry_limits[1],
            },
            "gold_point_is_hard_anchor": False,
            "split_source": (
                "continuous_master_boundary"
                if isinstance(query.get("query_master_boundary"), dict)
                else "legacy_resampled_walls"
            ),
            "canonical_seam_fraction": (
                query.get("query_master_boundary", {}).get(
                    "nominal_seam_fraction"
                )
                if isinstance(query.get("query_master_boundary"), dict)
                else None
            ),
            "annotated_gold_seam_fraction": (
                query.get("query_master_boundary", {}).get(
                    "annotated_seam_fraction"
                )
                if isinstance(query.get("query_master_boundary"), dict)
                else None
            ),
        },
        "reference_rim_split": {
            "label": reference_split_label,
            "offset_fraction": float(reference_split_offset),
            "candidate_labels": [
                label for label, _ in REFERENCE_RIM_SPLITS
            ],
            "candidate_offsets": [
                float(offset) for _, offset in REFERENCE_RIM_SPLITS
            ],
            "source": (
                "continuous_master_boundary"
                if isinstance(
                    reference.get("reference_master_boundary"), dict
                )
                else "reconstructed_legacy_boundary"
            ),
        },
        "hypothesis_strategy": hypothesis_strategy,
        "cheap_hypothesis": {
            "composite_score": chosen_candidate["composite_score"],
            "transform_score": chosen_candidate["transform_score"],
            "broad_score": chosen_candidate["broad_score"],
            "ribbon_score": chosen_candidate["ribbon_score"],
            "coverage": chosen_candidate["coverage"],
        },
        "interval": [
            min(intervals["wall_a"][0], intervals["wall_b"][0]),
            max(intervals["wall_a"][1], intervals["wall_b"][1]),
        ],
        "wall_intervals": {
            name: list(value) for name, value in intervals.items()
        },
        "matched_reference_fraction": matched_reference_fraction,
        "query_coverage": float(transport_diagnostics["query_mass_coverage"]),
        "query_mass_preserved": bool(
            transport_diagnostics["query_mass_coverage"] >= 1.0 - 1e-8
        ),
        "interval_iterations": interval_iterations,
        "interval_converged": interval_converged,
        "overall_score": float(overall),
        "fgw_cost": float(fgw_distance),
        "structural_gw_cost": structural_distance,
        "initial_fgw_cost": (
            float(unprojected_fgw_distance)
            if unprojected_fgw_distance is not None
            else float(fgw_distance)
        ),
        "unprojected_fgw_cost": (
            float(unprojected_fgw_distance)
            if unprojected_fgw_distance is not None
            else None
        ),
        "rtc_feature_cost": feature_distance,
        "salience_penalty": salience,
        "salience": salience_metrics,
        "three_curve_cost": alignment["ribbon_cost"],
        "ribbon_cost": alignment["ribbon_cost"],
        "rim_region_cost": alignment["rim_region_cost"],
        "transform_reliability": {
            "overall": transform_reliability,
            "scale_disagreement_penalty": scale_reliability_penalty,
            "nonconvergence_penalty": nonconvergence_penalty,
            "initialization_penalty": initialization_penalty,
            "hypothesis_instability_penalty": 0.0,
        },
        "normalized_score_terms": normalized_terms,
        "score_components": score_components,
        "transport": {
            key: value
            for key, value in transport_diagnostics.items()
            if key != "progress"
        },
        "joint_alignment": {
            key: value
            for key, value in joint_diagnostics.items()
            if key != "progress"
        },
        "alignment": {
            key: value
            for key, value in alignment.items()
            if key not in {"points", "target_points", "matched_indices", "distances"}
        },
        "warnings": sorted(set(diagnostic_warnings)),
        "orientation_score_spread": float(np.ptp(orientation_scores)),
        "orientation_stability": float(
            1.0 / (1.0 + np.std(orientation_scores))
        ),
        # Backward-compatible names retained for existing saved-result UI.
        "initialization_score_spread": float(np.ptp(orientation_scores)),
        "initialization_stability": float(
            1.0 / (1.0 + np.std(orientation_scores))
        ),
        "_query_points": query_points,
        "_reference_points": reference_points,
        "_interval_points": interval_points,
        "_aligned_query": alignment["points"],
        "_query_labels": query_labels,
        "_reference_labels": reference_labels,
        "_full_interval_indices": full_indices,
        "_distances": alignment["distances"],
        "_correspondence_targets": alignment["target_points"],
        "_query_artifact": query,
        "_reference_artifact": reference,
    }


def _apply_hypothesis_reliability(
    result: dict[str, Any], instability: float
) -> None:
    reliability = result["transform_reliability"]
    reliability["hypothesis_instability_penalty"] = float(instability)
    overall = float(
        0.60 * reliability["scale_disagreement_penalty"]
        + 0.15 * reliability["nonconvergence_penalty"]
        + 0.10 * reliability["initialization_penalty"]
        + 0.15 * instability
    )
    reliability["overall"] = overall
    result["normalized_score_terms"]["transform_reliability"] = overall
    result["score_components"]["transform_reliability"] = (
        RANK_WEIGHTS["transform_reliability"] * overall
    )
    result["overall_score"] = float(sum(result["score_components"].values()))
    if overall > 0.55:
        result["warnings"] = sorted(
            set(result["warnings"] + ["unstable_similarity_transform"])
        )
    if result.get("rim_seam", {}).get("boundary_saturated"):
        result["warnings"] = sorted(
            set(result["warnings"] + ["rim_seam_search_boundary_saturated"])
        )


def _match_one(
    query: dict[str, Any],
    reference: dict[str, Any],
    samples: int,
    *,
    exact_transport: bool = True,
    rim_seam_hint: float | None = None,
) -> dict[str, Any]:
    """Evaluate diverse hypotheses before committing to one exact match."""
    if not exact_transport:
        result = _match_one_single(
            query,
            reference,
            samples,
            exact_transport=False,
            rim_seam_hint=rim_seam_hint,
            hypothesis_strategy="composite",
        )
        _apply_hypothesis_reliability(result, 0.0)
        result["hypothesis_search"] = {
            "requested": 1,
            "evaluated": 1,
            "distinct": 1,
            "selected_strategy": "composite",
            "instability_penalty": 0.0,
            "summaries": [],
        }
        return result

    # The medium pass is a pruning pass, so use one exact, composite objective
    # consistent with the coarse pass. Diverse expensive hypotheses are only
    # useful for the five finalists; previously repeating them for 40 medium
    # candidates roughly doubled this dominant stage and could reverse coarse
    # ordering for reasons unrelated to final scoring.
    strategies = (
        ("composite",)
        if samples <= 48
        else ("ribbon", "transform", "broad")
    )
    evaluated: list[dict[str, Any]] = []
    errors: list[str] = []
    for strategy in strategies:
        try:
            evaluated.append(
                _match_one_single(
                    query,
                    reference,
                    samples,
                    exact_transport=True,
                    rim_seam_hint=rim_seam_hint,
                    hypothesis_strategy=strategy,
                )
            )
        except MatcherError as exc:
            errors.append(f"{strategy}: {exc}")
    if not evaluated:
        raise MatcherError(
            "No exact alignment hypothesis succeeded"
            + (f" ({'; '.join(errors)})" if errors else "")
        )

    distinct: dict[tuple[Any, ...], dict[str, Any]] = {}
    for result in evaluated:
        signature = (
            round(result["rim_seam"]["query_offset_fraction"], 6),
            result.get("reference_rim_split", {}).get("label"),
            result["reverse_traversal"],
            result["wall_swap"],
            tuple(result["wall_intervals"]["wall_a"]),
            tuple(result["wall_intervals"]["wall_b"]),
        )
        existing = distinct.get(signature)
        if existing is None or result["overall_score"] < existing["overall_score"]:
            distinct[signature] = result
    hypotheses = list(distinct.values())
    if len(hypotheses) > 1:
        rotations = np.asarray(
            [item["alignment"]["rotation_degrees"] for item in hypotheses],
            dtype=float,
        )
        scales = np.asarray(
            [max(item["alignment"]["scale"], 1e-12) for item in hypotheses],
            dtype=float,
        )
        seams = np.asarray(
            [item["rim_seam"]["query_offset_fraction"] for item in hypotheses],
            dtype=float,
        )
        instability = float(
            np.mean(
                [
                    _bounded_score(float(np.std(rotations)), 10.0),
                    _bounded_score(float(np.std(np.log(scales))), 0.15),
                    _bounded_score(float(np.std(seams)), 0.05),
                ]
            )
        )
    else:
        instability = 0.0
    for result in hypotheses:
        _apply_hypothesis_reliability(result, instability)
    selected = min(
        hypotheses,
        key=lambda item: (
            item["overall_score"],
            item["hypothesis_strategy"],
        ),
    )
    selected["hypothesis_search"] = {
        "requested": len(strategies),
        "evaluated": len(evaluated),
        "distinct": len(hypotheses),
        "selected_strategy": selected["hypothesis_strategy"],
        "instability_penalty": instability,
        "errors": errors,
        "summaries": [
            {
                "strategy": item["hypothesis_strategy"],
                "score": item["overall_score"],
                "seam_offset": item["rim_seam"]["query_offset_fraction"],
                "reference_rim_split": item.get(
                    "reference_rim_split", {}
                ).get("label"),
                "interval": item["interval"],
                "rotation_degrees": item["alignment"]["rotation_degrees"],
                "scale": item["alignment"]["scale"],
                "scale_log_disagreement": item["alignment"][
                    "scale_log_disagreement"
                ],
            }
            for item in sorted(
                hypotheses,
                key=lambda candidate: candidate["overall_score"],
            )
        ],
    }
    return selected


def _diagnostic_transform(point_sets: list[np.ndarray], size: int = 1024, margin: int = 70):
    all_points = np.vstack(point_sets)
    minimum = all_points.min(axis=0)
    maximum = all_points.max(axis=0)
    extent = np.maximum(maximum - minimum, 1e-8)
    scale = min((size - 2 * margin) / extent[0], (size - 2 * margin) / extent[1])
    offset = np.array([margin, margin]) + ((size - 2 * margin) - extent * scale) / 2.0

    def transform(points):
        result = (points - minimum) * scale + offset
        return result

    return transform


def _draw_dashed(draw: ImageDraw.ImageDraw, points: np.ndarray, color, width=4):
    for index in range(len(points) - 1):
        if (index // 4) % 2 == 0:
            draw.line([tuple(points[index]), tuple(points[index + 1])], fill=color, width=width)


def _draw_weighted_segments(
    draw: ImageDraw.ImageDraw,
    points: np.ndarray,
    confidence: np.ndarray,
    *,
    color,
    width: int,
    dashed: bool,
) -> None:
    for index in range(len(points) - 1):
        reliable = min(float(confidence[index]), float(confidence[index + 1]))
        if dashed and (index // 4) % 2 != 0:
            continue
        segment_color = color if reliable > 0.05 else (180, 185, 195)
        segment_width = width if reliable > 0.05 else max(2, width // 2)
        draw.line(
            [tuple(points[index]), tuple(points[index + 1])],
            fill=segment_color,
            width=segment_width,
        )


def render_diagnostic(result: dict[str, Any], path: Path) -> None:
    reference = result["_reference_points"]
    aligned = result["_aligned_query"]
    correspondence_targets = np.asarray(
        result.get("_correspondence_targets", []), dtype=float
    )
    labels = result["_reference_labels"]
    query_labels = result["_query_labels"]
    interval_indices = set(map(int, result["_full_interval_indices"]))
    reference_fractures = [
        np.asarray(
            result["_reference_artifact"].get("curves", {}).get(name, []),
            dtype=float,
        )
        for name in ("fracture_start", "fracture_end", "fracture")
    ]
    query_fractures = [
        np.asarray(
            result["_query_artifact"].get("curves", {}).get(name, []),
            dtype=float,
        )
        for name in ("fracture_start", "fracture_end", "fracture")
    ]
    angle_radians = math.radians(float(result["alignment"]["rotation_degrees"]))
    rotation = np.array(
        [
            [math.cos(angle_radians), -math.sin(angle_radians)],
            [math.sin(angle_radians), math.cos(angle_radians)],
        ]
    )
    translation = np.asarray(result["alignment"]["translation"], dtype=float)
    scale = float(result["alignment"]["scale"])
    aligned_query_fractures = [
        scale * (points @ rotation.T) + translation
        for points in query_fractures
        if len(points)
    ]
    diagnostic_sets = [reference, aligned]
    if len(correspondence_targets):
        diagnostic_sets.append(correspondence_targets)
    diagnostic_sets.extend(points for points in reference_fractures if len(points))
    diagnostic_sets.extend(aligned_query_fractures)
    transform = _diagnostic_transform(diagnostic_sets)
    reference_canvas = transform(reference.copy())
    query_canvas = transform(aligned.copy())
    target_canvas = (
        transform(correspondence_targets.copy())
        if len(correspondence_targets)
        else np.empty((0, 2))
    )
    image = Image.new("RGB", (1024, 1024), "white")
    draw = ImageDraw.Draw(image)
    colors = ((25, 110, 190), (225, 90, 45), (35, 150, 85))
    for curve_index in range(len(CURVE_NAMES)):
        indices = np.flatnonzero(labels == curve_index)
        draw.line([tuple(point) for point in reference_canvas[indices]], fill=(185, 190, 200), width=4)
        kept = np.asarray([index for index in indices if int(index) in interval_indices], dtype=int)
        if len(kept) > 1:
            draw.line([tuple(point) for point in reference_canvas[kept]], fill=colors[curve_index], width=7)
        query_indices = np.flatnonzero(query_labels == curve_index)
        _draw_dashed(draw, query_canvas[query_indices], tuple(max(0, value - 25) for value in colors[curve_index]), 5)
    if len(target_canvas) == len(query_canvas):
        step = max(1, len(query_canvas) // 40)
        for index in range(0, len(query_canvas), step):
            draw.line(
                [tuple(query_canvas[index]), tuple(target_canvas[index])],
                fill=(210, 210, 215),
                width=1,
            )
    for points in reference_fractures:
        if len(points):
            canvas = transform(points.copy())
            draw.line(
                [tuple(point) for point in canvas],
                fill=(145, 70, 185),
                width=5,
            )
    for points in aligned_query_fractures:
        canvas = transform(points.copy())
        _draw_dashed(draw, canvas, (120, 45, 160), 4)
    distances = result["_distances"]
    maximum = max(float(np.quantile(distances, 0.95)), 1e-8)
    step = max(1, len(aligned) // 32)
    for index in range(0, len(aligned), step):
        value = min(1.0, float(distances[index]) / maximum)
        color = (int(230 * value), int(180 * (1.0 - value)), 70)
        point = query_canvas[index]
        radius = 4
        draw.ellipse((point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius), fill=color)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")


def _public_result(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if not key.startswith("_")}


def preprocess_query(
    project_path: Path,
    image: Image.Image,
    *,
    original_filename: str,
    metadata: dict[str, Any],
    manual_curves: dict[str, Any],
) -> dict[str, Any]:
    query_id = uuid.uuid4().hex
    root = query_root(project_path) / query_id
    root.mkdir(parents=True, exist_ok=True)
    source_path = root / "query.png"
    image.save(source_path, format="PNG")
    artifact = build_manual_query_artifact(
        image,
        reference_id=query_id,
        source_filename=original_filename,
        manual_curves=manual_curves,
    )
    artifact["query_id"] = query_id
    artifact["metadata"] = dict(metadata)
    artifact["metadata_used"] = False
    artifact_path = root / "artifact.json"
    _atomic_json(artifact_path, artifact)
    render_previews(artifact, root / "preview.png", root / "overlay.png")
    return {
        "query_id": query_id,
        "artifact": artifact,
        "preview": f"queries/{query_id}/preview.png",
        "overlay": f"queries/{query_id}/overlay.png",
    }


def _reference_metadata(project_path: Path) -> dict[str, dict[str, str]]:
    """Index saved, reviewed sherd metadata by mask stem."""
    path = Path(project_path) / "cards" / "mask_info.csv"
    if not path.exists():
        return {}
    indexed: dict[str, dict[str, str]] = {}
    with open(path, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            stem = card_stem(row.get("mask_file") or row.get("file") or "")
            if not stem:
                continue
            figure = str(row.get("Figure") or "").strip()
            item = str(row.get("No.") or row.get("Number") or "").strip()
            parts = []
            if figure:
                parts.append(f"Figure {figure}")
            if item:
                parts.append(f"Item {item}")
            indexed[stem] = {
                "figure": figure,
                "item": item,
                "citation_label": " ".join(parts) or stem,
            }
    return indexed


def load_query(project_path: Path, query_id: str) -> dict[str, Any]:
    if not query_id or any(char not in "0123456789abcdef" for char in query_id.lower()):
        raise MatcherError("Invalid query identifier")
    path = query_root(project_path) / query_id / "artifact.json"
    if not path.exists():
        raise MatcherError("Query was not found")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _attach_reference_metadata(project_path: Path, references: list[dict[str, Any]]) -> None:
    reference_metadata = _reference_metadata(project_path)
    for reference in references:
        identity = reference_metadata.get(
            card_stem(reference.get("source_filename", "")),
            {},
        )
        reference["figure"] = identity.get("figure", "")
        reference["item"] = identity.get("item", "")
        reference["citation_label"] = identity.get(
            "citation_label", reference.get("reference_id", "")
        )


def _score_candidate_level(
    query: dict[str, Any],
    candidate: dict[str, Any],
    samples: int,
    exact_transport: bool,
) -> dict[str, Any] | None:
    """Score one independent candidate; geometric rejection is expected."""
    try:
        result = _match_one(
            query,
            candidate["artifact"],
            samples,
            exact_transport=exact_transport,
            rim_seam_hint=(
                candidate.get("previous", {})
                .get("rim_seam", {})
                .get("query_offset_fraction")
            ),
        )
    except MatcherError:
        return None
    result["retrieval"] = dict(candidate["retrieval"])
    return result


def _score_candidates_parallel(
    query: dict[str, Any],
    candidates: list[dict[str, Any]],
    samples: int,
    *,
    exact_transport: bool,
    level_index: int,
    progress=None,
    workers: int | None = None,
) -> list[dict[str, Any]]:
    """Score candidates concurrently and leave final ordering deterministic."""
    total = len(candidates)
    worker_count = min(total, workers or matcher_worker_count())
    if worker_count <= 1:
        results = []
        for index, candidate in enumerate(candidates):
            if progress:
                progress(
                    level_index,
                    index,
                    total,
                    f"Matching {candidate['artifact']['reference_id']} at {samples} samples per curve",
                )
            result = _score_candidate_level(
                query, candidate, samples, exact_transport
            )
            if result is not None:
                results.append(result)
        return results

    results = []
    with ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="sherdscope-match",
    ) as executor:
        futures = {
            executor.submit(
                _score_candidate_level,
                query,
                candidate,
                samples,
                exact_transport,
            ): candidate
            for candidate in candidates
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            candidate = futures[future]
            result = future.result()
            if result is not None:
                results.append(result)
            if progress:
                progress(
                    level_index,
                    completed,
                    total,
                    f"Matched {candidate['artifact']['reference_id']} at {samples} samples per curve",
                )
    return results


def _select_cascade_survivors(
    results: list[dict[str, Any]], keep: int, reserve: int
) -> list[dict[str, Any]]:
    """Keep score top-K intact and append a few retrieval-channel champions."""
    score_key = "fused_score" if any("fused_score" in item for item in results) else "overall_score"
    ordered = sorted(results, key=lambda item: (item[score_key], item["reference_id"]))
    if reserve <= 0 or len(ordered) <= keep:
        return ordered[:keep]
    selected = {item["reference_id"]: item for item in ordered[:keep]}
    rank_fields = ("outline_rank", "ribbon_rank")
    for field in rank_fields:
        channel = sorted(
            ordered,
            key=lambda item: (
                int((item.get("retrieval") or {}).get(field, 10**9)),
                item[score_key],
                item["reference_id"],
            ),
        )
        appended = 0
        for item in channel:
            if item["reference_id"] in selected:
                continue
            selected[item["reference_id"]] = item
            appended += 1
            if appended >= reserve:
                break
    return sorted(
        selected.values(),
        key=lambda item: (item[score_key], item["reference_id"]),
    )


def _stage_record(
    name: str,
    samples: int,
    seconds: float,
    results: list[dict[str, Any]],
    survivors: list[dict[str, Any]],
    target: dict[str, str] | None,
) -> dict[str, Any]:
    survivor_ids = {item["reference_id"] for item in survivors}
    ordered = sorted(
        results, key=lambda item: (item["overall_score"], item["reference_id"])
    )
    rows = [
        {
            "rank": rank,
            "reference_id": item["reference_id"],
            "figure": item.get("figure", ""),
            "item": item.get("item", ""),
            "score": float(item["overall_score"]),
            "survived": item["reference_id"] in survivor_ids,
            "retrieval": dict(item.get("retrieval") or {}),
        }
        for rank, item in enumerate(ordered, start=1)
    ]
    target_row = None
    if target:
        for row in rows:
            if (
                str(row["figure"]) == str(target.get("figure", ""))
                and str(row["item"]) == str(target.get("item", ""))
            ):
                target_row = dict(row)
                break
    return {
        "name": name,
        "samples_per_wall": samples,
        "seconds": float(seconds),
        "input_count": len(results),
        "survivor_count": len(survivors),
        "target": target_row,
        "candidates": rows,
    }


def run_match(
    project_path: Path,
    query_id: str,
    *,
    run_id: str | None = None,
    progress=None,
    known_target: dict[str, str] | None = None,
    query_metadata: dict[str, Any] | None = None,
    reference_metadata: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    project_path = Path(project_path)
    solver = pot_runtime_status()
    if not solver["available"]:
        raise MatcherError(
            "POT is required for matching but is unavailable in the running "
            "SherdScope process. Install POT>=0.9.6,<0.10 and restart SherdScope."
        )
    query = load_query(project_path, query_id)
    references = load_ready_artifacts(project_path)
    if not references:
        raise MatcherError("The reference contour library is empty")
    run_id = run_id or uuid.uuid4().hex
    destination = run_root(project_path) / run_id
    destination.mkdir(parents=True, exist_ok=True)
    _attach_reference_metadata(project_path, references)
    if progress:
        progress(
            -1,
            0,
            len(references),
            "Running cheap outline and ribbon retrieval",
        )
    retrieval_started = time.perf_counter()
    candidates, retrieval_diagnostics = retrieve_candidates(
        project_path,
        query,
        references,
        keep=RETRIEVAL_KEEP,
        known_target=known_target,
        query_metadata=query_metadata,
        reference_metadata=reference_metadata,
    )
    retrieval_diagnostics["total_seconds"] = float(
        time.perf_counter() - retrieval_started
    )
    stages = []
    fine_scored_pool: list[dict[str, Any]] = []
    for level_index, (samples, keep) in enumerate(COARSE_LEVELS):
        level_started = time.perf_counter()
        results = _score_candidates_parallel(
            query,
            candidates,
            samples,
            exact_transport=level_index > 0,
            level_index=level_index,
            progress=progress,
            workers=(matcher_worker_count() if level_index == 0 else 1),
        )
        if not results:
            raise MatcherError(
                "No references passed the rim-anchored rotation-limited matcher checks"
            )
        if query_metadata and reference_metadata:
            results = fuse_shape_results(results, query_metadata, reference_metadata)
        score_key = "fused_score" if query_metadata and reference_metadata else "overall_score"
        results.sort(key=lambda item: (item[score_key], item["reference_id"]))
        if level_index == len(COARSE_LEVELS) - 1:
            fine_scored_pool = list(results)
        survivors = _select_cascade_survivors(
            results,
            min(keep, len(results)),
            CASCADE_CHANNEL_APPEND[level_index],
        )
        stages.append(
            _stage_record(
                ("coarse", "medium", "fine")[level_index],
                samples,
                time.perf_counter() - level_started,
                results,
                survivors,
                known_target,
            )
        )
        results = survivors
        candidates = [
            {
                "artifact": result["_reference_artifact"],
                "previous": result,
                "retrieval": result["retrieval"],
            }
            for result in results
        ]
    # This is the exact fine-scored pool for this arm. In the metadata arm,
    # metadata has already participated in retrieval and every pruning stage.
    shape_candidate_pool = [
        _public_result(result)
        for result in sorted(
            fine_scored_pool,
            key=lambda item: (item.get("fused_score", item["overall_score"]), item["reference_id"]),
        )
    ]
    final_results = [candidate["previous"] for candidate in candidates[:5]]
    for rank, result in enumerate(final_results, start=1):
        result["rank"] = rank
        result["diagnostic"] = f"diagnostic_{rank}.png"
        render_diagnostic(result, destination / result["diagnostic"])
    public = [_public_result(result) for result in final_results]
    margin = (
        float(public[1]["overall_score"] - public[0]["overall_score"])
        if len(public) > 1
        else None
    )
    payload = {
        "schema_version": MATCHER_SCHEMA_VERSION,
        "algorithm_version": MATCHER_ALGORITHM_VERSION,
        "run_id": run_id,
        "query_id": query_id,
        "metadata": query.get("metadata", {}),
        "metadata_used": bool(query_metadata and reference_metadata),
        "config": matcher_config(),
        "retrieval": retrieval_diagnostics,
        "stages": stages,
        "runtime": {
            "retrieval_seconds": retrieval_diagnostics["total_seconds"],
            "stage_seconds": {
                stage["name"]: stage["seconds"] for stage in stages
            },
            "total_seconds": float(
                retrieval_diagnostics["total_seconds"]
                + sum(stage["seconds"] for stage in stages)
            ),
        },
        "confidence_margin": margin,
        "results": public,
        "shape_candidate_pool": shape_candidate_pool,
    }
    _atomic_json(destination / "result.json", payload)
    if progress:
        progress(len(COARSE_LEVELS), len(public), len(public), "Match complete")
    return payload


def score_reference_by_citation(
    project_path: Path,
    run_id: str,
    *,
    figure: str,
    item: str,
) -> dict[str, Any]:
    """Development scorer for a known Figure/Item outside the top-five shortlist."""
    project_path = Path(project_path)
    run = load_run(project_path, run_id)
    query = load_query(project_path, str(run.get("query_id") or ""))
    lookup_matches = find_contours_by_citation(project_path, figure, item)
    if not lookup_matches:
        raise MatcherError("No canonical contour was found for that Figure and Item")
    wanted_filenames = {
        str(match.get("source_filename") or "") for match in lookup_matches
    }
    references = [
        reference
        for reference in load_ready_artifacts(project_path)
        if str(reference.get("source_filename") or "") in wanted_filenames
    ]
    if not references:
        raise MatcherError("The matching contour for that Figure and Item is not ready")
    _attach_reference_metadata(project_path, references)
    destination = run_root(project_path) / run_id
    destination.mkdir(parents=True, exist_ok=True)
    results = []
    for index, reference in enumerate(references, start=1):
        try:
            result = _match_one(query, reference, COARSE_LEVELS[-1][0], exact_transport=True)
        except MatcherError:
            continue
        result["rank"] = f"dev-{index}"
        result["diagnostic"] = f"forced_{index}_{Path(reference['source_filename']).stem}.png"
        render_diagnostic(result, destination / result["diagnostic"])
        results.append(_public_result(result))
    if not results:
        raise MatcherError(
            "That Figure and Item did not pass the rim-anchored rotation-limited matcher checks"
        )
    results.sort(key=lambda candidate: (candidate["overall_score"], candidate["reference_id"]))
    payload = {
        "schema_version": MATCHER_SCHEMA_VERSION,
        "algorithm_version": MATCHER_ALGORITHM_VERSION,
        "run_id": run_id,
        "query_id": run.get("query_id"),
        "figure": figure,
        "item": item,
        "results": results,
    }
    # Keep the searched sherd separate from the immutable top-five result.
    # A unique run can therefore be safely re-exported without altering it.
    _atomic_json(destination / "forced_result.json", payload)
    return payload


def load_run(project_path: Path, run_id: str) -> dict[str, Any]:
    if not run_id or any(char not in "0123456789abcdef" for char in run_id.lower()):
        raise MatcherError("Invalid run identifier")
    path = run_root(project_path) / run_id / "result.json"
    if not path.exists():
        raise MatcherError("Match run was not found")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)
