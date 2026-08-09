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
    _retrieval_index_path,
    _retrieval_signature,
    _score_candidates_parallel,
    _split_continuous_boundary,
    retrieve_candidates,
)


BENCHMARK_SCHEMA_VERSION = 2
DEFAULT_TOP_K = (1, 5, 10, 50, 150, 300, 400)
RECALL_CURVE_MAX_K = 150
REPORT_TOP_K = (1, 5, 10, 50, 150)
WILSON_Z_95 = 1.959963984540054
DOSE_CONDITIONS = ("partial_25", "partial_50", "partial_75")


@dataclass(frozen=True)
class SyntheticCondition:
    name: str
    point_noise: float
    smooth_noise: float
    fixed_coverage: float | None = None
    fixed_asymmetry: float | None = None


CONDITIONS = {
    "grid_clean": SyntheticCondition("grid_clean", 0.0, 0.0),
    "clean": SyntheticCondition("clean", 0.0, 0.0),
    "light": SyntheticCondition("light", 0.0008, 0.0015),
    "moderate": SyntheticCondition("moderate", 0.0018, 0.0035),
    "partial_75": SyntheticCondition("partial_75", 0.0, 0.0, 0.75, 1.0),
    "partial_50": SyntheticCondition("partial_50", 0.0, 0.0, 0.50, 1.0),
    "partial_25": SyntheticCondition("partial_25", 0.0, 0.0, 0.25, 1.0),
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
    if condition.fixed_coverage is not None:
        coverage = float(
            condition.fixed_coverage if coverage is None else coverage
        )
        asymmetry = float(condition.fixed_asymmetry or 1.0)
    elif condition.name == "grid_clean":
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
        "target_visible_fraction": condition.fixed_coverage,
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


def _parent_channel_ranks(
    candidates: Iterable[dict[str, Any]], parent_reference_id: str
) -> tuple[int | None, int | None]:
    """Return parent ranks for both retrieval channels when it is shortlisted.

    With a 400-candidate combined pool, every item that is Top-150 in either
    channel must be present: the union of two Top-150 lists has at most 300
    items. A missing parent can therefore be counted safely as a failure for
    per-channel recall through K=150 without mutating the core retriever.
    """
    for candidate in candidates:
        artifact = candidate.get("artifact", candidate)
        if artifact.get("reference_id") != parent_reference_id:
            continue
        retrieval = candidate.get("retrieval") or {}
        return retrieval.get("outline_rank"), retrieval.get("ribbon_rank")
    return None, None


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    """Two-sided 95% Wilson score interval for a binomial proportion."""
    if total <= 0:
        return 0.0, 0.0
    proportion = successes / total
    z2 = WILSON_Z_95 * WILSON_Z_95
    denominator = 1.0 + z2 / total
    centre = (proportion + z2 / (2.0 * total)) / denominator
    margin = (
        WILSON_Z_95
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z2 / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, centre - margin), min(1.0, centre + margin)


def intervals_overlap(
    first: tuple[float, float], second: tuple[float, float]
) -> bool:
    return max(first[0], second[0]) <= min(first[1], second[1])


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


def _summary(
    rows: list[dict[str, Any]],
    field: str,
    *,
    max_reliable_k: int | None = None,
    include_median: bool = True,
) -> dict[str, Any]:
    ranks = [row.get(field) for row in rows]
    present = [int(rank) for rank in ranks if isinstance(rank, int)]
    total = len(rows)
    summary = {
        "total": total,
        "missing": total - len(present),
        "mean_reciprocal_rank": (
            float(np.mean([1.0 / rank for rank in present])) if present else 0.0
        ),
        "median_rank": (
            float(np.median(present)) if present and include_median else None
        ),
        "median_rank_censored": (
            float(np.median([rank if isinstance(rank, int) else RETRIEVAL_KEEP + 1 for rank in ranks]))
            if ranks and include_median else None
        ),
        "max_reliable_k": max_reliable_k,
    }
    for k in DEFAULT_TOP_K:
        if max_reliable_k is not None and k > max_reliable_k:
            continue
        summary[f"top_{k}"] = sum(rank <= k for rank in present)
        summary[f"top_{k}_accuracy"] = (
            summary[f"top_{k}"] / total if total else 0.0
        )
        low, high = wilson_interval(summary[f"top_{k}"], total)
        summary[f"top_{k}_ci95_low"] = low
        summary[f"top_{k}_ci95_high"] = high
    curve = []
    for k in range(1, RECALL_CURVE_MAX_K + 1):
        hits = sum(rank <= k for rank in present)
        low, high = wilson_interval(hits, total)
        curve.append({
            "k": k,
            "hits": hits,
            "total": total,
            "recall": hits / total if total else 0.0,
            "ci95_low": low,
            "ci95_high": high,
        })
    summary["recall_curve"] = curve
    return summary


def _retrieval_index_ready(
    project_path: Path, references: list[dict[str, Any]]
) -> bool:
    """Check the shared retrieval cache without creating or replacing it."""
    path = _retrieval_index_path(project_path)
    if not path.exists():
        return False
    expected_ids = [str(item.get("reference_id") or "") for item in references]
    try:
        with np.load(path, allow_pickle=False) as stored:
            return (
                str(stored["signature"].item()) == _retrieval_signature(references)
                and [str(value) for value in stored["reference_ids"]] == expected_ids
                and int(stored["outline_descriptors"].shape[0]) == len(references)
                and int(stored["ribbon_descriptors"].shape[0]) == len(references)
            )
    except (OSError, ValueError, KeyError):
        return False


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summary_table_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for condition in report["conditions"]:
        condition_summary = report["summary"][condition]
        for channel in ("retrieval", "outline", "ribbon"):
            summary = condition_summary[channel]
            for k in REPORT_TOP_K:
                rows.append({
                    "condition": condition,
                    "channel": channel,
                    "k": k,
                    "hits": summary[f"top_{k}"],
                    "total": summary["total"],
                    "recall": summary[f"top_{k}_accuracy"],
                    "ci95_low": summary[f"top_{k}_ci95_low"],
                    "ci95_high": summary[f"top_{k}_ci95_high"],
                    "median_rank_retrieved": summary["median_rank"],
                    "median_rank_censored": summary["median_rank_censored"],
                })
    return rows


def _recall_curve_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for condition in report["conditions"]:
        for point in report["summary"][condition]["retrieval"]["recall_curve"]:
            rows.append({"condition": condition, **point})
    return rows


def _noise_overlap_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    if "clean" not in report["summary"]:
        return []
    rows = []
    clean = report["summary"]["clean"]["retrieval"]
    ranks_by_condition = {
        condition: {
            row["parent_reference_id"]: row.get("retrieval_rank")
            for row in report.get("rows", []) if row.get("condition") == condition
        }
        for condition in ("clean", "light", "moderate")
    }
    for other_name in ("light", "moderate"):
        if other_name not in report["summary"]:
            continue
        other = report["summary"][other_name]["retrieval"]
        for k in REPORT_TOP_K:
            clean_ci = (clean[f"top_{k}_ci95_low"], clean[f"top_{k}_ci95_high"])
            other_ci = (other[f"top_{k}_ci95_low"], other[f"top_{k}_ci95_high"])
            clean_only = 0
            other_only = 0
            common_parents = set(ranks_by_condition["clean"]) & set(ranks_by_condition[other_name])
            for parent_id in common_parents:
                clean_rank = ranks_by_condition["clean"][parent_id]
                other_rank = ranks_by_condition[other_name][parent_id]
                clean_hit = isinstance(clean_rank, int) and clean_rank <= k
                other_hit = isinstance(other_rank, int) and other_rank <= k
                clean_only += int(clean_hit and not other_hit)
                other_only += int(other_hit and not clean_hit)
            discordant = clean_only + other_only
            if discordant:
                tail = sum(
                    math.comb(discordant, value)
                    for value in range(0, min(clean_only, other_only) + 1)
                ) / (2 ** discordant)
                paired_p = min(1.0, 2.0 * tail)
            else:
                paired_p = 1.0
            rows.append({
                "condition_a": "clean",
                "condition_b": other_name,
                "k": k,
                "condition_a_recall": clean[f"top_{k}_accuracy"],
                "condition_a_ci95_low": clean_ci[0],
                "condition_a_ci95_high": clean_ci[1],
                "condition_b_recall": other[f"top_{k}_accuracy"],
                "condition_b_ci95_low": other_ci[0],
                "condition_b_ci95_high": other_ci[1],
                "ci95_overlap": intervals_overlap(clean_ci, other_ci),
                "paired_clean_only_hits": clean_only,
                "paired_noise_only_hits": other_only,
                "mcnemar_exact_p": paired_p,
                "note": "CI overlap is descriptive; McNemar exact p uses paired parent outcomes",
            })
    return rows


def _scaling_rows(
    current: dict[str, Any], comparison_reports: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    reports = {f"current_{current['reference_count']}": current, **comparison_reports}
    rows = []
    for label, report in reports.items():
        for condition, condition_summary in report.get("summary", {}).items():
            retrieval = condition_summary.get("retrieval", {})
            for k in REPORT_TOP_K:
                accuracy_key = f"top_{k}_accuracy"
                if accuracy_key not in retrieval:
                    continue
                hits = retrieval.get(f"top_{k}")
                total = retrieval.get("total")
                fallback_ci = (
                    wilson_interval(int(hits), int(total))
                    if hits is not None and total is not None else (None, None)
                )
                rows.append({
                    "index_label": label,
                    "reference_count": report.get("reference_count"),
                    "selected_parent_count": report.get("selected_parent_count"),
                    "condition": condition,
                    "k": k,
                    "hits": hits,
                    "total": total,
                    "recall": retrieval.get(accuracy_key),
                    "ci95_low": retrieval.get(f"top_{k}_ci95_low", fallback_ci[0]),
                    "ci95_high": retrieval.get(f"top_{k}_ci95_high", fallback_ci[1]),
                    "median_rank_retrieved": retrieval.get("median_rank"),
                })
    return rows


def _plot_recall_curves(report: dict[str, Any], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(10, 6.2))
    for condition in report["conditions"]:
        curve = report["summary"][condition]["retrieval"]["recall_curve"]
        axis.plot(
            [point["k"] for point in curve],
            [point["recall"] for point in curve],
            label=condition.replace("_", " "),
            linewidth=2,
        )
    axis.set(title="True-parent retrieval recall", xlabel="K", ylabel="Recall@K")
    axis.set_xlim(1, RECALL_CURVE_MAX_K)
    axis.set_ylim(0, 1.01)
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, fontsize=9)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_dose_response(report: dict[str, Any], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    available = [name for name in DOSE_CONDITIONS if name in report["summary"]]
    if not available:
        return
    available.sort(key=lambda name: int(name.rsplit("_", 1)[1]))
    fractions = [int(name.rsplit("_", 1)[1]) for name in available]
    figure, axis = plt.subplots(figsize=(9, 6.2))
    for k in (1, 5, 10, 50, 150):
        summaries = [report["summary"][name]["retrieval"] for name in available]
        values = [item[f"top_{k}_accuracy"] for item in summaries]
        lower = [value - item[f"top_{k}_ci95_low"] for value, item in zip(values, summaries)]
        upper = [item[f"top_{k}_ci95_high"] - value for value, item in zip(values, summaries)]
        axis.errorbar(
            fractions, values, yerr=[lower, upper], marker="o", capsize=4,
            linewidth=2, label=f"Recall@{k}",
        )
    axis.set(
        title="Dose-response: retrieval versus visible profile fraction",
        xlabel="Visible profile arc (%)",
        ylabel="True-parent recall",
    )
    axis.set_xticks(fractions)
    axis.set_ylim(0, 1.01)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _format_interval(summary: dict[str, Any], k: int) -> str:
    return (
        f"{summary[f'top_{k}_accuracy']:.1%} "
        f"[{summary[f'top_{k}_ci95_low']:.1%}, {summary[f'top_{k}_ci95_high']:.1%}]"
    )


def _write_markdown_summary(
    report: dict[str, Any], overlap_rows: list[dict[str, Any]], path: Path
) -> None:
    lines = [
        "# Synthetic retrieval benchmark summary",
        "",
        f"- References searched: **{report['reference_count']}**",
        f"- Parent references sampled: **{report['selected_parent_count']}**",
        f"- Total queries: **{report['query_count']}**",
        "- Mode: **retrieval only** (no full fine scoring)",
        "",
        "## Combined retrieval",
        "",
        "| Condition | Top-1 (95% CI) | Top-5 (95% CI) | Top-10 (95% CI) | Top-50 (95% CI) | Top-150 (95% CI) | Median rank* |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in report["conditions"]:
        summary = report["summary"][condition]["retrieval"]
        lines.append(
            f"| {condition} | {_format_interval(summary, 1)} | {_format_interval(summary, 5)} | "
            f"{_format_interval(summary, 10)} | {_format_interval(summary, 50)} | "
            f"{_format_interval(summary, 150)} | {summary['median_rank']} |"
        )
    lines.extend([
        "",
        "*Median rank is calculated among retrieved parents; the CSV also reports a censored median with misses assigned rank 401.",
        "",
        "## Clean versus noise CI-overlap check",
        "",
        "| Comparison | K | 95% CIs overlap? | Paired exact p |",
        "|---|---:|---:|---:|",
    ])
    for row in overlap_rows:
        lines.append(
            f"| {row['condition_a']} vs {row['condition_b']} | {row['k']} | "
            f"{'yes' if row['ci95_overlap'] else 'no'} | {row['mcnemar_exact_p']:.4f} |"
        )
    lines.extend([
        "",
        "CI overlap is descriptive. The paired exact p-value uses each parent as its own control.",
        "Per-channel outline and ribbon results are available in `summary_top_k.csv`.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_analysis_outputs(
    report: dict[str, Any],
    output_dir: Path,
    *,
    comparison_reports: dict[str, dict[str, Any]] | None = None,
) -> dict[str, str]:
    """Write human-readable tables and plots for a completed retrieval run."""
    output_dir = Path(output_dir)
    summary_rows = _summary_table_rows(report)
    curve_rows = _recall_curve_rows(report)
    overlap_rows = _noise_overlap_rows(report)
    scaling_rows = _scaling_rows(report, comparison_reports or {})
    paths = {
        "summary_table_csv": output_dir / "summary_top_k.csv",
        "recall_curve_csv": output_dir / "recall_at_k_1_150.csv",
        "noise_ci_overlap_csv": output_dir / "noise_ci_overlap.csv",
        "scaling_comparison_csv": output_dir / "scaling_comparison.csv",
        "summary_markdown": output_dir / "summary.md",
        "recall_curve_plot": output_dir / "recall_at_k_1_150.png",
        "dose_response_plot": output_dir / "dose_response.png",
    }
    _write_csv(paths["summary_table_csv"], summary_rows)
    _write_csv(paths["recall_curve_csv"], curve_rows)
    _write_csv(paths["noise_ci_overlap_csv"], overlap_rows)
    _write_csv(paths["scaling_comparison_csv"], scaling_rows)
    _write_markdown_summary(report, overlap_rows, paths["summary_markdown"])
    _plot_recall_curves(report, paths["recall_curve_plot"])
    _plot_dose_response(report, paths["dose_response_plot"])
    return {name: str(path) for name, path in paths.items() if path.exists()}


def run_synthetic_benchmark(
    project_path: Path,
    output_dir: Path,
    *,
    sample_size: int = 300,
    seed: int = 20260801,
    condition_names: tuple[str, ...] = (
        "grid_clean", "clean", "light", "moderate",
        "partial_75", "partial_50", "partial_25",
    ),
    full_rerank_limit: int = 0,
    require_existing_index: bool = False,
    allow_output_overwrite: bool = False,
    comparison_report_paths: dict[str, Path] | None = None,
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
    if RETRIEVAL_KEEP < 2 * RECALL_CURVE_MAX_K:
        raise RuntimeError(
            "Per-channel recall through K=150 requires a combined pool of at least 300"
        )
    if require_existing_index and not _retrieval_index_ready(project_path, references):
        raise RuntimeError(
            "The retrieval index is missing or stale. Build it after active matcher "
            "jobs finish, or rerun without --require-existing-index."
        )
    protected_outputs = (
        "synthetic_benchmark.json", "synthetic_benchmark.csv", "summary_top_k.csv",
        "recall_at_k_1_150.csv", "noise_ci_overlap.csv", "scaling_comparison.csv",
        "summary.md", "recall_at_k_1_150.png", "dose_response.png",
    )
    existing = [name for name in protected_outputs if (output_dir / name).exists()]
    if existing and not allow_output_overwrite:
        raise FileExistsError(
            f"Output directory already contains benchmark artifacts: {', '.join(existing)}"
        )

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
            outline_rank, ribbon_rank = _parent_channel_ranks(candidates, parent_id)
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
                    "outline_rank": outline_rank,
                    "ribbon_rank": ribbon_rank,
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
            "outline": _summary(
                condition_rows, "outline_rank",
                max_reliable_k=RECALL_CURVE_MAX_K, include_median=False,
            ),
            "ribbon": _summary(
                condition_rows, "ribbon_rank",
                max_reliable_k=RECALL_CURVE_MAX_K, include_median=False,
            ),
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
        "retrieval_keep": RETRIEVAL_KEEP,
        "recall_curve_max_k": RECALL_CURVE_MAX_K,
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
    comparison_reports = {}
    for label, path in (comparison_report_paths or {}).items():
        comparison_reports[label] = json.loads(Path(path).read_text(encoding="utf-8"))
    report["artifacts"] = write_analysis_outputs(
        report, output_dir, comparison_reports=comparison_reports
    )
    report["json_path"] = str(json_path)
    report["csv_path"] = str(csv_path)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
