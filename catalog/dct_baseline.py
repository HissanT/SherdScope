"""Read-only Discrete Cosine Transform baseline for pottery rim retrieval.

The baseline intentionally stays separate from the production FGW matcher.  It
uses the same reviewed continuous boundaries, but reduces each normalized open
rim outline to DCT coefficients and ranks references by coefficient RMSD.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
from scipy.fft import dct, idct

from catalog.contours import load_ready_artifacts
from catalog.matcher import (
    _master_boundary,
    _polyline_interval,
    _reference_metadata,
    _split_continuous_boundary,
    load_query,
    query_root,
)
from catalog.matcher_benchmark import (
    CONDITIONS,
    _rim_oriented_boundary,
    synthetic_query_from_reference,
)


DCT_BASELINE_SCHEMA_VERSION = 1
DCT_BASELINE_ALGORITHM_VERSION = "paper-rim-dct-v2-coverage50"
DEFAULT_CONDITIONS = (
    "grid_clean",
    "clean",
    "light",
    "moderate",
    "partial_75",
    "partial_50",
    "partial_25",
)
REPORT_K = (1, 3, 5, 10)


@dataclass(frozen=True)
class DCTConfig:
    """All choices that can change a DCT score."""

    samples: int = 100
    harmonics: int = 20
    min_reference_coverage: float = 0.50
    max_reference_coverage: float = 1.00
    coverage_steps: int = 11
    include_dc: bool = True

    def validate(self) -> None:
        if self.samples < 16 or self.samples % 2:
            raise ValueError("DCT samples must be an even integer of at least 16")
        available = self.samples if self.include_dc else self.samples - 1
        if not 1 <= self.harmonics <= available:
            raise ValueError(
                f"DCT harmonics must be between 1 and {available} for this setup"
            )
        if not 0.05 <= self.min_reference_coverage <= 1.0:
            raise ValueError("Minimum reference coverage must be between 0.05 and 1")
        if not self.min_reference_coverage <= self.max_reference_coverage <= 1.0:
            raise ValueError(
                "Maximum reference coverage must be at least the minimum and at most 1"
            )
        if self.coverage_steps < 1:
            raise ValueError("Coverage steps must be at least 1")

    @property
    def coverages(self) -> np.ndarray:
        self.validate()
        return np.linspace(
            self.min_reference_coverage,
            self.max_reference_coverage,
            self.coverage_steps,
            dtype=float,
        )

    def provenance(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "coverage_values": self.coverages.tolist(),
            "normalization": "rim_origin_endpoint_axis_rms_scale",
            "coefficient_scaling": "orthonormal_dct_ii",
            "distance": "coefficient_root_mean_square_difference",
            "metadata_used": False,
            "production_matcher_used": False,
        }


@dataclass(frozen=True)
class DCTReferenceBank:
    references: tuple[dict[str, Any], ...]
    coverages: np.ndarray
    descriptors: np.ndarray
    config: DCTConfig


def _artifact_walls(artifact: dict[str, Any], samples_per_wall: int) -> dict[str, np.ndarray]:
    if isinstance(artifact.get("query_master_boundary"), dict):
        points, seam = _master_boundary(artifact)
    else:
        try:
            points, seam, _ = _rim_oriented_boundary(artifact)
        except ValueError:
            points, seam = _master_boundary(artifact)
    return _split_continuous_boundary(points, seam, 0.0, samples_per_wall)


def _canonical_open_outline(
    walls: dict[str, np.ndarray],
    *,
    samples: int,
    coverage: float = 1.0,
) -> np.ndarray:
    """Return a pose-normalized fracture-to-rim-to-fracture open outline.

    This is preprocessing, not iterative alignment.  The reviewed rim is put at
    the origin, the mean fracture direction is put on the positive vertical
    axis, and RMS radius removes global image scale.
    """
    if samples % 2:
        raise ValueError("DCT samples must be even")
    count = samples // 2
    coverage = float(np.clip(coverage, 0.01, 1.0))
    wall_a = _polyline_interval(
        np.asarray(walls["wall_a"], dtype=float), 0.0, coverage, count
    )
    wall_b = _polyline_interval(
        np.asarray(walls["wall_b"], dtype=float), 0.0, coverage, count
    )
    rim = 0.5 * (wall_a[0] + wall_b[0])
    wall_a = wall_a - rim
    wall_b = wall_b - rim

    tail_count = max(2, count // 10)
    direction = 0.5 * (
        np.mean(wall_a[-tail_count:], axis=0)
        + np.mean(wall_b[-tail_count:], axis=0)
    )
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-10:
        raise ValueError("The profile is too short to establish a canonical direction")
    angle = math.pi / 2.0 - math.atan2(float(direction[1]), float(direction[0]))
    rotation = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=float,
    )
    wall_a = wall_a @ rotation.T
    wall_b = wall_b @ rotation.T
    outline = np.vstack((wall_a[::-1], wall_b))
    scale = float(np.sqrt(np.mean(np.sum(outline**2, axis=1))))
    if scale <= 1e-10:
        raise ValueError("The profile has no measurable size")
    return outline / scale


def dct_descriptor(outline: np.ndarray, config: DCTConfig) -> np.ndarray:
    """Return paired x/y coefficients for an ordered open outline."""
    config.validate()
    outline = np.asarray(outline, dtype=float)
    if outline.shape != (config.samples, 2) or not np.all(np.isfinite(outline)):
        raise ValueError(
            f"Expected a finite {config.samples} by 2 canonical outline"
        )
    coefficients = dct(outline, type=2, axis=0, norm="ortho")
    start = 0 if config.include_dc else 1
    selected = coefficients[start : start + config.harmonics]
    return selected.reshape(-1)


def dct_reconstruction(outline: np.ndarray, config: DCTConfig) -> np.ndarray:
    """Reconstruct an outline using only the coefficients used for scoring."""
    config.validate()
    outline = np.asarray(outline, dtype=float)
    coefficients = dct(outline, type=2, axis=0, norm="ortho")
    kept = np.zeros_like(coefficients)
    start = 0 if config.include_dc else 1
    kept[start : start + config.harmonics] = coefficients[
        start : start + config.harmonics
    ]
    return idct(kept, type=2, axis=0, norm="ortho")


def artifact_descriptor(
    artifact: dict[str, Any],
    config: DCTConfig,
    *,
    coverage: float = 1.0,
) -> np.ndarray:
    walls = _artifact_walls(artifact, max(64, config.samples // 2))
    outline = _canonical_open_outline(
        walls, samples=config.samples, coverage=coverage
    )
    return dct_descriptor(outline, config)


def build_reference_bank(
    references: Iterable[dict[str, Any]], config: DCTConfig | None = None
) -> DCTReferenceBank:
    """Precompute all reference descriptors once for fast repeated queries."""
    config = config or DCTConfig()
    config.validate()
    references = tuple(references)
    if not references:
        raise ValueError("At least one reference profile is required")
    coverages = config.coverages
    descriptors = np.empty(
        (len(references), len(coverages), 2 * config.harmonics), dtype=float
    )
    for reference_index, reference in enumerate(references):
        walls = _artifact_walls(reference, max(64, config.samples // 2))
        for coverage_index, coverage in enumerate(coverages):
            outline = _canonical_open_outline(
                walls, samples=config.samples, coverage=float(coverage)
            )
            descriptors[reference_index, coverage_index] = dct_descriptor(
                outline, config
            )
    return DCTReferenceBank(references, coverages, descriptors, config)


def score_query(
    query: dict[str, Any],
    bank: DCTReferenceBank,
    *,
    exclude_reference_id: str | None = None,
) -> list[dict[str, Any]]:
    """Rank every reference using the best tested rim-coverage hypothesis."""
    query_descriptor = artifact_descriptor(query, bank.config)
    squared = (bank.descriptors - query_descriptor[None, None, :]) ** 2
    errors = np.sqrt(np.mean(squared, axis=2))
    best_coverage_indices = np.argmin(errors, axis=1)
    best_errors = errors[np.arange(len(bank.references)), best_coverage_indices]
    reference_metadata = []
    for index, reference in enumerate(bank.references):
        reference_id = str(reference.get("reference_id") or "")
        if exclude_reference_id and reference_id == exclude_reference_id:
            continue
        reference_metadata.append(
            {
                "reference_id": reference_id,
                "source_filename": str(reference.get("source_filename") or ""),
                "figure": str(reference.get("figure") or ""),
                "item": str(reference.get("item") or ""),
                "citation_label": str(
                    reference.get("citation_label") or reference_id
                ),
                "dct_rmsd": float(best_errors[index]),
                "best_reference_coverage": float(
                    bank.coverages[best_coverage_indices[index]]
                ),
            }
        )
    reference_metadata.sort(
        key=lambda row: (row["dct_rmsd"], row["reference_id"])
    )
    for rank, row in enumerate(reference_metadata, start=1):
        row["rank"] = rank
    return reference_metadata


def _prepare_output_dir(output_dir: Path, filenames: Iterable[str], overwrite: bool) -> None:
    output_dir = Path(output_dir)
    existing = [name for name in filenames if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Output directory already contains DCT artifacts: " + ", ".join(existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _attach_metadata(project_path: Path, references: list[dict[str, Any]]) -> None:
    metadata = _reference_metadata(project_path)
    for reference in references:
        stem = Path(str(reference.get("source_filename") or "")).stem
        identity = metadata.get(stem, {})
        reference["figure"] = identity.get(
            "figure", str(reference.get("figure") or "")
        )
        reference["item"] = identity.get(
            "item", str(reference.get("item") or "")
        )
        reference["citation_label"] = identity.get(
            "citation_label", reference.get("reference_id", "")
        )


def _write_reconstruction_plot(
    query: dict[str, Any], config: DCTConfig, path: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    walls = _artifact_walls(query, max(64, config.samples // 2))
    outline = _canonical_open_outline(walls, samples=config.samples)
    reconstructed = dct_reconstruction(outline, config)
    figure, axis = plt.subplots(figsize=(5.5, 6.0))
    axis.plot(outline[:, 0], outline[:, 1], color="#6b7280", linewidth=2, label="100-point outline")
    axis.plot(
        reconstructed[:, 0], reconstructed[:, 1], color="#dc2626", linewidth=2,
        linestyle="--", label=f"{config.harmonics}-harmonic DCT",
    )
    axis.set_aspect("equal")
    axis.invert_yaxis()
    axis.set_title("DCT query reconstruction")
    axis.set_xlabel("Normalized horizontal position")
    axis.set_ylabel("Normalized vertical position")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run_query_experiment(
    project_path: Path,
    output_dir: Path,
    *,
    query_id: str | None = None,
    query_artifact_path: Path | None = None,
    config: DCTConfig | None = None,
    top_k: int = 20,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Score one saved or file-backed query and persist ranked results."""
    if bool(query_id) == bool(query_artifact_path):
        raise ValueError("Provide exactly one of query_id or query_artifact_path")
    config = config or DCTConfig()
    project_path = Path(project_path)
    output_dir = Path(output_dir)
    files = ("dct_results.json", "dct_results.csv", "query_dct_reconstruction.png")
    _prepare_output_dir(output_dir, files, overwrite)
    if query_id:
        query = load_query(project_path, query_id)
        query_source = {"kind": "saved_query", "query_id": query_id}
    else:
        query_artifact_path = Path(query_artifact_path)  # type: ignore[arg-type]
        query = json.loads(query_artifact_path.read_text(encoding="utf-8"))
        query_source = {"kind": "artifact_json", "path": str(query_artifact_path)}
    references = load_ready_artifacts(project_path)
    _attach_metadata(project_path, references)
    started = time.perf_counter()
    bank = build_reference_bank(references, config)
    ranked = score_query(query, bank)
    elapsed = time.perf_counter() - started
    kept = ranked[: max(1, int(top_k))]
    report = {
        "schema_version": DCT_BASELINE_SCHEMA_VERSION,
        "algorithm_version": DCT_BASELINE_ALGORITHM_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "query": query_source,
        "reference_count": len(references),
        "elapsed_seconds": elapsed,
        "config": config.provenance(),
        "results": kept,
    }
    (output_dir / "dct_results.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    _write_csv(output_dir / "dct_results.csv", kept)
    _write_reconstruction_plot(query, config, output_dir / files[2])
    return report


def _rank_of(rows: list[dict[str, Any]], reference_id: str) -> int | None:
    for row in rows:
        if row["reference_id"] == reference_id:
            return int(row["rank"])
    return None


def _rank_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ranks = [int(row["rank"]) for row in rows if row.get("rank") is not None]
    total = len(rows)
    summary = {
        "total": total,
        "missing": total - len(ranks),
        "mean_reciprocal_rank": (
            float(np.mean([1.0 / rank for rank in ranks])) if ranks else 0.0
        ),
        "median_rank": float(np.median(ranks)) if ranks else None,
    }
    for k in REPORT_K:
        hits = sum(rank <= k for rank in ranks)
        summary[f"top_{k}"] = hits
        summary[f"top_{k}_accuracy"] = hits / total if total else 0.0
    return summary


def _summary_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# DCT synthetic baseline",
        "",
        f"- Algorithm: `{report['algorithm_version']}`",
        f"- References: **{report['reference_count']}**",
        f"- Parent profiles: **{report['selected_parent_count']}**",
        f"- Generated queries: **{report['query_count']}**",
        f"- Harmonics: **{report['config']['harmonics']}**",
        f"- Samples: **{report['config']['samples']}**",
        "",
        "| Condition | Top-1 | Top-3 | Top-5 | Top-10 | MRR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition in report["conditions"]:
        summary = report["summary"][condition]
        lines.append(
            f"| {condition} | {summary['top_1_accuracy']:.3f} | "
            f"{summary['top_3_accuracy']:.3f} | {summary['top_5_accuracy']:.3f} | "
            f"{summary['top_10_accuracy']:.3f} | "
            f"{summary['mean_reciprocal_rank']:.3f} |"
        )
    lines.extend(
        [
            "",
            "The known parent remains in the reference library. This measures exact-parent",
            "retrieval on controlled fragments and is not the class-level leave-one-out test",
            "reported by Wilczek et al.",
            "",
        ]
    )
    return "\n".join(lines)


def run_synthetic_experiment(
    project_path: Path,
    output_dir: Path,
    *,
    sample_size: int = 300,
    seed: int = 20260801,
    condition_names: tuple[str, ...] = DEFAULT_CONDITIONS,
    config: DCTConfig | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run deterministic DCT scoring on the existing synthetic conditions."""
    config = config or DCTConfig()
    project_path = Path(project_path)
    output_dir = Path(output_dir)
    files = ("dct_synthetic_benchmark.json", "dct_synthetic_benchmark.csv", "summary.md")
    _prepare_output_dir(output_dir, files, overwrite)
    invalid = [name for name in condition_names if name not in CONDITIONS]
    if invalid:
        raise ValueError("Unknown synthetic condition(s): " + ", ".join(invalid))
    references = load_ready_artifacts(project_path)
    if not references:
        raise ValueError("The project has no approved contour references")
    _attach_metadata(project_path, references)
    rng = np.random.default_rng(seed)
    selected_count = min(max(1, int(sample_size)), len(references))
    selected_indices = rng.choice(len(references), size=selected_count, replace=False)
    selected = [references[int(index)] for index in selected_indices]
    started = time.perf_counter()
    bank = build_reference_bank(references, config)
    rows: list[dict[str, Any]] = []
    for reference in selected:
        paired_seed = int(rng.integers(0, np.iinfo(np.int64).max))
        for condition_name in condition_names:
            query, provenance = synthetic_query_from_reference(
                reference,
                rng=np.random.default_rng(paired_seed),
                condition=CONDITIONS[condition_name],
            )
            query_started = time.perf_counter()
            ranked = score_query(query, bank)
            parent_id = str(provenance["parent_reference_id"])
            rank = _rank_of(ranked, parent_id)
            parent = next(
                (row for row in ranked if row["reference_id"] == parent_id), None
            )
            rows.append(
                {
                    **provenance,
                    "rank": rank,
                    "parent_dct_rmsd": parent["dct_rmsd"] if parent else None,
                    "parent_best_reference_coverage": (
                        parent["best_reference_coverage"] if parent else None
                    ),
                    "top_reference_id": ranked[0]["reference_id"],
                    "top_dct_rmsd": ranked[0]["dct_rmsd"],
                    "query_seconds": time.perf_counter() - query_started,
                    "reference_count": len(references),
                }
            )
    summary = {
        condition: _rank_summary(
            [row for row in rows if row["condition"] == condition]
        )
        for condition in condition_names
    }
    report = {
        "schema_version": DCT_BASELINE_SCHEMA_VERSION,
        "algorithm_version": DCT_BASELINE_ALGORITHM_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "reference_count": len(references),
        "selected_parent_count": len(selected),
        "query_count": len(rows),
        "conditions": list(condition_names),
        "elapsed_seconds": time.perf_counter() - started,
        "config": config.provenance(),
        "summary": summary,
        "rows": rows,
    }
    (output_dir / files[0]).write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    _write_csv(output_dir / files[1], rows)
    (output_dir / files[2]).write_text(_summary_markdown(report), encoding="utf-8")
    return report


def _manifest_query_tasks(manifest_paths: Iterable[Path]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    seen_query_ids: set[str] = set()
    for manifest_path in manifest_paths:
        manifest_path = Path(manifest_path)
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
        queries = document.get("queries")
        if not isinstance(queries, dict):
            raise ValueError(f"Batch manifest has no query mapping: {manifest_path}")
        cohort = manifest_path.parent.name
        for key, raw_entry in queries.items():
            if not isinstance(raw_entry, dict):
                raise ValueError(f"Invalid query entry {key} in {manifest_path}")
            entry = dict(raw_entry)
            query_id = str(entry.get("query_id") or "")
            if not query_id:
                raise ValueError(f"Query entry {key} has no query_id in {manifest_path}")
            if query_id in seen_query_ids:
                raise ValueError(f"Query {query_id} occurs in more than one manifest")
            seen_query_ids.add(query_id)
            target = entry.get("known_target")
            if isinstance(target, dict):
                figure = str(target.get("figure") or "").strip()
                item = str(target.get("item") or "").strip()
            else:
                figure = str(entry.get("figure") or "").strip()
                item = str(entry.get("item") or "").strip()
            if not figure or not item:
                raise ValueError(
                    f"Query entry {key} has no true-parent Figure and Item"
                )
            tasks.append(
                {
                    "cohort": cohort,
                    "manifest": str(manifest_path),
                    "query_number": int(entry.get("number", key)),
                    "query_label": str(entry.get("label") or f"Query {key}"),
                    "query_id": query_id,
                    "source_filename": str(entry.get("source_filename") or ""),
                    "artifact_sha256": str(entry.get("artifact_sha256") or ""),
                    "true_figure": figure,
                    "true_item": item,
                }
            )
    return tasks


def _reference_id_for_target(
    references: Iterable[dict[str, Any]], figure: str, item: str
) -> str:
    matches = [
        str(reference.get("reference_id") or "")
        for reference in references
        if str(reference.get("figure") or "").strip() == figure
        and str(reference.get("item") or "").strip() == item
    ]
    matches = sorted({value for value in matches if value})
    if len(matches) != 1:
        raise ValueError(
            f"Expected one reference for Figure {figure}, Item {item}; found {len(matches)}"
        )
    return matches[0]


def _batch_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return _rank_summary(
        [{"rank": row.get("true_parent_rank")} for row in rows]
    )


def _batch_summary_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# DCT known-parent batch",
        "",
        f"- Algorithm: `{report['algorithm_version']}`",
        f"- Queries: **{report['query_count']}**",
        f"- References: **{report['reference_count']}**",
        f"- Workers: **{report['workers']}**",
        "- Allowed reference coverage: **50% through 100%**",
        "",
        "| Cohort | Queries | Top-1 | Top-3 | Top-5 | Top-10 | MRR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    summaries = {"combined": report["summary"]["combined"], **report["summary"]["cohorts"]}
    for name, summary in summaries.items():
        lines.append(
            f"| {name} | {summary['total']} | {summary['top_1_accuracy']:.3f} | "
            f"{summary['top_3_accuracy']:.3f} | {summary['top_5_accuracy']:.3f} | "
            f"{summary['top_10_accuracy']:.3f} | "
            f"{summary['mean_reciprocal_rank']:.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


def run_manifest_batch_experiment(
    project_path: Path,
    output_dir: Path,
    manifest_paths: Iterable[Path],
    *,
    config: DCTConfig | None = None,
    workers: int = 1,
    top_k: int = 20,
    overwrite: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Score known-parent saved queries from one or more batch manifests."""
    config = config or DCTConfig()
    config.validate()
    manifest_paths = tuple(Path(path) for path in manifest_paths)
    workers = int(workers)
    if not 1 <= workers <= 64:
        raise ValueError("Workers must be between 1 and 64")
    project_path = Path(project_path)
    output_dir = Path(output_dir)
    files = ("dct_batch.json", "dct_batch.csv", "summary.md")
    _prepare_output_dir(output_dir, files, overwrite)
    tasks = _manifest_query_tasks(manifest_paths)
    if not tasks:
        raise ValueError("No queries were found in the supplied manifests")
    emit = progress or (lambda _message: None)
    emit(
        f"Loaded {len(tasks)} known-parent queries from "
        f"{len(manifest_paths)} manifest(s)."
    )
    references = load_ready_artifacts(project_path)
    if not references:
        raise ValueError("The project has no approved contour references")
    _attach_metadata(project_path, references)
    for task in tasks:
        task["true_reference_id"] = _reference_id_for_target(
            references, task["true_figure"], task["true_item"]
        )
    started = time.perf_counter()
    emit(
        f"Building one shared DCT bank for {len(references)} references "
        f"at {len(config.coverages)} coverage values..."
    )
    bank_started = time.perf_counter()
    bank = build_reference_bank(references, config)
    emit(f"Reference bank ready in {time.perf_counter() - bank_started:.1f}s.")

    def evaluate(task: dict[str, Any]) -> dict[str, Any]:
        artifact_path = query_root(project_path) / task["query_id"] / "artifact.json"
        if not artifact_path.exists():
            raise ValueError(f"Saved query artifact was not found: {artifact_path}")
        expected_hash = task.get("artifact_sha256")
        if expected_hash:
            observed_hash = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
            if observed_hash != expected_hash:
                raise ValueError(
                    f"Saved query artifact changed for {task['cohort']} "
                    f"Query {task['query_number']}"
                )
        query = load_query(project_path, task["query_id"])
        query_started = time.perf_counter()
        ranked = score_query(query, bank)
        true_reference_id = task["true_reference_id"]
        true_parent = next(
            row for row in ranked if row["reference_id"] == true_reference_id
        )
        return {
            **task,
            "true_parent_rank": int(true_parent["rank"]),
            "true_parent_dct_rmsd": float(true_parent["dct_rmsd"]),
            "true_parent_best_reference_coverage": float(
                true_parent["best_reference_coverage"]
            ),
            "top_reference_id": ranked[0]["reference_id"],
            "top_dct_rmsd": float(ranked[0]["dct_rmsd"]),
            "top_best_reference_coverage": float(
                ranked[0]["best_reference_coverage"]
            ),
            "query_seconds": time.perf_counter() - query_started,
            "top_results": ranked[: max(1, int(top_k))],
        }

    rows = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(evaluate, task): task for task in tasks}
        for completed, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            rows.append(row)
            emit(
                f"[{completed}/{len(tasks)}] {row['cohort']} "
                f"Query {row['query_number']} complete: true-parent rank "
                f"{row['true_parent_rank']}, coverage "
                f"{row['true_parent_best_reference_coverage']:.0%}, "
                f"query {row['query_seconds']:.2f}s, total "
                f"{time.perf_counter() - started:.1f}s."
            )
    rows.sort(key=lambda row: (row["cohort"], row["query_number"]))
    cohorts = sorted({row["cohort"] for row in rows})
    summaries = {
        "combined": _batch_summary(rows),
        "cohorts": {
            cohort: _batch_summary(
                [row for row in rows if row["cohort"] == cohort]
            )
            for cohort in cohorts
        },
    }
    report = {
        "schema_version": DCT_BASELINE_SCHEMA_VERSION,
        "algorithm_version": DCT_BASELINE_ALGORITHM_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project": str(project_path),
        "manifests": [str(path) for path in manifest_paths],
        "query_count": len(rows),
        "reference_count": len(references),
        "workers": workers,
        "top_k_saved": max(1, int(top_k)),
        "elapsed_seconds": time.perf_counter() - started,
        "config": config.provenance(),
        "summary": summaries,
        "rows": rows,
    }
    (output_dir / files[0]).write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    flat_rows = [
        {key: value for key, value in row.items() if key != "top_results"}
        for row in rows
    ]
    _write_csv(output_dir / files[1], flat_rows)
    (output_dir / files[2]).write_text(
        _batch_summary_markdown(report), encoding="utf-8"
    )
    emit(
        f"Finished {len(rows)} queries in {report['elapsed_seconds']:.1f}s. "
        f"Results: {output_dir}"
    )
    return report


__all__ = [
    "DCT_BASELINE_ALGORITHM_VERSION",
    "DCT_BASELINE_SCHEMA_VERSION",
    "DCTConfig",
    "DCTReferenceBank",
    "artifact_descriptor",
    "build_reference_bank",
    "dct_descriptor",
    "dct_reconstruction",
    "run_query_experiment",
    "run_manifest_batch_experiment",
    "run_synthetic_experiment",
    "score_query",
]
