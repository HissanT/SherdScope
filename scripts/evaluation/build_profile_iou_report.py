"""Evaluate automatic profile proposals against accepted reviewer masks."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROJECT = (
    ROOT / "projects" / "Finalized_Hesban_Corpus_Digitization_20260805_154332"
)
DEFAULT_REPORT = ROOT / "PROFILE_IOU_EVAL.md"
DEFAULT_DETAILS = ROOT / "outputs" / "profile_iou_evaluation" / "profile_iou_results.json"
REVIEWED_STATUSES = {"approved", "edited"}


def mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L")) > 0


def evaluate_pair(auto_path: Path, accepted_path: Path) -> dict[str, Any]:
    automatic = mask(auto_path)
    accepted = mask(accepted_path)
    if automatic.shape != accepted.shape:
        return {
            "valid": False,
            "reason": "dimension_mismatch",
            "auto_shape": list(automatic.shape),
            "accepted_shape": list(accepted.shape),
        }
    auto_pixels = int(automatic.sum())
    accepted_pixels = int(accepted.sum())
    intersection = int(np.logical_and(automatic, accepted).sum())
    union = int(np.logical_or(automatic, accepted).sum())
    if union == 0:
        return {
            "valid": False,
            "reason": "both_masks_empty",
            "auto_pixels": auto_pixels,
            "accepted_pixels": accepted_pixels,
            "intersection_pixels": intersection,
            "union_pixels": union,
        }
    iou = intersection / union
    total = auto_pixels + accepted_pixels
    return {
        "valid": True,
        "auto_pixels": auto_pixels,
        "accepted_pixels": accepted_pixels,
        "intersection_pixels": intersection,
        "union_pixels": union,
        "changed_pixels": int(np.logical_xor(automatic, accepted).sum()),
        "iou": iou,
        "dice": (2 * intersection / total) if total else None,
        "precision": (intersection / auto_pixels) if auto_pixels else 0.0,
        "recall": (intersection / accepted_pixels) if accepted_pixels else 0.0,
        "accepted_to_auto_area_ratio": (
            accepted_pixels / auto_pixels if auto_pixels else None
        ),
    }


def percentile(values: list[float], percentage: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), percentage))


def summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "population_std": statistics.pstdev(values),
        "minimum": min(values),
        "p05": percentile(values, 5),
        "p25": percentile(values, 25),
        "p75": percentile(values, 75),
        "p95": percentile(values, 95),
        "maximum": max(values),
    }


def fmt(value: float | None, digits: int = 4) -> str:
    return "—" if value is None or not math.isfinite(value) else f"{value:.{digits}f}"


def collect(project: Path) -> dict[str, Any]:
    cards = project / "cards"
    review_path = cards / "profile_review.json"
    with open(review_path, encoding="utf-8") as handle:
        records = json.load(handle).get("profiles", {})

    rows: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}
    for filename, review in sorted(records.items()):
        status = str(review.get("review_status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        if status not in REVIEWED_STATUSES:
            continue
        stem = Path(filename).stem
        auto_path = cards / "profiles" / "auto" / f"{stem}_profile.png"
        accepted_path = cards / "profiles" / "accepted" / f"{stem}_profile.png"
        row: dict[str, Any] = {
            "filename": filename,
            "review_status": status,
            "auto_mask": auto_path.relative_to(project).as_posix(),
            "accepted_mask": accepted_path.relative_to(project).as_posix(),
        }
        if not auto_path.is_file() or not accepted_path.is_file():
            row.update(
                {
                    "valid": False,
                    "reason": "missing_auto_mask"
                    if not auto_path.is_file()
                    else "missing_accepted_mask",
                }
            )
        else:
            row.update(evaluate_pair(auto_path, accepted_path))
        rows.append(row)

    valid = [row for row in rows if row.get("valid")]
    ious = [float(row["iou"]) for row in valid]
    dice = [float(row["dice"]) for row in valid]
    precision = [float(row["precision"]) for row in valid]
    recall = [float(row["recall"]) for row in valid]
    ratios = [
        float(row["accepted_to_auto_area_ratio"])
        for row in valid
        if row.get("accepted_to_auto_area_ratio") is not None
    ]
    thresholds = {
        f"ge_{str(value).replace('.', '_')}": sum(iou >= value for iou in ious)
        for value in (0.95, 0.90, 0.80, 0.70, 0.50)
    }
    invalid_reasons: dict[str, int] = {}
    for row in rows:
        if row.get("valid"):
            continue
        reason = str(row.get("reason") or "unknown")
        invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1

    total_intersection = sum(int(row["intersection_pixels"]) for row in valid)
    total_union = sum(int(row["union_pixels"]) for row in valid)
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project": str(project.resolve()),
        "comparison": "automatic proposal versus accepted reviewer mask",
        "accepted_mask_role": "expert reference",
        "binarization": "grayscale pixel > 0 is foreground",
        "total_profile_records": len(records),
        "reviewed_records": len(rows),
        "valid_pairs": len(valid),
        "invalid_pairs": len(rows) - len(valid),
        "status_counts": status_counts,
        "invalid_reasons": invalid_reasons,
        "iou": summary(ious),
        "micro_iou": total_intersection / total_union if total_union else None,
        "zero_overlap_pairs": sum(value == 0.0 for value in ious),
        "dice": summary(dice),
        "precision": summary(precision),
        "recall": summary(recall),
        "accepted_to_auto_area_ratio": summary(ratios),
        "threshold_counts": thresholds,
        "rows": rows,
    }


def markdown(result: dict[str, Any]) -> str:
    valid = int(result["valid_pairs"])
    iou = result["iou"]
    dice = result["dice"]
    precision = result["precision"]
    recall = result["recall"]
    ratio = result["accepted_to_auto_area_ratio"]
    thresholds = result["threshold_counts"]
    valid_rows = sorted(
        (row for row in result["rows"] if row.get("valid")),
        key=lambda row: (float(row["iou"]), row["filename"]),
    )
    lines = [
        "# Diagnostic profile IoU evaluation",
        "",
        "## Evaluation scope",
        "",
        f"- Project: `{Path(result['project']).name}`",
        f"- Generated: `{result['generated_at']}`",
        f"- Total profile records: **{result['total_profile_records']:,}**",
        f"- Reviewed profiles evaluated: **{result['reviewed_records']:,}**",
        f"- Valid automatic/accepted pairs: **{valid:,}**",
        f"- Invalid or excluded pairs: **{result['invalid_pairs']:,}**",
        f"- Zero-overlap pairs: **{result['zero_overlap_pairs']:,}**",
        "- Before mask: preserved automatic proposal in `cards/profiles/auto`.",
        "- After mask: accepted reviewer mask in `cards/profiles/accepted`.",
        "- Foreground definition: grayscale value greater than zero.",
        "- Interpretation: accepted masks are treated as expert reference masks. These values measure automatic segmentation agreement with the reviewed result.",
        "",
        "## Headline metrics",
        "",
        "| Metric | Mean | Median | Std. dev. | Minimum | P05 | P25 | P75 | P95 | Maximum |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| IoU | {fmt(iou['mean'])} | {fmt(iou['median'])} | {fmt(iou['population_std'])} | {fmt(iou['minimum'])} | {fmt(iou['p05'])} | {fmt(iou['p25'])} | {fmt(iou['p75'])} | {fmt(iou['p95'])} | {fmt(iou['maximum'])} |",
        f"| Dice | {fmt(dice['mean'])} | {fmt(dice['median'])} | {fmt(dice['population_std'])} | {fmt(dice['minimum'])} | {fmt(dice['p05'])} | {fmt(dice['p25'])} | {fmt(dice['p75'])} | {fmt(dice['p95'])} | {fmt(dice['maximum'])} |",
        f"| Precision | {fmt(precision['mean'])} | {fmt(precision['median'])} | {fmt(precision['population_std'])} | {fmt(precision['minimum'])} | {fmt(precision['p05'])} | {fmt(precision['p25'])} | {fmt(precision['p75'])} | {fmt(precision['p95'])} | {fmt(precision['maximum'])} |",
        f"| Recall | {fmt(recall['mean'])} | {fmt(recall['median'])} | {fmt(recall['population_std'])} | {fmt(recall['minimum'])} | {fmt(recall['p05'])} | {fmt(recall['p25'])} | {fmt(recall['p75'])} | {fmt(recall['p95'])} | {fmt(recall['maximum'])} |",
        "",
        f"The foreground-weighted micro-IoU is **{result['micro_iou']:.4f}**. Macro-IoU above gives every profile equal weight; micro-IoU gives larger masks more influence.",
        "",
        "## IoU threshold distribution",
        "",
        "| Threshold | Profiles | Percentage |",
        "|---|---:|---:|",
    ]
    for value in (0.95, 0.90, 0.80, 0.70, 0.50):
        count = thresholds[f"ge_{str(value).replace('.', '_')}"]
        lines.append(f"| IoU ≥ {value:.2f} | {count:,} | {count / valid:.1%} |")
    lines.extend(
        [
            "",
            "## Mask-area change",
            "",
            f"The accepted mask contained a median of **{ratio['median']:.3f}×** the automatic-mask foreground area (mean **{ratio['mean']:.3f}×**). Values above 1 indicate that review added foreground; values below 1 indicate that review removed foreground.",
            "",
            "## Lowest-IoU reviewed profiles",
            "",
            "These cases required the greatest pixel-level change and are useful for qualitative failure analysis.",
            "",
            "| Rank | Profile | IoU | Dice | Precision | Recall | Auto pixels | Accepted pixels | Area ratio |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for index, row in enumerate(valid_rows[:50], start=1):
        lines.append(
            f"| {index} | `{row['filename']}` | {row['iou']:.4f} | "
            f"{row['dice']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | "
            f"{row['auto_pixels']:,} | {row['accepted_pixels']:,} | "
            f"{fmt(row.get('accepted_to_auto_area_ratio'), 3)} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundaries",
            "",
            "- IoU evaluates pixel overlap, so boundary corrections and removal of side branches lower the score even when the automatic mask was visually close.",
            "- This is an evaluation of the automatic proposal against the final reviewed mask, not inter-annotator agreement.",
            "- `no_profile` and empty-mask cases must be reported separately rather than mixed into the mean. None were silently assigned IoU 1.",
            "- The complete per-profile measurements are preserved in `outputs/profile_iou_evaluation/profile_iou_results.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--details", type=Path, default=DEFAULT_DETAILS)
    args = parser.parse_args()
    result = collect(args.project.resolve())
    args.details.parent.mkdir(parents=True, exist_ok=True)
    args.details.write_text(json.dumps(result, indent=2), encoding="utf-8")
    args.report.write_text(markdown(result), encoding="utf-8")
    print(args.report.resolve())
    print(args.details.resolve())


if __name__ == "__main__":
    main()
