"""Render exact, seeded examples from a saved synthetic benchmark."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from catalog.contours import load_ready_artifacts
from catalog.matcher import _master_boundary
from catalog.matcher_benchmark import CONDITIONS, synthetic_query_from_reference


COLORS = {
    "grid_clean": (0, 110, 210),
    "clean": (0, 145, 85),
    "light": (220, 135, 0),
    "moderate": (210, 45, 55),
    "partial_75": (80, 85, 200),
    "partial_50": (130, 70, 180),
    "partial_25": (175, 45, 130),
}


def _font(size: int, *, bold: bool = False):
    names = (
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
    )
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _inverse_synthetic_transform(
    points: np.ndarray, provenance: dict[str, Any]
) -> np.ndarray:
    angle = math.radians(float(provenance["rotation_degrees"]))
    rotation = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=float,
    )
    translation = np.asarray(provenance["translation"], dtype=float)
    scale = float(provenance["scale"])
    return ((points - translation) / scale) @ rotation


def _screen_transform(parent: np.ndarray, width: int, height: int, margin: int):
    lower = np.min(parent, axis=0)
    upper = np.max(parent, axis=0)
    span = np.maximum(upper - lower, 1e-9)
    scale = min((width - 2 * margin) / span[0], (height - 2 * margin) / span[1])
    offset = np.array(
        [
            (width - scale * span[0]) / 2.0,
            (height - scale * span[1]) / 2.0,
        ]
    )

    def transform(points: np.ndarray) -> list[tuple[int, int]]:
        values = (np.asarray(points) - lower) * scale + offset
        return [(int(round(x)), int(round(y))) for x, y in values]

    return transform


def _cell(
    reference: dict[str, Any],
    query: dict[str, Any] | None,
    provenance: dict[str, Any] | None,
    *,
    condition: str,
    rank: int | None,
    show_rank: bool = True,
    width: int = 330,
    height: int = 430,
) -> Image.Image:
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = _font(20, bold=True)
    small_font = _font(15)
    parent, _ = _master_boundary(reference)
    transform = _screen_transform(parent, width, height - 82, 25)
    draw.line(transform(parent), fill=(188, 194, 204), width=7, joint="curve")
    if query is not None and provenance is not None:
        query_points, _ = _master_boundary(query)
        aligned = _inverse_synthetic_transform(query_points, provenance)
        draw.line(
            transform(aligned),
            fill=COLORS[condition],
            width=5,
            joint="curve",
        )
        coverage = float(provenance["coverage"])
        if show_rank:
            rank_text = ">150" if rank is None else str(rank)
            detail = f"coverage {coverage:.2f}   retrieval rank {rank_text}"
        else:
            detail = f"coverage {coverage:.2f}   preflight: not scored"
    else:
        detail = "full approved parent contour"
    draw.rectangle((0, height - 80, width, height), fill=(247, 249, 252))
    draw.text((14, height - 70), condition.replace("_", " ").title(), fill=(18, 31, 52), font=title_font)
    draw.text((14, height - 39), detail, fill=(74, 85, 104), font=small_font)
    draw.rectangle((0, 0, width - 1, height - 1), outline=(215, 220, 228), width=2)
    return image


def render_gallery(
    project_path: Path,
    benchmark_path: Path,
    output_dir: Path,
    *,
    sample_count: int = 8,
    show_saved_ranks: bool = True,
) -> list[Path]:
    report = json.loads(Path(benchmark_path).read_text(encoding="utf-8"))
    references = load_ready_artifacts(Path(project_path))
    by_id = {item["reference_id"]: item for item in references}
    rows = report["rows"]
    ordered_parent_ids = []
    for row in rows:
        parent_id = row["parent_reference_id"]
        if parent_id not in ordered_parent_ids:
            ordered_parent_ids.append(parent_id)
    ordered_parent_ids = ordered_parent_ids[: max(1, sample_count)]

    # Reproduce numpy's exact selection and per-parent seed sequence used by
    # run_synthetic_benchmark. The selected ID order is checked against the
    # saved report so gallery images cannot silently represent another run.
    rng = np.random.default_rng(int(report["seed"]))
    selected_indices = rng.choice(
        len(references), size=int(report["selected_parent_count"]), replace=False
    )
    selected_ids = [references[int(index)]["reference_id"] for index in selected_indices]
    if selected_ids != [
        row["parent_reference_id"]
        for row in rows[:: len(report["conditions"])]
    ]:
        raise RuntimeError("The project reference order no longer reproduces this benchmark")

    examples: dict[str, dict[str, tuple[dict[str, Any], dict[str, Any]]]] = {}
    for reference in (references[int(index)] for index in selected_indices):
        paired_seed = int(rng.integers(0, np.iinfo(np.int64).max))
        if reference["reference_id"] not in ordered_parent_ids:
            continue
        examples[reference["reference_id"]] = {}
        for condition_name in report["conditions"]:
            query, provenance = synthetic_query_from_reference(
                reference,
                rng=np.random.default_rng(paired_seed),
                condition=CONDITIONS[condition_name],
            )
            examples[reference["reference_id"]][condition_name] = (query, provenance)

    row_lookup = {
        (row["parent_reference_id"], row["condition"]): row for row in rows
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    destinations = []
    conditions = list(report["conditions"])
    cell_width, cell_height = 330, 430
    header_height = 74
    sheet = Image.new(
        "RGB",
        ((len(conditions) + 1) * cell_width, header_height + len(ordered_parent_ids) * cell_height),
        (241, 244, 248),
    )
    header = ImageDraw.Draw(sheet)
    header.text(
        (20, 15),
        f"Exact examples from the paired {report['query_count']}-query synthetic run",
        fill=(15, 28, 48),
        font=_font(27, bold=True),
    )
    header.text((20, 47), "Grey = complete approved reference; colour = generated partial query before random display transform", fill=(75, 87, 105), font=_font(16))
    for row_index, parent_id in enumerate(ordered_parent_ids):
        reference = by_id[parent_id]
        y = header_height + row_index * cell_height
        sheet.paste(_cell(reference, None, None, condition="parent", rank=None), (0, y))
        for column, condition_name in enumerate(conditions, start=1):
            query, provenance = examples[parent_id][condition_name]
            rank = (
                row_lookup[(parent_id, condition_name)]["retrieval_rank"]
                if show_saved_ranks
                else None
            )
            sheet.paste(
                _cell(
                    reference, query, provenance,
                    condition=condition_name, rank=rank,
                    show_rank=show_saved_ranks,
                ),
                (column * cell_width, y),
            )
    combined_path = output_dir / "synthetic_gallery_all_conditions.png"
    sheet.save(combined_path)
    destinations.append(combined_path)

    for condition_name in conditions:
        columns = 4
        rows_needed = math.ceil(len(ordered_parent_ids) / columns)
        condition_sheet = Image.new(
            "RGB", (columns * cell_width, header_height + rows_needed * cell_height), (241, 244, 248)
        )
        condition_draw = ImageDraw.Draw(condition_sheet)
        condition_draw.text((20, 16), f"{condition_name.replace('_', ' ').title()} examples", fill=(15, 28, 48), font=_font(27, bold=True))
        for index, parent_id in enumerate(ordered_parent_ids):
            reference = by_id[parent_id]
            query, provenance = examples[parent_id][condition_name]
            rank = (
                row_lookup[(parent_id, condition_name)]["retrieval_rank"]
                if show_saved_ranks
                else None
            )
            cell = _cell(
                reference, query, provenance,
                condition=condition_name, rank=rank,
                show_rank=show_saved_ranks,
            )
            x = (index % columns) * cell_width
            y = header_height + (index // columns) * cell_height
            condition_sheet.paste(cell, (x, y))
        path = output_dir / f"synthetic_gallery_{condition_name}.png"
        condition_sheet.save(path)
        destinations.append(path)
    return destinations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Render corrected queries without displaying ranks from the prior run.",
    )
    args = parser.parse_args()
    for path in render_gallery(
        args.project,
        args.benchmark,
        args.output,
        sample_count=args.samples,
        show_saved_ranks=not args.preflight,
    ):
        print(path)


if __name__ == "__main__":
    main()
