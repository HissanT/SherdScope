import json

import numpy as np
from PIL import Image, ImageDraw

from catalog.contours import build_contour_artifact
from catalog.matcher import _master_boundary
from catalog.matcher_benchmark import (
    CONDITIONS,
    DEFAULT_TOP_K,
    _rim_oriented_boundary,
    _summary,
    synthetic_query_from_reference,
)


def test_benchmark_reports_the_full_retrieval_pool():
    assert DEFAULT_TOP_K[-1] == 400


def silhouette():
    canvas = Image.new("L", (360, 360), 0)
    draw = ImageDraw.Draw(canvas)
    ys = np.linspace(55, 300, 90)
    centre = 180 + 13 * np.sin((ys - 55) / 245 * np.pi)
    left = centre - 27 - 5 * np.sin((ys - 55) / 245 * np.pi * 1.4)
    left -= 16 * np.exp(-((ys - 145) / 13) ** 2)
    right = centre + 24 + 3 * np.sin((ys - 55) / 245 * np.pi)
    polygon = list(zip(left, ys)) + list(zip(right[::-1], ys[::-1]))
    draw.polygon(polygon, fill=255)
    return canvas


def test_synthetic_query_is_partial_deterministic_and_keeps_parent_immutable():
    reference = build_contour_artifact(
        silhouette(),
        reference_id="parent",
        source_filename="parent.png",
    )
    before = json.dumps(reference, sort_keys=True)
    first, provenance = synthetic_query_from_reference(
        reference,
        rng=np.random.default_rng(17),
        condition=CONDITIONS["light"],
        coverage=0.4,
    )
    second, _ = synthetic_query_from_reference(
        reference,
        rng=np.random.default_rng(17),
        condition=CONDITIONS["light"],
        coverage=0.4,
    )

    parent_points, _ = _master_boundary(reference)
    query_points, seam = _master_boundary(first)
    assert json.dumps(reference, sort_keys=True) == before
    assert first["query_master_boundary"] == second["query_master_boundary"]
    assert len(query_points) == 192
    assert 0.08 <= seam <= 0.92
    assert provenance["parent_reference_id"] == "parent"
    assert np.linalg.norm(np.ptp(query_points, axis=0)) < np.linalg.norm(
        np.ptp(parent_points, axis=0)
    )


def test_benchmark_summary_counts_missing_as_failure():
    rows = [
        {"rank": 1},
        {"rank": 5},
        {"rank": 6},
        {"rank": None},
    ]
    summary = _summary(rows, "rank")
    assert summary["top_1"] == 1
    assert summary["top_5"] == 2
    assert summary["top_10"] == 3
    assert summary["missing"] == 1
    assert summary["top_5_accuracy"] == 0.5


def test_clean_and_noisy_conditions_share_break_and_transform():
    reference = build_contour_artifact(
        silhouette(), reference_id="parent", source_filename="parent.png"
    )
    clean, clean_provenance = synthetic_query_from_reference(
        reference,
        rng=np.random.default_rng(91),
        condition=CONDITIONS["clean"],
    )
    noisy, noisy_provenance = synthetic_query_from_reference(
        reference,
        rng=np.random.default_rng(91),
        condition=CONDITIONS["moderate"],
    )
    for field in (
        "coverage", "left_coverage", "right_coverage",
        "rotation_degrees", "scale",
    ):
        assert clean_provenance[field] == noisy_provenance[field]
    clean_points, _ = _master_boundary(clean)
    noisy_points, _ = _master_boundary(noisy)
    assert not np.array_equal(clean_points, noisy_points)


def test_synthetic_rim_selection_uses_upper_source_cap():
    reference = build_contour_artifact(
        silhouette(), reference_id="parent", source_filename="parent.png"
    )
    master = reference["reference_master_boundary"]
    source = np.asarray(master["source_points"], dtype=float)
    seam = float(master["nominal_seam_fraction"])
    seam_index = int(round(seam * (len(source) - 1)))

    # Force the nominal seam below the opposite cap in upright source pixels.
    source[:, 1] += 100.0
    source[0, 1] = 10.0
    source[-1, 1] = 12.0
    source[max(0, seam_index - 2): seam_index + 3, 1] = 250.0
    master["source_points"] = source.tolist()

    _, _, diagnostics = _rim_oriented_boundary(reference)
    assert diagnostics["rim_selection"] == "opposite_cap"
    assert diagnostics["source_opposite_cap_y"] < diagnostics["source_nominal_cap_y"]
