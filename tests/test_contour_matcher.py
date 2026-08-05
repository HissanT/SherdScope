import hashlib
import builtins
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from catalog.contours import (
    _centreline_reliability,
    approve_all_flags,
    auto_query_wall_curves_from_fracture,
    build_contour_artifact,
    build_manual_query_artifact,
    build_reference_library,
    find_contours_by_citation,
    library_status,
    read_manifest,
    write_manifest,
)
from catalog.matcher import (
    COARSE_LEVELS,
    MatcherError,
    REFERENCE_RIM_SPLITS,
    RETRIEVAL_KEEP,
    RETRIEVAL_METHOD_KEEP,
    _completeness_penalty,
    _diagnostic_transform,
    _match_one,
    _query_curves_for_rim_seam,
    _query_retrieval_descriptors,
    _reference_curves_for_rim_split,
    _shared_monotonic_transport,
    _srfgw,
    _weighted_similarity,
    _reference_metadata,
    _score_candidates_parallel,
    preprocess_query,
    retrieve_candidates,
    run_match,
)


def silhouette(*, dark=False, lip=True, angle=0, scale=1.0):
    canvas = Image.new("L", (360, 360), 255 if dark else 0)
    draw = ImageDraw.Draw(canvas)
    ys = np.linspace(55, 300, 90)
    centre = 180 + 13 * np.sin((ys - 55) / 245 * np.pi)
    left = centre - 27 - 5 * np.sin((ys - 55) / 245 * np.pi * 1.4)
    if lip:
        left -= 16 * np.exp(-((ys - 145) / 13) ** 2)
    right = centre + 24 + 3 * np.sin((ys - 55) / 245 * np.pi)
    polygon = list(zip(left, ys)) + list(zip(right[::-1], ys[::-1]))
    draw.polygon(polygon, fill=0 if dark else 255)
    if scale != 1.0:
        size = max(20, int(round(360 * scale)))
        canvas = canvas.resize((size, size), Image.Resampling.LANCZOS)
    if angle:
        canvas = canvas.rotate(
            angle,
            resample=Image.Resampling.BICUBIC,
            expand=True,
            fillcolor=255 if dark else 0,
        )
    return canvas


def drawn_query_curves(*, lip=True):
    ys = np.linspace(55, 300, 90)
    centre = 180 + 13 * np.sin((ys - 55) / 245 * np.pi)
    left = centre - 27 - 5 * np.sin((ys - 55) / 245 * np.pi * 1.4)
    if lip:
        left -= 16 * np.exp(-((ys - 145) / 13) ** 2)
    right = centre + 24 + 3 * np.sin((ys - 55) / 245 * np.pi)
    return {
        "exterior": np.column_stack((left, ys)).tolist(),
        "interior": np.column_stack((right, ys)).tolist(),
        "fracture": np.column_stack(
            (np.linspace(left[-1], right[-1], 12), np.full(12, ys[-1]))
        ).tolist(),
    }


def reviewed_project(tmp_path: Path, statuses):
    project = tmp_path / "project"
    cards = project / "cards"
    accepted = cards / "profiles" / "accepted"
    accepted.mkdir(parents=True)
    profiles = {}
    for index, status in enumerate(statuses):
        filename = f"sherd_{index}.png"
        card = silhouette()
        card.save(cards / filename)
        mask = silhouette()
        mask.save(accepted / f"sherd_{index}_profile.png")
        profiles[filename] = {
            "filename": filename,
            "review_status": status,
            "accepted_mask": f"profiles/accepted/sherd_{index}_profile.png",
        }
    (cards / "profile_review.json").write_text(
        json.dumps({"schema_version": 1, "profiles": profiles}),
        encoding="utf-8",
    )
    return project


def test_foreground_polarity_produces_equivalent_curves():
    light = build_contour_artifact(
        silhouette(dark=False),
        reference_id="light",
        source_filename="light.png",
    )
    dark = build_contour_artifact(
        silhouette(dark=True),
        reference_id="dark",
        source_filename="dark.png",
    )
    for name in ("wall_a", "wall_b", "centreline"):
        assert np.allclose(light["curves"][name], dark["curves"][name], atol=1e-7)
    assert not light["qc"]["foreground_dark"]
    assert dark["qc"]["foreground_dark"]


def test_smoothing_is_capped_and_keeps_salient_features():
    artifact = build_contour_artifact(
        silhouette(lip=True),
        reference_id="lip",
        source_filename="lip.png",
    )
    assert artifact["qc"]["max_displacement"] <= 0.750001
    assert artifact["salient_indices"]["wall_a"]
    assert len(artifact["curves"]["wall_a"]) == 96
    assert len(artifact["curves"]["centreline"]) == 96
    assert "fracture_start" not in artifact["curves"]
    assert "fracture_end" not in artifact["curves"]
    assert artifact["qc"]["boundary_roles"]["automatic_fracture_segmentation"] is False


def test_reference_rim_splits_are_distinct_and_preserve_one_boundary():
    artifact = build_contour_artifact(
        silhouette(lip=True),
        reference_id="reference",
        source_filename="reference.png",
    )
    master = np.asarray(
        artifact["reference_master_boundary"]["points"], dtype=float
    )
    split_curves = {
        label: _reference_curves_for_rim_split(
            artifact, offset, 192
        )
        for label, offset in REFERENCE_RIM_SPLITS
    }
    assert [label for label, _ in REFERENCE_RIM_SPLITS] == ["A", "B", "C"]
    assert np.linalg.norm(
        split_curves["A"]["wall_a"][0] - split_curves["B"]["wall_a"][0]
    ) > 0.02
    assert np.linalg.norm(
        split_curves["B"]["wall_a"][0] - split_curves["C"]["wall_a"][0]
    ) > 0.02

    for curves in split_curves.values():
        reconstructed = np.vstack(
            (curves["wall_a"][::-1], curves["wall_b"][1:])
        )
        forward = np.mean(
            np.min(
                np.linalg.norm(
                    reconstructed[:, None, :] - master[None, :, :], axis=2
                ),
                axis=1,
            )
        )
        assert forward < 0.004


def test_cheap_retrieval_is_similarity_invariant_and_cached(tmp_path):
    correct = build_contour_artifact(
        silhouette(lip=True),
        reference_id="correct",
        source_filename="correct.png",
        source_hash="correct-hash",
    )
    wrong = build_contour_artifact(
        silhouette(lip=False),
        reference_id="wrong",
        source_filename="wrong.png",
        source_hash="wrong-hash",
    )
    query = json.loads(json.dumps(correct))
    query["reference_id"] = "query"
    master = np.asarray(
        query["reference_master_boundary"]["points"], dtype=float
    )
    angle = np.deg2rad(31.0)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    transformed = 2.7 * (master @ rotation.T) + np.array([4.0, -3.0])
    query["reference_master_boundary"]["points"] = transformed.tolist()

    original_descriptor = _query_retrieval_descriptors(correct)
    transformed_descriptor = _query_retrieval_descriptors(query)
    assert np.allclose(
        original_descriptor, transformed_descriptor, atol=2e-5
    )
    from catalog.matcher import _query_local_shape_descriptors
    assert np.allclose(
        _query_local_shape_descriptors(correct),
        _query_local_shape_descriptors(query),
        atol=2e-4,
    )

    selected, first = retrieve_candidates(
        tmp_path, query, [wrong, correct], keep=1
    )
    assert selected[0]["artifact"]["reference_id"] == "correct"
    assert selected[0]["retrieval"]["outline_rank"] >= 1
    assert selected[0]["retrieval"]["ribbon_rank"] >= 1
    assert selected[0]["retrieval"]["selected_by"]
    assert first["cache_hit"] is False
    assert first["input_count"] == 2
    assert first["kept_count"] == 1
    assert first["methods"] == [
        "continuous_outline",
        "split_flexible_ribbon",
    ]

    selected_again, second = retrieve_candidates(
        tmp_path, query, [wrong, correct], keep=1
    )
    assert selected_again[0]["artifact"]["reference_id"] == "correct"
    assert second["cache_hit"] is True


def test_retrieval_safety_pool_and_coarse_levels_are_expanded():
    assert RETRIEVAL_KEEP == 400
    assert RETRIEVAL_METHOD_KEEP == 150
    assert COARSE_LEVELS == ((24, 40), (48, 12), (96, 5))


def test_metadata_can_evict_implausible_diameter_before_retrieval_top_k(
    tmp_path, monkeypatch
):
    references = [
        {"reference_id": f"r{index:03d}", "source_filename": f"r{index:03d}.png"}
        for index in range(100)
    ]
    scores = np.arange(100, dtype=float) / 1000.0
    monkeypatch.setattr(
        "catalog.matcher._load_or_build_retrieval_index",
        lambda project, refs: (
            np.zeros((100, 1)), np.zeros((100, 1)),
            [row["reference_id"] for row in refs], False, 0.0,
        ),
    )
    monkeypatch.setattr("catalog.matcher._descriptor_scores", lambda *args: scores.copy())
    monkeypatch.setattr("catalog.matcher._query_retrieval_descriptors", lambda query: np.zeros(1))
    monkeypatch.setattr("catalog.matcher._query_ribbon_retrieval_descriptors", lambda query: np.zeros(1))

    shape_only, shape_diagnostics = retrieve_candidates(
        tmp_path, {}, references, keep=1
    )
    assert shape_only[0]["artifact"]["reference_id"] == "r000"
    assert shape_diagnostics["metadata_used"] is False

    metadata = {
        row["reference_id"]: {
            "Rim Diameter (cm)": "50" if row["reference_id"] == "r000" else "20"
        }
        for row in references
    }
    combined, combined_diagnostics = retrieve_candidates(
        tmp_path,
        {},
        references,
        keep=1,
        query_metadata={"rim_diameter_cm": {"value": "20", "reliability": .9}},
        reference_metadata=metadata,
    )
    assert combined[0]["artifact"]["reference_id"] != "r000"
    assert combined[0]["retrieval"]["metadata_rank"] == 1
    assert combined_diagnostics["metadata_used"] is True


def test_persistent_curvature_keeps_real_small_bump_but_downweights_stair_steps():
    from catalog.matcher import _persistent_curvature_features

    y = np.linspace(0.0, 1.0, 96)
    smooth = np.column_stack((np.zeros_like(y), y))
    bumped = smooth.copy()
    bumped[:, 0] += 0.035 * np.exp(-0.5 * ((y - 0.45) / 0.045) ** 2)
    stair = smooth.copy()
    stair[:, 0] += 0.004 * ((np.arange(len(y)) % 2) * 2 - 1)

    bump_detail = np.max(np.abs(_persistent_curvature_features(bumped)[:, 0]))
    stair_detail = np.max(np.abs(_persistent_curvature_features(stair)[:, 0]))
    assert bump_detail > stair_detail


def test_cascade_keeps_score_top_k_and_appends_retrieval_champions():
    from catalog.matcher import _select_cascade_survivors

    results = []
    for index in range(20):
        results.append({
            "reference_id": f"r{index:02d}",
            "overall_score": index / 100.0,
            "retrieval": {
                "outline_rank": 100 + index,
                "ribbon_rank": 100 + index,
                "local_shape_rank": 100 + index,
            },
        })
    results[15]["retrieval"]["outline_rank"] = 1
    results[16]["retrieval"]["ribbon_rank"] = 1

    protected = _select_cascade_survivors(results, keep=8, reserve=1)
    protected_ids = [item["reference_id"] for item in protected]
    assert set(f"r{index:02d}" for index in range(8)) <= set(protected_ids)
    assert {"r15", "r16"} <= set(protected_ids)
    assert len(protected_ids) == 10
    final = _select_cascade_survivors(results, keep=5, reserve=0)
    assert [item["reference_id"] for item in final] == [
        "r00", "r01", "r02", "r03", "r04"
    ]


def test_parallel_candidate_scoring_returns_same_rank_as_single_worker(monkeypatch):
    def fake_score(query, candidate, samples, exact_transport):
        identifier = candidate["artifact"]["reference_id"]
        return {
            "reference_id": identifier,
            "overall_score": float(candidate["artifact"]["score"]),
        }

    monkeypatch.setattr("catalog.matcher._score_candidate_level", fake_score)
    candidates = [
        {
            "artifact": {"reference_id": identifier, "score": score},
            "retrieval": {},
        }
        for identifier, score in (("c", 0.3), ("a", 0.1), ("b", 0.2))
    ]
    single = _score_candidates_parallel(
        {}, candidates, 24, exact_transport=False,
        level_index=0, workers=1,
    )
    parallel = _score_candidates_parallel(
        {}, candidates, 24, exact_transport=False,
        level_index=0, workers=3,
    )
    key = lambda item: (item["overall_score"], item["reference_id"])
    assert [item["reference_id"] for item in sorted(single, key=key)] == [
        item["reference_id"] for item in sorted(parallel, key=key)
    ] == ["a", "b", "c"]


def test_completeness_penalty_is_mild_and_monotonic():
    assert _completeness_penalty(1.0) == 0.0
    assert _completeness_penalty(0.8) < _completeness_penalty(0.5)
    assert _completeness_penalty(0.5) < _completeness_penalty(0.2)
    assert _completeness_penalty(0.5) == pytest.approx(0.25)


def test_terminal_centreline_reliability_does_not_change_wall_features():
    y = np.linspace(0.0, 100.0, 96)
    wall_a = np.column_stack((4.0 * np.sin(y / 24.0), y))
    wall_b = np.column_stack((20.0 + 3.0 * np.sin(y / 24.0), y))
    wall_b[-12:, 0] += np.linspace(0.0, 28.0, 12)
    before_a = wall_a.copy()
    before_b = wall_b.copy()
    confidence, qc = _centreline_reliability(wall_a, wall_b)
    assert np.array_equal(wall_a, before_a)
    assert np.array_equal(wall_b, before_b)
    assert confidence[len(confidence) // 2] == 1.0
    assert confidence[-1] < 0.1
    assert qc["terminal_trim_end"] > 0


def test_fracture_curve_shape_does_not_drive_match():
    reference = build_contour_artifact(
        silhouette(lip=True),
        reference_id="reference",
        source_filename="reference.png",
    )
    diverted = build_manual_query_artifact(
        silhouette(lip=True),
        reference_id="diverted",
        source_filename="diverted.png",
        manual_curves=drawn_query_curves(lip=True),
    )
    diverted["reference_id"] = "diverted"
    fracture = np.asarray(diverted["curves"]["fracture"], dtype=float)
    fracture[:, 0] += np.linspace(2.0, 0.0, len(fracture))
    diverted["curves"]["fracture"] = fracture.tolist()
    result = _match_one(
        diverted,
        reference,
        24,
        exact_transport=False,
    )
    assert result["ribbon_cost"] < 0.08
    assert result["matched_reference_fraction"] >= 0.20
    assert abs(result["alignment"]["rotation_degrees"]) <= 45.0


def test_auto_query_wall_curves_split_silhouette_by_fracture():
    curves = drawn_query_curves(lip=True)
    rim_point = curves["exterior"][0]
    traced = auto_query_wall_curves_from_fracture(
        silhouette(lip=True),
        {"fracture": curves["fracture"], "rim_point": rim_point},
    )

    assert set(traced) == {
        "exterior",
        "interior",
        "fracture",
        "rim_point",
        "master_boundary",
    }
    assert len(traced["exterior"]) == 96
    assert len(traced["interior"]) == 96
    exterior = np.asarray(traced["exterior"], dtype=float)
    turning = np.linalg.norm(np.diff(exterior, n=2, axis=0), axis=1)
    assert float(np.quantile(turning, 0.95)) < 2.5
    artifact = build_manual_query_artifact(
        silhouette(lip=True),
        reference_id="auto",
        source_filename="auto.png",
        manual_curves=traced,
    )
    assert len(artifact["curves"]["wall_a"]) == 96
    assert len(artifact["curves"]["wall_b"]) == 96


def test_matcher_caps_large_rotation_and_scores_terminal_patch_poorly():
    reference = build_contour_artifact(
        silhouette(lip=True),
        reference_id="reference",
        source_filename="reference.png",
    )
    query = build_manual_query_artifact(
        silhouette(lip=True),
        reference_id="query",
        source_filename="query.png",
        manual_curves=drawn_query_curves(lip=True),
    )
    query["reference_id"] = "query"
    for name in ("wall_a", "wall_b"):
        points = np.asarray(query["curves"][name], dtype=float)
        rotated = np.column_stack((-points[:, 1], points[:, 0]))
        query["curves"][name] = rotated.tolist()
    master = np.asarray(query["query_master_boundary"]["points"], dtype=float)
    query["query_master_boundary"]["points"] = np.column_stack(
        (-master[:, 1], master[:, 0])
    ).tolist()

    result = _match_one(query, reference, 24, exact_transport=False)
    assert abs(result["alignment"]["rotation_degrees"]) <= 45.0
    assert result["overall_score"] > 0.45


def test_shared_transport_projection_is_ordered_and_couples_walls():
    query_labels = np.array([0] * 24 + [1] * 24)
    reference_labels = np.array([0] * 17 + [1] * 16)
    soft = np.zeros((len(query_labels), len(reference_labels)), dtype=float)
    mass = np.full(len(query_labels), 1.0 / len(query_labels))
    for query_index, label in enumerate(query_labels):
        candidates = np.flatnonzero(reference_labels == label)
        # Deliberately non-monotonic and many-to-one input from a soft solver.
        target = candidates[(query_index * 7) % len(candidates)]
        soft[query_index, target] = mass[query_index]

    transport, diagnostics = _shared_monotonic_transport(
        soft, query_labels, reference_labels, mass
    )

    assert transport.shape == soft.shape
    assert np.allclose(transport.sum(axis=1), mass)
    assert diagnostics["ordered"] is True
    assert diagnostics["shared_across_walls"] is True
    assert diagnostics["backward_steps"] == 0
    assert diagnostics["query_mass_coverage"] == pytest.approx(1.0)


def test_missing_pot_stops_matching_instead_of_using_fake_fallback(monkeypatch):
    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "ot":
            raise ImportError("POT intentionally unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    labels = np.array([0, 1])
    with pytest.raises(MatcherError, match="POT is required"):
        _srfgw(
            np.zeros((2, 2)),
            np.zeros((2, 2)),
            np.zeros((2, 2)),
            labels,
            labels,
        )


def test_cleaned_contours_keep_source_orientation_and_match_wall_swaps():
    reference = build_contour_artifact(
        silhouette(lip=True, angle=38),
        reference_id="reference",
        source_filename="reference.png",
    )
    assert reference["normalization"]["rotation_radians"] == 0.0
    assert reference["qc"]["boundary_roles"]["automatic_fracture_segmentation"] is False

    swapped = json.loads(json.dumps(reference))
    swapped["reference_id"] = "swapped"
    swapped["curves"]["wall_a"], swapped["curves"]["wall_b"] = (
        swapped["curves"]["wall_b"],
        swapped["curves"]["wall_a"],
    )
    master = swapped["reference_master_boundary"]
    master["points"] = master["points"][::-1]
    master["source_points"] = master["source_points"][::-1]
    master["nominal_seam_fraction"] = (
        1.0 - float(master["nominal_seam_fraction"])
    )
    result = _match_one(reference, swapped, 24)
    assert result["wall_swap"] is True
    assert result["overall_score"] < 0.25


def test_library_requires_resolved_profiles_and_preserves_masks(tmp_path):
    project = reviewed_project(tmp_path, ["approved", "pending"])
    status = library_status(project)
    assert not status["ready_to_build"]
    assert status["pending"] == 1

    state_path = project / "cards" / "profile_review.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["profiles"]["sherd_1.png"]["review_status"] = "no_profile"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    mask_path = project / "cards" / "profiles" / "accepted" / "sherd_0_profile.png"
    before = hashlib.sha256(mask_path.read_bytes()).hexdigest()
    result = build_reference_library(project)
    after = hashlib.sha256(mask_path.read_bytes()).hexdigest()
    assert before == after
    assert result["built"] == 1
    manifest_path = project / "matcher" / "contours" / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest["references"]["sherd_0.png"]
    assert (manifest_path.parent / entry["clean_mask"]).exists()


def test_library_rebuild_preserves_unchanged_canonical_contours(tmp_path):
    project = reviewed_project(tmp_path, ["approved"])
    first = build_reference_library(project)
    manifest = read_manifest(project)
    artifact = project / "matcher" / "contours" / manifest["references"]["sherd_0.png"]["artifact"]
    before = hashlib.sha256(artifact.read_bytes()).hexdigest()

    second = build_reference_library(project)

    assert first["built_now"] == 1
    assert second["built_now"] == 0
    assert second["preserved_existing"] == 1
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == before


def test_approve_all_flags_accepts_ready_and_leaves_failed_unresolved(tmp_path):
    project = reviewed_project(tmp_path, ["approved", "approved"])
    build_reference_library(project)
    manifest = read_manifest(project)
    entries = list(manifest["references"].values())
    for entry in entries:
        entry["flagged"] = True
        entry["review_resolution"] = None
    entries[1]["state"] = "failed"
    write_manifest(project, manifest)

    result = approve_all_flags(project)

    saved = read_manifest(project)
    saved_entries = list(saved["references"].values())
    assert result["approved"] == 1
    assert result["skipped_failed"] == 1
    assert saved_entries[0]["review_resolution"] == "accepted"
    assert saved_entries[1]["review_resolution"] is None


def test_diagnostic_coordinates_keep_source_y_direction():
    transform = _diagnostic_transform(
        [np.array([[0.0, 10.0], [0.0, 20.0]])],
        size=100,
        margin=10,
    )
    rendered = transform(np.array([[0.0, 10.0], [0.0, 20.0]]))
    assert rendered[0, 1] < rendered[1, 1]


def test_contour_lookup_uses_saved_figure_and_item(tmp_path):
    project = reviewed_project(tmp_path, ["approved"])
    (project / "cards" / "mask_info.csv").write_text(
        "mask_file,Figure,No.\n"
        "sherd_0,3.17,10\n",
        encoding="utf-8",
    )
    build_reference_library(project)
    matches = find_contours_by_citation(project, "Figure 3.17", "Item 10")
    assert len(matches) == 1
    assert matches[0]["citation_label"] == "Figure 3.17 Item 10"
    assert matches[0]["preview"]
    assert matches[0]["clean_mask"]


def test_reference_metadata_preserves_dotted_pdf_card_names(tmp_path):
    project = reviewed_project(tmp_path, ["approved"])
    card = "Hesban_Complement_3.1-3.50_page_0_mask_layer_0"
    (project / "cards" / "mask_info.csv").write_text(
        f"mask_file,Figure,No.\n{card},3.10,1\n", encoding="utf-8")

    metadata = _reference_metadata(project)

    assert metadata[card]["citation_label"] == "Figure 3.10 Item 1"


def test_rotation_scale_and_polarity_do_not_change_self_match():
    reference = build_contour_artifact(
        silhouette(lip=True),
        reference_id="reference",
        source_filename="reference.png",
    )
    query = build_contour_artifact(
        silhouette(dark=True, lip=True, angle=37, scale=0.72),
        reference_id="query",
        source_filename="query.png",
    )
    wrong = build_contour_artifact(
        silhouette(dark=True, lip=False, angle=-18, scale=1.15),
        reference_id="wrong",
        source_filename="wrong.png",
    )
    correct_result = _match_one(query, reference, 24)
    wrong_result = _match_one(query, wrong, 24)
    assert correct_result["query_coverage"] == 1.0
    assert correct_result["overall_score"] < wrong_result["overall_score"]
    assert correct_result["salience_penalty"] < wrong_result["salience_penalty"]
    assert correct_result["transport"]["backward_steps"] == 0
    assert correct_result["transport"]["shared_across_walls"] is True
    assert correct_result["fgw_cost"] == pytest.approx(
        0.45 * correct_result["rtc_feature_cost"]
        + 0.55 * correct_result["structural_gw_cost"]
    )
    assert correct_result["fgw_cost"] != pytest.approx(
        correct_result["rtc_feature_cost"]
    )
    assert correct_result["overall_score"] == pytest.approx(
        sum(correct_result["score_components"].values())
    )
    assert "completeness" in correct_result["score_components"]
    assert correct_result["score_components"]["completeness"] == pytest.approx(
        0.06
        * _completeness_penalty(
            correct_result["matched_reference_fraction"]
        )
    )


def test_similarity_alignment_forbids_reflection():
    source = np.array([[0.0, 0.0], [1.0, 0.0], [0.2, 1.0], [0.8, 0.4]])
    reflected = source * np.array([-1.0, 1.0])
    scale, rotation, translation = _weighted_similarity(
        source, reflected, np.ones(len(source))
    )
    aligned = scale * (source @ rotation.T) + translation
    assert np.linalg.det(rotation) > 0
    assert np.sqrt(np.mean(np.sum((aligned - reflected) ** 2, axis=1))) > 0.1


def test_joint_alignment_recovers_similarity_transform_and_never_worsens_objective():
    reference = build_contour_artifact(
        silhouette(lip=True),
        reference_id="reference",
        source_filename="reference.png",
    )
    query = json.loads(json.dumps(reference))
    query["reference_id"] = "transformed-query"
    angle = np.deg2rad(31.0)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    for name in ("wall_a", "wall_b", "centreline"):
        points = np.asarray(query["curves"][name], dtype=float)
        query["curves"][name] = (
            1.6 * (points @ rotation.T) + np.array([0.28, -0.17])
        ).tolist()

    result = _match_one(query, reference, 48)
    assert result["alignment"]["rotation_degrees"] == pytest.approx(-31.0, abs=0.75)
    assert result["alignment"]["scale"] == pytest.approx(1.0 / 1.6, rel=0.04)
    assert result["transport"]["backward_steps"] == 0
    assert result["joint_alignment"]["iterations"] >= 1
    assert (
        result["joint_alignment"]["objective"]
        <= result["joint_alignment"]["history"][0]["objective"] + 1e-8
    )
    assert result["overall_score"] < 0.08


def test_joint_alignment_rejects_catastrophic_scale():
    reference = build_contour_artifact(
        silhouette(lip=True),
        reference_id="reference",
        source_filename="reference.png",
    )
    query = json.loads(json.dumps(reference))
    query["reference_id"] = "catastrophic-scale"
    for name in ("wall_a", "wall_b", "centreline"):
        points = np.asarray(query["curves"][name], dtype=float)
        query["curves"][name] = (points * 0.04).tolist()

    with pytest.raises(MatcherError, match="scale"):
        _match_one(query, reference, 24)


def test_gold_point_seam_shift_does_not_change_self_match_materially():
    reference = build_contour_artifact(
        silhouette(lip=True),
        reference_id="reference",
        source_filename="reference.png",
    )
    baseline_query = build_manual_query_artifact(
        silhouette(lip=True),
        reference_id="baseline",
        source_filename="baseline.png",
        manual_curves=drawn_query_curves(lip=True),
    )
    shifted_query = json.loads(json.dumps(baseline_query))
    shifted_query["reference_id"] = "shifted-gold-point"
    master = shifted_query["query_master_boundary"]
    master["annotated_seam_fraction"] = 0.37

    baseline = _match_one(
        baseline_query, reference, 24, exact_transport=False
    )
    moved = _match_one(
        shifted_query, reference, 24, exact_transport=False
    )

    assert moved["rim_seam"]["gold_point_is_hard_anchor"] is False
    assert abs(moved["overall_score"] - baseline["overall_score"]) < 0.06
    assert (
        abs(
            moved["alignment"]["rotation_degrees"]
            - baseline["alignment"]["rotation_degrees"]
        )
        < 5.0
    )
    assert moved["alignment"]["scale"] == pytest.approx(
        baseline["alignment"]["scale"], rel=0.12
    )


def test_gold_point_variants_preserve_one_master_boundary():
    curves = drawn_query_curves(lip=True)
    image = silhouette(lip=True)
    left = auto_query_wall_curves_from_fracture(
        image,
        {"fracture": curves["fracture"], "rim_point": curves["exterior"][0]},
    )
    right = auto_query_wall_curves_from_fracture(
        image,
        {"fracture": curves["fracture"], "rim_point": curves["interior"][0]},
    )
    left_artifact = build_manual_query_artifact(
        image,
        reference_id="left-gold",
        source_filename="query.png",
        manual_curves=left,
    )
    right_artifact = build_manual_query_artifact(
        image,
        reference_id="right-gold",
        source_filename="query.png",
        manual_curves=right,
    )

    left_master = np.asarray(
        left_artifact["query_master_boundary"]["points"], dtype=float
    )
    right_master = np.asarray(
        right_artifact["query_master_boundary"]["points"], dtype=float
    )
    direct = np.max(np.linalg.norm(left_master - right_master, axis=1))
    reversed_distance = np.max(
        np.linalg.norm(left_master - right_master[::-1], axis=1)
    )
    assert min(direct, reversed_distance) < 1e-8
    assert (
        left_artifact["query_master_boundary"]["split_before_wall_resampling"]
        is True
    )
    reference = build_contour_artifact(
        image,
        reference_id="reference",
        source_filename="reference.png",
    )
    left_match = _match_one(
        left_artifact, reference, 24, exact_transport=False
    )
    right_match = _match_one(
        right_artifact, reference, 24, exact_transport=False
    )
    assert abs(left_match["overall_score"] - right_match["overall_score"]) < 0.03
    assert abs(
        left_match["alignment"]["rotation_degrees"]
        - right_match["alignment"]["rotation_degrees"]
    ) < 3.0


def test_every_seam_candidate_is_cut_from_the_same_physical_boundary():
    curves = drawn_query_curves(lip=True)
    traced = auto_query_wall_curves_from_fracture(
        silhouette(lip=True),
        {"fracture": curves["fracture"], "rim_point": curves["exterior"][0]},
    )
    artifact = build_manual_query_artifact(
        silhouette(lip=True),
        reference_id="query",
        source_filename="query.png",
        manual_curves=traced,
    )
    master = np.asarray(artifact["query_master_boundary"]["points"], dtype=float)
    master_sample = np.asarray(
        _query_curves_for_rim_seam(artifact, 0.0, 192)["wall_a"][::-1].tolist()
        + _query_curves_for_rim_seam(artifact, 0.0, 192)["wall_b"][1:].tolist()
    )

    for offset in (-0.20, -0.10, 0.0, 0.10, 0.20):
        candidate = _query_curves_for_rim_seam(artifact, offset, 192)
        reconstructed = np.vstack(
            (candidate["wall_a"][::-1], candidate["wall_b"][1:])
        )
        forward = np.mean(
            np.min(
                np.linalg.norm(
                    reconstructed[:, None, :] - master[None, :, :], axis=2
                ),
                axis=1,
            )
        )
        assert forward < 0.004
        assert np.all(np.isfinite(reconstructed))
    assert len(master_sample) == 383


def test_rim_search_spans_full_allowed_neighbourhood_without_local_trap():
    reference = build_contour_artifact(
        silhouette(lip=True),
        reference_id="reference",
        source_filename="reference.png",
    )
    query = build_manual_query_artifact(
        silhouette(lip=True),
        reference_id="query",
        source_filename="query.png",
        manual_curves=drawn_query_curves(lip=True),
    )
    master = query["query_master_boundary"]
    nominal = float(master["nominal_seam_fraction"])
    available = min(nominal, 1.0 - nominal)
    master["nominal_seam_fraction"] = nominal - 0.20 * available

    result = _match_one(query, reference, 24, exact_transport=False)

    assert {-0.25, 0.0, 0.25}.issubset(
        set(result["rim_seam"]["candidate_offsets"])
    )
    assert result["rim_seam"]["split_source"] == "continuous_master_boundary"
    assert result["rim_seam"]["gold_point_is_hard_anchor"] is False


def test_exact_match_retains_diverse_hypotheses_and_reliability_score():
    reference = build_contour_artifact(
        silhouette(lip=True),
        reference_id="reference",
        source_filename="reference.png",
    )
    query = build_manual_query_artifact(
        silhouette(lip=True),
        reference_id="query",
        source_filename="query.png",
        manual_curves=drawn_query_curves(lip=True),
    )

    result = _match_one(query, reference, 48, exact_transport=True)

    assert result["hypothesis_search"]["requested"] == 1
    assert result["hypothesis_search"]["evaluated"] >= 1
    assert result["hypothesis_search"]["selected_strategy"] == "composite"
    assert result["reference_rim_split"]["label"] in {"A", "B", "C"}
    assert result["reference_rim_split"]["candidate_offsets"] == [
        -0.125,
        0.0,
        0.125,
    ]
    assert 0.0 <= result["transform_reliability"]["overall"] <= 1.0
    assert result["overall_score"] == pytest.approx(
        sum(result["score_components"].values())
    )


def test_full_match_run_persists_metrics_and_diagnostic(tmp_path):
    project = reviewed_project(tmp_path, ["approved"])
    (project / "cards" / "mask_info.csv").write_text(
        "mask_file,Figure,No.\n"
        "sherd_0,3.19,7\n",
        encoding="utf-8",
    )
    build_reference_library(project)
    query = preprocess_query(
        project,
        silhouette(dark=True),
        original_filename="query.png",
        metadata={"query_id": "Q1", "diameter": "12.5"},
        manual_curves=drawn_query_curves(),
    )
    result = run_match(project, query["query_id"])
    assert result["metadata_used"] is False
    assert len(result["results"]) == 1
    assert result["retrieval"]["input_count"] == 1
    assert result["retrieval"]["kept_count"] == 1
    candidate = result["results"][0]
    assert candidate["rank"] == 1
    assert candidate["retrieval"]["rank"] == 1
    assert candidate["citation_label"] == "Figure 3.19 Item 7"
    assert candidate["query_coverage"] == 1.0
    assert "rtc_feature_cost" in candidate
    assert "three_curve_cost" in candidate
    assert "per_curve" in candidate["alignment"]
    assert "initialization_stability" in candidate
    assert "hausdorff95" in candidate["alignment"]
    assert (project / "matcher" / "runs" / result["run_id"] / candidate["diagnostic"]).exists()
