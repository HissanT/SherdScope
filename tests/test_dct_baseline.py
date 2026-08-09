import copy
import hashlib
import json
import math

import numpy as np
import pytest
from PIL import Image, ImageDraw

import catalog.dct_baseline as dct_module
from catalog.contours import build_contour_artifact
from catalog.dct_baseline import (
    DCTConfig,
    artifact_descriptor,
    build_reference_bank,
    dct_reconstruction,
    run_manifest_batch_experiment,
    run_query_experiment,
    run_synthetic_experiment,
    score_query,
)
from catalog.matcher_benchmark import (
    CONDITIONS,
    _rim_oriented_boundary,
    synthetic_query_from_reference,
)


def silhouette(*, lip=True, width_delta=0.0):
    canvas = Image.new("L", (360, 360), 0)
    draw = ImageDraw.Draw(canvas)
    ys = np.linspace(55, 300, 90)
    centre = 180 + 13 * np.sin((ys - 55) / 245 * np.pi)
    left = centre - 27 - width_delta
    if lip:
        left -= 16 * np.exp(-((ys - 145) / 13) ** 2)
    right = centre + 24 + width_delta
    polygon = list(zip(left, ys)) + list(zip(right[::-1], ys[::-1]))
    draw.polygon(polygon, fill=255)
    return canvas


def artifact(reference_id, *, lip=True, width_delta=0.0):
    return build_contour_artifact(
        silhouette(lip=lip, width_delta=width_delta),
        reference_id=reference_id,
        source_filename=f"{reference_id}.png",
    )


def transformed_full_query(reference):
    points, seam, _ = _rim_oriented_boundary(reference)
    angle = math.radians(31.0)
    rotation = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    transformed = 1.7 * (points @ rotation.T) + np.array([0.41, -0.28])
    query = copy.deepcopy(reference)
    query["query_master_boundary"] = {
        "points": transformed.tolist(),
        "nominal_seam_fraction": seam,
        "annotated_seam_fraction": seam,
    }
    return query


def test_descriptor_is_invariant_to_translation_rotation_and_scale():
    reference = artifact("correct")
    query = transformed_full_query(reference)
    config = DCTConfig()
    expected = artifact_descriptor(reference, config)
    observed = artifact_descriptor(query, config)
    assert np.allclose(observed, expected, atol=1e-7)


def test_default_coverage_grid_excludes_values_below_half():
    coverages = DCTConfig().coverages
    assert len(coverages) == 11
    assert coverages[0] == pytest.approx(0.50)
    assert coverages[-1] == pytest.approx(1.00)
    assert np.allclose(np.diff(coverages), 0.05)


def test_coverage_search_recovers_partial_parent_and_fraction():
    correct = artifact("correct", lip=True)
    distractor = artifact("distractor", lip=False, width_delta=9.0)
    query, _ = synthetic_query_from_reference(
        correct,
        rng=np.random.default_rng(71),
        condition=CONDITIONS["partial_50"],
    )
    config = DCTConfig()
    ranked = score_query(query, build_reference_bank([distractor, correct], config))
    assert ranked[0]["reference_id"] == "correct"
    assert ranked[0]["best_reference_coverage"] == pytest.approx(0.50)
    # The synthetic generator and baseline independently resample the same
    # curve, so a small interpolation residue is expected rather than zero.
    assert ranked[0]["dct_rmsd"] < 0.005


def test_more_harmonics_reconstruct_the_outline_more_closely():
    reference = artifact("profile")
    config_low = DCTConfig(harmonics=4)
    config_high = DCTConfig(harmonics=20)
    walls = dct_module._artifact_walls(reference, 64)
    outline = dct_module._canonical_open_outline(walls, samples=100)
    low = dct_reconstruction(outline, config_low)
    high = dct_reconstruction(outline, config_high)
    assert np.mean((high - outline) ** 2) < np.mean((low - outline) ** 2)


def test_synthetic_experiment_writes_isolated_reproducible_outputs(
    tmp_path, monkeypatch
):
    references = [artifact("correct", lip=True), artifact("other", lip=False)]
    monkeypatch.setattr(dct_module, "load_ready_artifacts", lambda _project: references)
    report = run_synthetic_experiment(
        tmp_path / "project",
        tmp_path / "output",
        sample_size=1,
        seed=19,
        condition_names=("partial_50",),
        config=DCTConfig(),
    )
    assert report["query_count"] == 1
    assert report["config"]["harmonics"] == 20
    assert (tmp_path / "output" / "dct_synthetic_benchmark.json").exists()
    assert (tmp_path / "output" / "dct_synthetic_benchmark.csv").exists()
    assert (tmp_path / "output" / "summary.md").exists()
    saved = json.loads(
        (tmp_path / "output" / "dct_synthetic_benchmark.json").read_text(
            encoding="utf-8"
        )
    )
    assert saved["algorithm_version"] == "paper-rim-dct-v2-coverage50"

    with pytest.raises(FileExistsError):
        run_synthetic_experiment(
            tmp_path / "project",
            tmp_path / "output",
            sample_size=1,
            condition_names=("partial_50",),
        )


def test_single_query_experiment_writes_rankings_and_reconstruction(
    tmp_path, monkeypatch
):
    correct = artifact("correct", lip=True)
    references = [artifact("other", lip=False), correct]
    query = transformed_full_query(correct)
    query_path = tmp_path / "query.json"
    query_path.write_text(json.dumps(query), encoding="utf-8")
    monkeypatch.setattr(dct_module, "load_ready_artifacts", lambda _project: references)
    report = run_query_experiment(
        tmp_path / "project",
        tmp_path / "score_output",
        query_artifact_path=query_path,
        top_k=2,
    )
    assert report["results"][0]["reference_id"] == "correct"
    assert (tmp_path / "score_output" / "dct_results.json").exists()
    assert (tmp_path / "score_output" / "dct_results.csv").exists()
    reconstruction = tmp_path / "score_output" / "query_dct_reconstruction.png"
    assert reconstruction.exists()
    assert reconstruction.stat().st_size > 1_000


def test_manifest_batch_scores_known_parents_with_workers(tmp_path, monkeypatch):
    first = artifact("first", lip=True)
    first["figure"], first["item"] = "3.1", "8"
    second = artifact("second", lip=False, width_delta=9.0)
    second["figure"], second["item"] = "3.2", "4"
    references = [first, second]
    monkeypatch.setattr(dct_module, "load_ready_artifacts", lambda _project: references)

    project = tmp_path / "project"
    entries = {}
    specifications = [
        (1, "a" * 32, first, {"figure": "3.1", "item": "8"}),
        (2, "b" * 32, second, {"figure": "3.2", "item": "4"}),
    ]
    for number, query_id, reference, target in specifications:
        path = project / "matcher" / "queries" / query_id / "artifact.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(transformed_full_query(reference)), encoding="utf-8")
        entries[str(number)] = {
            "number": number,
            "query_id": query_id,
            "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "known_target": target,
        }
    manifest = tmp_path / "known_2" / "batch_manifest.json"
    manifest.parent.mkdir()
    manifest.write_text(json.dumps({"queries": entries}), encoding="utf-8")

    progress = []
    report = run_manifest_batch_experiment(
        project,
        tmp_path / "batch_output",
        [manifest],
        workers=2,
        top_k=2,
        progress=progress.append,
    )
    assert report["query_count"] == 2
    assert report["workers"] == 2
    assert report["summary"]["combined"]["top_1"] == 2
    assert (tmp_path / "batch_output" / "dct_batch.json").exists()
    assert (tmp_path / "batch_output" / "dct_batch.csv").exists()
    assert (tmp_path / "batch_output" / "summary.md").exists()
    assert any("Building one shared DCT bank" in message for message in progress)
    assert any("[1/2]" in message for message in progress)
    assert progress[-1].startswith("Finished 2 queries")


@pytest.mark.parametrize(
    "config",
    [
        DCTConfig(samples=99),
        DCTConfig(samples=20, harmonics=21),
        DCTConfig(min_reference_coverage=0.8, max_reference_coverage=0.4),
        DCTConfig(coverage_steps=0),
    ],
)
def test_invalid_hyperparameters_are_rejected(config):
    with pytest.raises(ValueError):
        config.validate()
