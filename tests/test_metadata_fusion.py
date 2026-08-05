from catalog.metadata_fusion import (
    MetadataFusionConfig, compare_metadata, fuse_shape_results,
)


def test_missing_metadata_is_neutral():
    report = compare_metadata({"diameter": ""}, {"Rim Diameter (cm)": "20"})
    assert report["evidence"] == 0
    assert report["compared_fields"] == 0


def test_diameter_is_continuous_not_a_hard_cutoff():
    close = compare_metadata({"diameter": "10"}, {"Rim Diameter (cm)": "10.8"})
    medium = compare_metadata({"diameter": "10"}, {"Rim Diameter (cm)": "13"})
    far = compare_metadata({"diameter": "10"}, {"Rim Diameter (cm)": "20"})
    values = [
        next(field for field in report["fields"] if field["field"] == "rim_diameter_cm")["compatibility"]
        for report in (close, medium, far)
    ]
    assert values[0] > values[1] > values[2] > 0


def test_explicit_uncertainty_softens_a_difference():
    narrow = compare_metadata(
        {"diameter": {"value": 12, "uncertainty": .2}},
        {"diameter": {"value": 14, "uncertainty": .2}},
    )
    broad = compare_metadata(
        {"diameter": {"value": 12, "uncertainty": 2}},
        {"diameter": {"value": 14, "uncertainty": 2}},
    )
    get = lambda report: next(
        field for field in report["fields"] if field["field"] == "rim_diameter_cm"
    )
    assert get(broad)["compatibility"] > get(narrow)["compatibility"]


def test_diameter_small_difference_is_mild_but_large_difference_is_strong():
    close = compare_metadata(
        {"diameter": {"value": 20, "uncertainty": 1.5}},
        {"diameter": {"value": 22, "uncertainty": .75}},
    )
    far = compare_metadata(
        {"diameter": {"value": 10, "uncertainty": 1.5}},
        {"diameter": {"value": 20, "uncertainty": .75}},
    )
    get = lambda report: next(
        field for field in report["fields"] if field["field"] == "rim_diameter_cm"
    )
    assert get(close)["compatibility"] > .95
    assert get(far)["compatibility"] < .15
    assert "effective comparison tolerance" in get(close)["explanation"]


def test_five_centimetres_at_twenty_is_still_compatible():
    report = compare_metadata(
        {"diameter": {"value": 20, "uncertainty": 1.5}},
        {"diameter": {"value": 25, "uncertainty": .75}},
    )
    field = next(
        item for item in report["fields"] if item["field"] == "rim_diameter_cm"
    )
    assert field["compatibility"] >= .75


def test_severe_diameter_penalty_keeps_growing_past_comparator_saturation():
    rows = [
        {"reference_id": f"r{diameter}", "overall_score": .20}
        for diameter in (20, 30, 40, 50)
    ]
    references = {
        f"r{diameter}": {"Rim Diameter (cm)": str(diameter)}
        for diameter in (20, 30, 40, 50)
    }
    fused = fuse_shape_results(
        rows,
        {"rim_diameter_cm": {
            "value": "20", "source": "human", "reliability": .65,
            "uncertainty": 1.5,
        }},
        references,
    )
    by_id = {row["reference_id"]: row for row in fused}
    penalties = [by_id[f"r{diameter}"]["metadata_adjustment"] for diameter in (30, 40, 50)]
    assert penalties[0] < penalties[1] < penalties[2]
    assert penalties[1] - penalties[0] > .012
    assert penalties[2] - penalties[1] > .025
    assert by_id["r20"]["diameter_tail_penalty"] == 0


def test_neighbouring_munsell_is_better_than_distant_colour():
    near = compare_metadata({"fabric_core": "5YR 7/4"}, {"fabric_core": "7.5YR 7/4"})
    far = compare_metadata({"fabric_core": "5YR 7/4"}, {"fabric_core": "5B 3/1"})
    get = lambda report: next(x for x in report["fields"] if x["field"] == "fabric_core")
    assert get(near)["compatibility"] > get(far)["compatibility"]


def test_archaeological_colour_names_preserve_related_regions():
    adjacent = compare_metadata(
        {"fabric_core": "5YR 7/4\nPink"},
        {"fabric_core": "2.5YR 6/4\nLight reddish brown"},
    )
    distant = compare_metadata(
        {"fabric_core": "5YR 7/4\nPink"},
        {"fabric_core": "10YR 8/2\nWhite"},
    )
    get = lambda report: next(x for x in report["fields"] if x["field"] == "fabric_core")
    assert get(adjacent)["compatibility"] > get(distant)["compatibility"]
    assert "Munsell H/V/C" in get(adjacent)["explanation"]


def test_chip_only_input_gets_interpretable_descriptors():
    report = compare_metadata({"fabric_core": "2.5YR 6/4"}, {"fabric_core": "5YR 7/2"})
    field = next(x for x in report["fields"] if x["field"] == "fabric_core")
    assert "2.5YR, middle, moderate" in field["explanation"]
    assert "5YR, light, muted" in field["explanation"]


def test_lightness_modifier_matters_inside_same_colour_family():
    close = compare_metadata(
        {"fabric_core": "Light reddish brown"}, {"fabric_core": "Reddish brown"}
    )
    far = compare_metadata(
        {"fabric_core": "Light reddish brown"}, {"fabric_core": "Dark reddish brown"}
    )
    get = lambda report: next(x for x in report["fields"] if x["field"] == "fabric_core")
    assert get(close)["compatibility"] > get(far)["compatibility"]


def test_hesban_ordinal_codes_allow_nearby_categories():
    near = compare_metadata({"nonplastics_size": "4"}, {"nonplastics_size": "5"})
    far = compare_metadata({"nonplastics_size": "1"}, {"nonplastics_size": "7"})
    get = lambda report: next(x for x in report["fields"] if x["field"] == "nonplastics_size")
    assert get(near)["compatibility"] > get(far)["compatibility"]


def test_multiple_codes_get_partial_credit():
    partial = compare_metadata({"decor": "GR+R"}, {"decor": "GR+N"})
    mismatch = compare_metadata({"decor": "GR+R"}, {"decor": "PA+BA"})
    get = lambda report: next(x for x in report["fields"] if x["field"] == "decor")
    assert get(partial)["compatibility"] > get(mismatch)["compatibility"]


def test_fusion_uses_a_smaller_continuous_metadata_score():
    rows = [{"reference_id": "a", "score": .10}, {"reference_id": "b", "score": .11}]
    missing = fuse_shape_results(rows, {"diameter": "12"}, {})
    assert [x["reference_id"] for x in missing] == ["a", "b"]
    assert all(x["metadata_weight"] == 0 for x in missing)
    assert all(x["metadata_score"] is None for x in missing)
    fused = fuse_shape_results(rows, {"diameter": "12", "fire": "O"}, {
        "a": {"Rim Diameter (cm)": "24", "Fire": "R"},
        "b": {"Rim Diameter (cm)": "12.5", "Fire": "O"},
    })
    assert len(fused) == 2
    assert fused[0]["reference_id"] == "b"
    assert all("shape_score" in x and "metadata" in x for x in fused)
    assert all(0 < x["metadata_weight"] < .23 for x in fused)


def test_metadata_bonus_and_penalty_are_bounded_and_asymmetric():
    rows = [{"reference_id": "same", "score": .25}, {"reference_id": "off", "score": .25}]
    fused = fuse_shape_results(rows, {"diameter": "10", "fabric_core": "5YR 7/4\nPink"}, {
        "same": {"Rim Diameter (cm)": "10", "Fabric Color - Core": "5YR 7/4\nPink"},
        "off": {"Rim Diameter (cm)": "30", "Fabric Color - Core": "5YR 3/1\nDark gray"},
    })
    same = next(row for row in fused if row["reference_id"] == "same")
    off = next(row for row in fused if row["reference_id"] == "off")
    assert -.030 <= same["metadata_adjustment"] < 0
    assert 0 < off["metadata_adjustment"] <= .075
    assert abs(off["metadata_adjustment"]) > abs(same["metadata_adjustment"])


def test_fusion_summary_uses_the_effective_final_score_change():
    fused = fuse_shape_results(
        [
            {"reference_id": "same", "overall_score": 0.40},
            {"reference_id": "off", "overall_score": 0.40},
        ],
        {"rim_diameter_cm": 20},
        {
            "same": {"rim_diameter_cm": 20},
            "off": {"rim_diameter_cm": 50},
        },
    )
    for row in fused:
        delta = round(row["fused_score"] - row["shape_score"], 6)
        summary = row["metadata"]["summary"]
        assert f"{abs(delta):.6f}" in summary
        assert f"{row['shape_score']:.6f}" in summary
        assert f"{row['fused_score']:.6f}" in summary
        assert ("lowered" in summary) == (delta < 0)
        assert ("raised" in summary) == (delta > 0)


def test_fusion_summary_reports_no_change_when_clipping_absorbs_bonus():
    row = fuse_shape_results(
        [{"reference_id": "same", "overall_score": 0.0}],
        {"rim_diameter_cm": 20},
        {"same": {"rim_diameter_cm": 20}},
    )[0]
    assert row["metadata_adjustment"] < 0
    assert row["fused_score"] == row["shape_score"] == 0.0
    assert "did not change the final cost" in row["metadata"]["summary"]


def test_vessel_type_contributes_without_becoming_a_hard_rule():
    rows = [
        {"reference_id": "crater", "overall_score": .10},
        {"reference_id": "bowl", "overall_score": .12},
    ]
    fused = fuse_shape_results(rows, {"vessel_type": "Bowl"}, {
        "crater": {"Type": "Crater"},
        "bowl": {"Type": "Bowl"},
    })
    assert fused[0]["reference_id"] == "bowl"
    assert len(fused) == 2
    assert fused[1]["metadata_score"] > fused[0]["metadata_score"]


def test_fusion_reads_native_matcher_overall_score():
    rows = [
        {"reference_id": "a", "overall_score": .14},
        {"reference_id": "b", "overall_score": .21},
    ]
    fused = fuse_shape_results(rows, {}, {})
    assert [row["reference_id"] for row in fused] == ["a", "b"]
    assert [row["shape_score"] for row in fused] == [.14, .21]


def test_correlated_colour_fields_are_group_capped():
    config = MetadataFusionConfig(max_group_evidence=.5)
    report = compare_metadata(
        {"fabric_exterior": "5YR 7/4", "fabric_core": "5YR 7/4", "fabric_interior": "5YR 7/4"},
        {"fabric_exterior": "5YR 7/4", "fabric_core": "5YR 7/4", "fabric_interior": "5YR 7/4"},
        config,
    )
    assert report["group_evidence"]["colour"] <= .5
