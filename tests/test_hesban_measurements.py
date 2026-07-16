from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from hesban_measurements import (
    detect_hesban_scale,
    detect_rim_diameter,
    manual_calibration,
    measure_figure,
    verified_measurement,
)
from utils import _assign_px_per_cm


def draw_scale(draw: ImageDraw.ImageDraw, x: int, y: int, segment: int = 50) -> None:
    for index in range(5):
        left, right = x + index * segment, x + (index + 1) * segment
        draw.rectangle((left, y, right, y + 14), outline=0, width=2)
        if index % 2 == 0:
            draw.rectangle((left + 2, y + 2, right - 2, y + 12), fill=0)
    draw.line((x, y - 5, x, y + 15), fill=0, width=2)
    draw.line((x + segment * 5, y - 5, x + segment * 5, y + 15), fill=0, width=2)


def make_page(path: Path, *, scale=True, second_scale=False, drawing=True,
              blur=False, scale_y=1450) -> None:
    image = Image.new("L", (1200, 1600), 255)
    draw = ImageDraw.Draw(image)
    if drawing:
        draw.line((200, 300, 800, 300), fill=0, width=3)
        draw.line((500, 300, 500, 600), fill=0, width=3)
        # A lower tail deliberately expands the card bbox without changing the rim.
        draw.line((500, 600, 850, 640), fill=0, width=3)
    if scale:
        draw_scale(draw, 120, scale_y)
    if second_scale:
        draw_scale(draw, 700, 1420)
    if blur:
        image = image.filter(ImageFilter.GaussianBlur(0.65))
    image.save(path)


def test_detects_segmented_ten_centimetre_scale_and_rejects_plain_lines(tmp_path):
    page = tmp_path / "page.png"
    make_page(page, blur=True)
    result = detect_hesban_scale(page)
    assert result["status"] == "verified_automatic"
    assert result["real_cm"] == 10.0
    assert 24 < result["px_per_cm"] < 27
    assert result["method"] == "automatic"

    unrelated = tmp_path / "unrelated.png"
    make_page(unrelated, scale=False)
    assert detect_hesban_scale(unrelated)["status"] == "unresolved"


def test_detects_scale_above_the_lower_page_area(tmp_path):
    page = tmp_path / "upper_scale.png"
    make_page(page, drawing=False, scale_y=360)
    result = detect_hesban_scale(page)
    assert result["status"] == "verified_automatic"
    assert result["evidence_bounds"][1] < 500
    assert 24 < result["px_per_cm"] < 27


def test_multiple_rulers_are_withheld(tmp_path):
    page = tmp_path / "multiple.png"
    make_page(page, second_scale=True)
    result = detect_hesban_scale(page)
    assert result["status"] == "unresolved"
    assert result["warning"] == "multiple_scale_candidates"

    # Rejected evidence may retain endpoints for the review overlay, but it is
    # not a calibration and cannot produce a numerical diameter.
    diameter = detect_rim_diameter(page, [150, 260, 900, 700], result)
    assert diameter["status"] == "unresolved"
    assert diameter["warning"] == "missing_scale_calibration"


def test_detects_mildly_skewed_structural_ruler(tmp_path):
    original = tmp_path / "original.png"
    skewed = tmp_path / "skewed.png"
    make_page(original)
    Image.open(original).rotate(2.0, fillcolor=255).save(skewed)
    result = detect_hesban_scale(skewed)
    assert result["status"] == "verified_automatic"
    assert 24 < result["px_per_cm"] < 27


def test_detects_faint_broken_and_noisy_structural_rulers(tmp_path):
    base = tmp_path / "base.png"
    make_page(base)
    source = np.asarray(Image.open(base)).copy()

    faint = np.where(source < 128, 145, source).astype(np.uint8)
    Image.fromarray(faint).save(tmp_path / "faint.png")

    broken = source.copy()
    broken[1438:1470, 169:172] = 255
    broken[1438:1470, 269:272] = 255
    Image.fromarray(broken).save(tmp_path / "broken.png")

    noisy = source.copy()
    generator = np.random.default_rng(41)
    ys = generator.integers(900, 1580, 500)
    xs = generator.integers(0, 1200, 500)
    noisy[ys, xs] = generator.integers(80, 210, 500)
    Image.fromarray(noisy).save(tmp_path / "noisy.png")

    for name in ("faint.png", "broken.png", "noisy.png"):
        result = detect_hesban_scale(tmp_path / name)
        assert result["status"] == "verified_automatic", name
        assert 24 < result["px_per_cm"] < 27


def test_independent_project_median_rejects_outlier_scale(tmp_path):
    project = tmp_path / "median"
    (project / "images").mkdir(parents=True)
    (project / "cards").mkdir()
    page = project / "images" / "page.png"
    make_page(page, scale=False)
    image = Image.open(page)
    draw_scale(ImageDraw.Draw(image), 120, 1450, segment=60)
    image.save(page)
    figure = {
        "drawing_pages": [{"image_name": page.name}],
        "drawings": [{"image_name": page.name, "mask_file": "card_0",
                      "bbox": [150, 260, 900, 700], "fingerprint": "bbox"}],
    }
    measure_figure(project, figure, project_ratios=[25, "bad ratio"])
    calibration = figure["scale_calibrations"][page.name]
    assert calibration["status"] == "unresolved"
    assert calibration["warning"] == "scale_median_disagreement"
    assert figure["drawings"][0]["measurement"]["status"] == "unresolved"


def test_diameter_uses_detected_rim_ink_not_card_bbox(tmp_path):
    page = tmp_path / "diameter.png"
    make_page(page)
    calibration = detect_hesban_scale(page)
    result = detect_rim_diameter(page, [150, 260, 900, 700], calibration)
    assert result["status"] == "verified_automatic"
    assert math.isclose(result["diameter_px"], 600, abs_tol=3)
    assert result["diameter_px"] != 750
    assert result["axis_diameter_px"] == result["diameter_px"]
    assert result["agreement"] < 0.01
    assert np.allclose(result["connected_radius_endpoints"], [[200, 300], [500, 300]], atol=3)
    assert 23.5 < result["suggested_cm"] < 24.0


def test_disconnected_right_profile_uses_mirrored_left_radius(tmp_path):
    page = tmp_path / "disconnected_profile.png"
    make_page(page, drawing=False)
    image = Image.open(page)
    draw = ImageDraw.Draw(image)
    # The reliable rim line begins at the left external edge, crosses the
    # centreline, then stops before the separate black profile.
    draw.line((200, 300, 780, 300), fill=0, width=3)
    draw.line((500, 300, 500, 600), fill=0, width=3)
    draw.polygon([(800, 300), (818, 315), (812, 350), (802, 390),
                  (792, 370), (794, 325)], fill=0)
    image.save(page)

    result = detect_rim_diameter(page, [150, 260, 900, 700], detect_hesban_scale(page))
    assert result["status"] == "verified_automatic"
    assert math.isclose(result["diameter_px"], 600, abs_tol=5)
    assert math.isclose(result["observed_span_px"], 580, abs_tol=5)
    assert np.allclose(result["connected_radius_endpoints"], [[200, 300], [500, 300]], atol=3)
    assert np.allclose(result["rim_endpoints"], [[200, 300], [800, 300]], atol=5)
    assert result["agreement"] < 0.05


def test_top_ten_percent_ignores_longer_lower_vessel_line(tmp_path):
    page = tmp_path / "top_band.png"
    make_page(page, drawing=False)
    image = Image.open(page)
    draw = ImageDraw.Draw(image)
    draw.line((200, 300, 800, 300), fill=0, width=3)
    draw.line((500, 300, 500, 600), fill=0, width=3)
    # This lower line is longer, but it lies below the top 10% of the card.
    draw.line((160, 340, 840, 340), fill=0, width=3)
    image.save(page)

    result = detect_rim_diameter(page, [150, 260, 900, 700], detect_hesban_scale(page))
    assert result["status"] == "verified_automatic"
    assert math.isclose(result["diameter_px"], 600, abs_tol=3)
    assert np.allclose(result["rim_endpoints"], [[200, 300], [800, 300]], atol=3)
    assert result["rim_search_region"][3] <= 304


def test_diameter_is_withheld_when_axis_radius_disagrees_with_rim_span(tmp_path):
    page = tmp_path / "off_axis.png"
    make_page(page, drawing=False)
    image = Image.open(page)
    draw = ImageDraw.Draw(image)
    draw.line((200, 300, 800, 300), fill=0, width=3)
    draw.line((430, 300, 430, 600), fill=0, width=3)
    image.save(page)
    result = detect_rim_diameter(page, [150, 260, 900, 700], detect_hesban_scale(page))
    assert result["status"] == "unresolved"
    assert result["agreement"] > 0.05
    assert result["warning"] == "diameter_estimators_disagree"


def test_manual_calibration_and_verified_endpoint_measurement(tmp_path):
    page = tmp_path / "manual.png"
    make_page(page, scale=False)
    calibration = manual_calibration(page, [100, 1400], [350, 1400])
    assert calibration["status"] == "verified_manual"
    assert calibration["px_per_cm"] == 25
    measurement = verified_measurement(
        {"suggested_cm": 23.0}, rim_endpoints=[[200, 300], [800, 300]],
        px_per_cm=calibration["px_per_cm"])
    assert measurement["status"] == "verified_manual"
    assert measurement["verified_cm"] == 24
    assert measurement["reviewer_history"][-1]["action"] == "diameter_verified"

    try:
        manual_calibration(page, [math.nan, 1400], [350, 1400])
    except ValueError as exc:
        assert "endpoints" in str(exc).lower()
    else:
        raise AssertionError("Non-finite scale coordinates must be rejected")


def test_split_page_uses_sibling_scale_and_persists_card_ratio(tmp_path):
    project = tmp_path / "project"
    (project / "images").mkdir(parents=True)
    (project / "cards").mkdir()
    (project / "masks").mkdir()
    left, right = project / "images" / "page_left.png", project / "images" / "page_right.png"
    make_page(left, scale=False)
    make_page(right, drawing=False)
    manifest = {"pages": [
        {"image_name": left.name, "source_pdf": "hesban.pdf", "pdf_page_index": 3,
         "split_part": "left", "render_dpi": 400},
        {"image_name": right.name, "source_pdf": "hesban.pdf", "pdf_page_index": 3,
         "split_part": "right", "render_dpi": 400},
    ]}
    (project / "page_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (project / "cards" / "mask_info.csv").write_text(
        "file,mask_file\npage_left,page_left_mask_layer_0\n", encoding="utf-8")
    figure = {
        "drawing_pages": [{"image_name": left.name}],
        "drawings": [{"image_name": left.name, "mask_file": "page_left_mask_layer_0",
                      "bbox": [150, 260, 900, 700], "fingerprint": "bbox-one"}],
    }
    measure_figure(project, figure)
    calibration = figure["scale_calibrations"][left.name]
    assert calibration["method"] == "sibling_page"
    assert calibration["evidence_image"] == right.name
    assert figure["drawings"][0]["measurement"]["suggested_cm"] > 0
    csv_text = (project / "cards" / "mask_info.csv").read_text(encoding="utf-8")
    assert "px_per_cm" in csv_text
    assert (project / "masks" / "page_left_scale.json").exists()


def test_legacy_manual_scale_sidecar_remains_verified(tmp_path):
    project = tmp_path / "legacy"
    (project / "images").mkdir(parents=True)
    (project / "cards").mkdir()
    (project / "masks").mkdir()
    page = project / "images" / "page.png"
    make_page(page, scale=False)
    (project / "masks" / "page_scale.json").write_text(json.dumps({
        "image": "page", "scales": [{
            "p1": [100, 1450], "p2": [350, 1450], "real_cm": 10, "zone": None,
        }],
    }), encoding="utf-8")
    figure = {
        "drawing_pages": [{"image_name": page.name}],
        "drawings": [{"image_name": page.name, "mask_file": "card_0",
                      "bbox": [150, 260, 900, 700], "fingerprint": "bbox"}],
    }
    measure_figure(project, figure)
    calibration = figure["scale_calibrations"][page.name]
    assert calibration["status"] == "verified_manual"
    assert calibration["method"] == "manual"
    assert calibration["px_per_cm"] == 25


def test_persistence_preserves_zoned_scales_and_clears_rejected_ratio(tmp_path):
    project = tmp_path / "preserve"
    (project / "images").mkdir(parents=True)
    (project / "cards").mkdir()
    (project / "masks").mkdir()
    page = project / "images" / "page.png"
    make_page(page, second_scale=True)
    sidecar = project / "masks" / "page_scale.json"
    zoned = {"p1": [20, 20], "p2": [120, 20], "real_cm": 5,
             "zone": [0, 0, 300, 300]}
    sidecar.write_text(json.dumps({"image": "page", "scales": [zoned]}),
                       encoding="utf-8")
    (project / "cards" / "mask_info.csv").write_text(
        "file,mask_file,px_per_cm\npage,card_0,99\n", encoding="utf-8")
    figure = {
        "drawing_pages": [{"image_name": page.name}],
        "drawings": [{"image_name": page.name, "mask_file": "card_0",
                      "bbox": [150, 260, 900, 700], "fingerprint": "bbox"}],
    }
    measure_figure(project, figure)
    calibration = figure["scale_calibrations"][page.name]
    assert calibration["status"] == "unresolved"
    assert figure["drawings"][0]["measurement"]["status"] == "unresolved"
    saved_scales = json.loads(sidecar.read_text(encoding="utf-8"))["scales"]
    assert zoned in saved_scales
    assert any(scale.get("warning") == "multiple_scale_candidates"
               for scale in saved_scales)
    csv_text = (project / "cards" / "mask_info.csv").read_text(encoding="utf-8")
    assert "99" not in csv_text
    assert _assign_px_per_cm([calibration], (200, 200)) is None
