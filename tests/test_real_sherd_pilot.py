import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from real_sherd_pilot.pipeline import (
    automatic_roi,
    segmentation_metrics,
    smooth_mask,
    save_gold,
)
from real_sherd_pilot.server import create_app


def test_roi_and_metrics_are_stable():
    full = np.zeros((640, 640), dtype=bool)
    full[200:400, 250:390] = True
    roi = automatic_roi(full, (1280, 1920))
    assert roi[0] < 500 and roi[2] > 780
    assert roi[1] < 600 and roi[3] > 1200

    truth = np.zeros((640, 640), dtype=bool)
    truth[100:500, 200:430] = True
    perfect = segmentation_metrics(truth, truth)
    assert perfect["dice"] == 1.0
    assert perfect["iou"] == 1.0
    shifted = np.roll(truth, 8, axis=1)
    metrics = segmentation_metrics(shifted, truth)
    assert 0 < metrics["boundary_f1_at_3px"] < 1
    assert metrics["hausdorff95_px"] > 0


def test_smoothing_preserves_large_shape_and_removes_single_pixel():
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:80, 30:70] = True
    mask[2, 2] = True
    smoothed = smooth_mask(mask, 1.0)
    assert smoothed[40, 40]
    assert not smoothed[2, 2]
    assert abs(int(smoothed.sum()) - 2400) < 50


def _workspace(tmp_path: Path) -> Path:
    root = tmp_path / "pilot"
    crop = Image.new("RGB", (160, 120), "white")
    crop_path = root / "crops" / "photo.png"
    crop_path.parent.mkdir(parents=True)
    crop.save(crop_path)
    unet = Image.new("L", crop.size, 0)
    ImageDraw.Draw(unet).rectangle((42, 22, 122, 102), fill=255)
    for folder in ("raw_masks", "smooth_masks"):
        target = root / "unet" / folder / "photo.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        unet.save(target)
    original = root / "originals" / "photo.jpg"
    original.parent.mkdir(parents=True)
    crop.save(original)
    manifest = {
        "schema_version": 1, "device": "cpu", "threshold": .9,
        "entries": {"photo": {
            "id": "photo", "source_filename": "photo.jpg", "status": "pending",
            "original": "originals/photo.jpg", "original_size": [160, 120],
            "crop": "crops/photo.png", "crop_size": [160, 120], "roi": [0, 0, 160, 120],
            "unet_raw": "unet/raw_masks/photo.png", "unet_smooth": "unet/smooth_masks/photo.png",
        }},
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_gold_save_writes_metrics_overlay_and_query_exports(tmp_path):
    root = _workspace(tmp_path)
    mask = Image.new("L", (160, 120), 0)
    ImageDraw.Draw(mask).rectangle((40, 20, 120, 100), fill=255)
    import base64, io
    buffer = io.BytesIO(); mask.save(buffer, format="PNG")
    data = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    entry = save_gold(root, "photo", data, smoothing_sigma=1.0,
                      outline=[[40, 20], [120, 20], [120, 100], [40, 100]])
    assert entry["status"] == "completed"
    assert entry["metrics"]["unet_smoothed"]["dice"] > .95
    assert (root / entry["overlay"]).is_file()
    assert (root / entry["manual_query"]).is_file()
    assert (root / "reports" / "segmentation_metrics.csv").is_file()

    client = create_app(root).test_client()
    listing = client.get("/api/entries")
    assert listing.status_code == 200
    assert listing.get_json()["completed"] == 1

