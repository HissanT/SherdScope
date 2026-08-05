import base64
import io
import json
from pathlib import Path

import fitz
import numpy as np
import torch
from PIL import Image, ImageDraw

from training_curator.server import create_curator_app
from training_curator.unet import ProfileUNet, _anchor_prediction, deterministic_split
from training_curator.workspace import build_dpi_pilot, prepare_workspace


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _project(tmp_path: Path) -> tuple[Path, str]:
    project = tmp_path / "project"
    cards = project / "cards"
    accepted = cards / "profiles" / "accepted"
    pdf_source = project / "pdf_source"
    images = project / "images"
    for directory in (accepted, pdf_source, images):
        directory.mkdir(parents=True, exist_ok=True)
    _write_json(project / "project.json", {"project_id": "test"})

    source_pdf = pdf_source / "source.pdf"
    document = fitz.open()
    page = document.new_page(width=300, height=400)
    page.draw_rect(fitz.Rect(18, 20, 90, 72), color=(0, 0, 0), fill=(0, 0, 0))
    document.save(source_pdf)
    document.close()

    source_width = round(300 * 400 / 72)
    source_height = round(400 * 400 / 72)
    crop_bbox = [100, 100, 500, 400]
    crop_file = "page_0_mask_layer_0.png"
    card = Image.new("RGB", (400, 300), "white")
    draw = ImageDraw.Draw(card)
    draw.rectangle((0, 0, 300, 260), fill="black")
    card.save(cards / crop_file)
    Image.new("RGB", (source_width, source_height), "white").save(
        images / "page_0.jpg", quality=90
    )
    mask = Image.new("L", card.size, 0)
    ImageDraw.Draw(mask).rectangle((0, 0, 300, 260), fill=255)
    mask.save(accepted / "page_0_mask_layer_0_profile.png")

    _write_json(
        cards / "vessel_crops.json",
        {
            "schema_version": 1,
            "vessels": [
                {
                    "vessel_id": Path(crop_file).stem,
                    "crop_file": crop_file,
                    "image": "page_0",
                    "page_bbox": [124, 124, 476, 376],
                    "crop_bbox": crop_bbox,
                    "mask_provenance": {
                        "source_image_size": [source_width, source_height]
                    },
                }
            ],
        },
    )
    _write_json(
        cards / "profile_review.json",
        {
            "schema_version": 1,
            "profiles": {
                crop_file: {
                    "filename": crop_file,
                    "review_status": "approved",
                    "review_note": "checked",
                }
            },
        },
    )
    _write_json(
        project / "page_manifest.json",
        {
            "schema_version": 1,
            "pages": [
                {
                    "image_name": "page_0.jpg",
                    "source_pdf": "source.pdf",
                    "pdf_page_index": 0,
                    "render_dpi": 400,
                    "split_side": None,
                }
            ],
        },
    )
    return project, Path(crop_file).stem


def test_prepare_is_lossless_resumable_and_non_destructive(tmp_path):
    project, training_id = _project(tmp_path)
    card = project / "cards" / f"{training_id}.png"
    accepted = (
        project
        / "cards"
        / "profiles"
        / "accepted"
        / f"{training_id}_profile.png"
    )
    original_card = card.read_bytes()
    original_mask = accepted.read_bytes()

    first = prepare_workspace(project, dpi=600)
    second = prepare_workspace(project, dpi=600)

    assert first["rendered"] == 1
    assert first["reused"] == 0
    assert second["rendered"] == 0
    assert second["reused"] == 1
    assert card.read_bytes() == original_card
    assert accepted.read_bytes() == original_mask

    manifest = json.loads(first["manifest"].read_text(encoding="utf-8"))
    entry = manifest["entries"][training_id]
    assert entry["decision"] == "pending"
    assert entry["source_render_dpi"] == 400
    assert entry["training_render_dpi"] == 600
    candidate = first["workspace"] / entry["candidate_image"]
    draft = first["workspace"] / entry["draft_mask"]
    with Image.open(candidate) as image, Image.open(draft) as mask:
        assert image.format == "PNG"
        assert image.size == mask.size
        assert image.width > 400
        assert image.height > 300


def test_pilot_and_curator_approval(tmp_path):
    project, training_id = _project(tmp_path)
    pilot = build_dpi_pilot(project, dpi=600, count=1)
    assert pilot["report"].is_file()
    assert "Direct 600-DPI PNG crop" in pilot["report"].read_text(encoding="utf-8")

    prepared = prepare_workspace(project, dpi=600)
    app = create_curator_app(prepared["workspace"])
    client = app.test_client()
    listing = client.get("/api/entries")
    assert listing.status_code == 200
    item = listing.get_json()["entries"][0]
    assert item["training_id"] == training_id

    image_response = client.get(item["image_url"])
    assert image_response.status_code == 200
    with Image.open(io.BytesIO(image_response.data)) as image:
        mask = Image.new("L", image.size, 0)
        ImageDraw.Draw(mask).rectangle(
            (20, 20, image.width - 20, image.height - 20), fill=255
        )
        mask.putpixel((1, 1), 255)
    buffer = io.BytesIO()
    mask.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    approved = client.post(
        f"/api/entries/{training_id}/approve",
        json={"mask_data": f"data:image/png;base64,{encoded}", "review_note": "ready"},
    )
    assert approved.status_code == 200
    assert approved.get_json()["entry"]["decision"] == "approved"

    manifest = json.loads(prepared["manifest"].read_text(encoding="utf-8"))
    entry = manifest["entries"][training_id]
    assert entry["decision"] == "approved"
    assert entry["review_note"] == "ready"
    assert (prepared["workspace"] / entry["approved_image"]).is_file()
    assert (prepared["workspace"] / entry["approved_mask"]).is_file()
    with Image.open(prepared["workspace"] / entry["approved_mask"]) as saved_mask:
        assert saved_mask.getpixel((1, 1)) == 0
        assert saved_mask.getpixel((40, 40)) == 255
    assert entry["approval_cleanup"]["removed_pixels"] == 1

    rejected = client.post(
        f"/api/entries/{training_id}/reject",
        json={"review_note": "bad source"},
    )
    assert rejected.status_code == 200
    manifest = json.loads(prepared["manifest"].read_text(encoding="utf-8"))
    assert manifest["entries"][training_id]["decision"] == "rejected"
    assert (prepared["workspace"] / entry["approved_mask"]).is_file()


def test_recover_adds_high_resolution_profile_without_persisting(tmp_path):
    project, training_id = _project(tmp_path)
    prepared = prepare_workspace(project, dpi=600)
    manifest_before = prepared["manifest"].read_bytes()
    manifest = json.loads(manifest_before)
    entry = manifest["entries"][training_id]
    candidate_path = prepared["workspace"] / entry["candidate_image"]
    draft_path = prepared["workspace"] / entry["draft_mask"]
    with Image.open(candidate_path) as source:
        image = Image.new("RGB", source.size, "white")
    width, height = image.size
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        (width // 3, height // 3, width * 2 // 3, height * 4 // 5),
        fill="black",
    )
    draw.rectangle(
        (width * 5 // 12, height // 6, width * 7 // 12, height // 3),
        fill="black",
    )
    image.save(candidate_path)
    current = Image.new("L", image.size, 0)
    ImageDraw.Draw(current).rectangle(
        (width // 3, height // 3, width * 2 // 3, height * 4 // 5), fill=255
    )
    current.save(draft_path)
    buffer = io.BytesIO()
    current.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")

    client = create_curator_app(prepared["workspace"]).test_client()
    response = client.post(
        f"/api/entries/{training_id}/recover",
        json={"mask_data": f"data:image/png;base64,{encoded}"},
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["added_pixels"] > 0
    recovered_bytes = base64.b64decode(payload["mask_data"].split(",", 1)[1])
    with Image.open(io.BytesIO(recovered_bytes)) as recovered:
        assert recovered.getpixel((width // 2, height // 5)) == 255
    assert prepared["manifest"].read_bytes() == manifest_before


def test_unet_shape_and_split_are_stable():
    model = ProfileUNet(base_channels=4).eval()
    with torch.inference_mode():
        output = model(torch.zeros((2, 1, 64, 64)))
    assert output.shape == (2, 1, 64, 64)

    ids = [f"profile_{index}" for index in range(100)]
    first = deterministic_split(ids)
    second = deterministic_split(list(reversed(ids)))
    assert first == second
    assert set().union(*map(set, first.values())) == set(ids)
    assert all(first[name] for name in ("train", "validation", "test"))


def test_prediction_anchor_discards_unrelated_profile_blob():
    probability = np.zeros((80, 120), dtype=np.float32)
    probability[20:60, 10:35] = 0.9
    probability[20:60, 85:110] = 0.9
    anchor = Image.new("L", (120, 80), 0)
    ImageDraw.Draw(anchor).rectangle((88, 24, 106, 56), fill=255)
    selected = _anchor_prediction(probability, anchor, 0.55)
    assert not selected[30, 20]
    assert selected[30, 95]


def test_original_mask_remains_default_when_prediction_exists(tmp_path):
    project, training_id = _project(tmp_path)
    prepared = prepare_workspace(project, dpi=600)
    manifest = json.loads(prepared["manifest"].read_text(encoding="utf-8"))
    entry = manifest["entries"][training_id]
    draft_path = prepared["workspace"] / entry["draft_mask"]
    with Image.open(draft_path) as draft:
        prediction = Image.new("L", draft.size, 0)
    prediction_path = (
        prepared["workspace"] / "predictions" / "unet_v1" / "masks"
        / f"{training_id}.png"
    )
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    prediction.save(prediction_path)
    entry["model_prediction"] = prediction_path.relative_to(
        prepared["workspace"]
    ).as_posix()
    _write_json(prepared["manifest"], manifest)

    client = create_curator_app(prepared["workspace"]).test_client()
    listing = client.get("/api/entries").get_json()["entries"][0]
    with Image.open(io.BytesIO(client.get(listing["mask_url"]).data)) as original:
        assert np.asarray(original).sum() > 0
    with Image.open(io.BytesIO(client.get(listing["prediction_url"]).data)) as predicted:
        assert np.asarray(predicted).sum() == 0
