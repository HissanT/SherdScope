import json

import numpy as np
from PIL import Image, ImageDraw

from catalog.profile_segmentation import (
    generate_profile_proposals,
    profile_mask_path,
    propose_profile_mask,
    read_profile_review,
    save_profile_mask,
    write_profile_review,
)


def _crop_with_profile(*, connected_line=False, second_blob=False):
    image = Image.new("RGB", (180, 120), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([132, 18, 155, 105], fill="black")
    draw.ellipse([120, 16, 166, 42], fill="black")
    draw.ellipse([122, 78, 166, 110], fill="black")
    if connected_line:
        draw.line([16, 61, 132, 61], fill="black", width=2)
    else:
        draw.line([16, 61, 118, 61], fill="black", width=2)
    if second_blob:
        draw.rectangle([30, 18, 53, 105], fill="black")
        draw.ellipse([19, 16, 64, 42], fill="black")
        draw.ellipse([19, 78, 64, 110], fill="black")
    return image


def test_profile_proposal_prefers_thick_blob_over_connected_thin_line():
    mask, proposal = propose_profile_mask(_crop_with_profile(connected_line=True))
    assert proposal.confidence > 0.5
    assert proposal.bbox is not None
    x1, _y1, x2, _y2 = proposal.bbox
    assert x1 > 90
    assert x2 > 150
    assert np.asarray(mask)[:, :70].sum() == 0


def test_profile_proposal_removes_short_thin_stub_touching_profile():
    image = _crop_with_profile(connected_line=False)
    draw = ImageDraw.Draw(image)
    draw.line([92, 61, 132, 61], fill="black", width=2)

    mask, _proposal = propose_profile_mask(image)
    selected = np.asarray(mask) > 0

    assert not selected[61, 100]
    assert not selected[61, 116]
    assert selected[:, 132:166].sum() > 0


def test_profile_proposal_restores_light_outline_around_selected_blob():
    image = Image.new("RGB", (180, 120), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse([118, 15, 168, 112], fill=(145, 145, 145))
    draw.ellipse([128, 25, 158, 102], fill="black")
    draw.line([15, 58, 118, 58], fill=(145, 145, 145), width=2)

    mask, proposal = propose_profile_mask(image)
    selected = np.asarray(mask) > 0

    assert proposal.bbox is not None
    assert proposal.bbox[0] <= 120
    assert proposal.bbox[2] >= 166
    assert selected[60, 121]
    assert selected[60, 166]
    assert not selected[58, 30]


def test_profile_proposal_reports_ambiguous_multiple_thick_candidates():
    _mask, proposal = propose_profile_mask(_crop_with_profile(second_blob=True))
    assert proposal.candidate_count >= 2
    assert "several_similar_candidates" in proposal.reasons
    assert proposal.confidence <= 0.72


def test_generate_profile_proposals_writes_sidecar_and_masks(tmp_path):
    cards = tmp_path / "cards"
    cards.mkdir()
    _crop_with_profile(connected_line=True).save(cards / "page_mask_layer_0.png")

    summary = generate_profile_proposals(cards)
    assert summary["generated"] == 1
    assert profile_mask_path(cards, "page_mask_layer_0.png", "auto").exists()
    assert profile_mask_path(cards, "page_mask_layer_0.png", "accepted").exists()
    document = read_profile_review(cards)
    record = document["profiles"]["page_mask_layer_0.png"]
    assert record["review_status"] == "pending"
    assert record["proposal"]["algorithm_version"] == "thick-profile-v3"


def test_reviewed_profile_is_not_overwritten_without_force(tmp_path):
    cards = tmp_path / "cards"
    cards.mkdir()
    _crop_with_profile().save(cards / "page_mask_layer_0.png")
    generate_profile_proposals(cards)

    accepted = np.zeros((120, 180), dtype=np.uint8)
    accepted[20:40, 20:40] = 255
    save_profile_mask(profile_mask_path(cards, "page_mask_layer_0.png", "accepted"), accepted)
    document = read_profile_review(cards)
    document["profiles"]["page_mask_layer_0.png"]["review_status"] = "edited"
    document["profiles"]["page_mask_layer_0.png"]["review_note"] = "manual correction"
    write_profile_review(cards, document)

    summary = generate_profile_proposals(cards)
    assert summary["skipped_reviewed"] == 1
    current = np.asarray(Image.open(profile_mask_path(cards, "page_mask_layer_0.png", "accepted")))
    assert current[25, 25] == 255
    assert current[:, 130:160].sum() == 0
    with open(cards / "profile_review.json", encoding="utf-8") as handle:
        saved = json.load(handle)
    assert saved["profiles"]["page_mask_layer_0.png"]["review_note"] == "manual correction"


def test_force_profile_generation_resets_review_and_replaces_accepted_mask(tmp_path):
    cards = tmp_path / "cards"
    cards.mkdir()
    _crop_with_profile().save(cards / "page_mask_layer_0.png")
    generate_profile_proposals(cards)

    accepted = np.zeros((120, 180), dtype=np.uint8)
    accepted[20:40, 20:40] = 255
    save_profile_mask(profile_mask_path(cards, "page_mask_layer_0.png", "accepted"), accepted)
    document = read_profile_review(cards)
    document["profiles"]["page_mask_layer_0.png"]["review_status"] = "approved"
    document["profiles"]["page_mask_layer_0.png"]["review_note"] = "looks good"
    write_profile_review(cards, document)

    summary = generate_profile_proposals(cards, force=True)
    assert summary["generated"] == 1
    assert summary["skipped_reviewed"] == 0
    refreshed = read_profile_review(cards)["profiles"]["page_mask_layer_0.png"]
    assert refreshed["review_status"] == "pending"
    assert refreshed["review_note"] == ""
    current = np.asarray(Image.open(profile_mask_path(cards, "page_mask_layer_0.png", "accepted")))
    assert current[25, 25] == 0
    assert current[:, 130:160].sum() > 0
