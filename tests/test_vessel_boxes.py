import csv
import json
import numpy as np
import pandas as pd
from PIL import Image

from catalog.vessels import (
    apply_box_review,
    create_approved_crops,
    display_bbox_to_original,
    expanded_crop_bbox,
    migrate_legacy_review,
    read_vessel_boxes,
    reconcile_yolo_detections,
)
from processors import MaskExtractionConfig, ModelConfig
from processors.pipeline import MaskExtractor, ModelProcessor


def _detection(bbox, confidence=0.8, index=0):
    return {
        "bbox": bbox,
        "confidence": confidence,
        "mask_provenance": {"kind": "yolo_instance", "result_index": index},
    }


def test_nearby_detections_remain_separate_even_when_their_boxes_touch(tmp_path):
    document = reconcile_yolo_detections(
        tmp_path, "page", [400, 300],
        [_detection([20, 30, 100, 150], index=0),
         _detection([100, 30, 180, 150], index=1)],
    )
    assert len(document["detections"]) == 2
    assert len({item["detection_id"] for item in document["detections"]}) == 2
    assert [item["vessel_id"] for item in document["detections"]] == [
        "page_mask_layer_0", "page_mask_layer_1"
    ]


def test_model_processor_saves_each_box_confidence_and_instance_mask_separately(tmp_path):
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    Image.new("RGB", (100, 80), "white").save(images / "page.png")

    class TensorLike:
        def __init__(self, value): self.value = np.asarray(value)
        def cpu(self): return self
        def numpy(self): return self.value

    class Boxes:
        xyxy = TensorLike([[10, 10, 50, 60], [40, 10, 85, 60]])
        conf = TensorLike([0.91, 0.73])

    instance_masks = np.zeros((2, 80, 100), dtype=np.float32)
    instance_masks[0, 10:60, 10:50] = 1
    instance_masks[1, 10:60, 40:85] = 1

    class Masks:
        data = TensorLike(instance_masks)

    class Result:
        boxes = Boxes()
        masks = Masks()

    class Model:
        def predict(self, *_args, **_kwargs): return [Result()]

    processor = ModelProcessor(ModelConfig(models_dir=tmp_path, pred_output_dir=tmp_path))
    processor._process_single_image("page.png", images, Model(), 0.5, 1, 0, masks)
    document = read_vessel_boxes(masks, "page")
    assert [item["confidence"] for item in document["detections"]] == [0.91, 0.73]
    evidence = [masks / item["mask_provenance"]["file"] for item in document["detections"]]
    assert evidence[0] != evidence[1]
    assert all(path.exists() for path in evidence)


def test_overlapping_detections_keep_distinct_ids_across_reordered_model_output(tmp_path):
    first = reconcile_yolo_detections(
        tmp_path, "page", [500, 400],
        [_detection([40, 40, 240, 260], index=0),
         _detection([150, 80, 330, 280], index=1)],
    )
    ids_by_left = {item["detected_bbox"][0]: item["detection_id"] for item in first["detections"]}

    second = reconcile_yolo_detections(
        tmp_path, "page", [500, 400],
        [_detection([152, 81, 332, 281], index=0),
         _detection([41, 42, 241, 262], index=1)],
    )
    current = sorted(second["detections"], key=lambda item: item["detected_bbox"][0])
    assert current[0]["detection_id"] == ids_by_left[40]
    assert current[1]["detection_id"] == ids_by_left[150]


def test_review_edits_cannot_rebind_existing_detection_and_vessel_ids(tmp_path):
    original = reconcile_yolo_detections(
        tmp_path, "page", [300, 200],
        [_detection([10, 20, 80, 100]), _detection([120, 20, 190, 100], index=1)],
    )
    left, right = original["detections"]
    saved = apply_box_review(tmp_path, "page", [300, 200], [{
        "detection_id": left["detection_id"],
        "vessel_id": right["vessel_id"],  # stale/mismatched browser data
        "reviewed_bbox": [12, 22, 92, 112],
        "approved": True,
    }])
    by_id = {item["detection_id"]: item for item in saved["detections"]}
    assert by_id[left["detection_id"]]["vessel_id"] == left["vessel_id"]
    assert by_id[left["detection_id"]]["reviewed_bbox"] == [12, 22, 92, 112]
    assert by_id[left["detection_id"]]["approved"] is True
    # An omitted row is retained; omission is never interpreted as deletion.
    assert by_id[right["detection_id"]]["vessel_id"] == right["vessel_id"]


def test_manual_add_delete_and_approval_are_persisted(tmp_path):
    saved = apply_box_review(tmp_path, "page", [200, 150], [{
        "reviewed_bbox": [20, 25, 80, 100], "approved": True,
    }])
    item = saved["detections"][0]
    assert item["source"] == "manual"
    assert item["vessel_id"] == "page_mask_layer_0"
    assert item["status"] == "approved"

    deleted = apply_box_review(tmp_path, "page", [200, 150], [{
        "detection_id": item["detection_id"], "vessel_id": item["vessel_id"],
        "reviewed_bbox": item["reviewed_bbox"], "status": "deleted",
    }])
    assert deleted["detections"][0]["status"] == "deleted"
    assert deleted["detections"][0]["approved"] is False


def test_coordinate_translation_and_clipped_crop_margin():
    assert display_bbox_to_original([10, 20, 110, 120], [200, 150], [1000, 600]) == [
        50, 80, 550, 480
    ]
    assert expanded_crop_bbox([5, 8, 95, 88], [100, 90], 20) == [0, 0, 100, 90]


def test_approved_crop_is_unmasked_original_page_rectangle(tmp_path):
    page = np.zeros((80, 120, 3), dtype=np.uint8)
    page[:, :, 0] = np.arange(120, dtype=np.uint8)[None, :]
    page[:, :, 1] = np.arange(80, dtype=np.uint8)[:, None]
    image_path = tmp_path / "page.png"
    Image.fromarray(page).save(image_path)
    cards = tmp_path / "cards"
    document = {
        "image": "page", "image_size": [120, 80],
        "detections": [{
            "detection_id": "det-a", "vessel_id": "page_mask_layer_0",
            "reviewed_bbox": [20, 10, 60, 50], "approved": True,
            "status": "approved", "confidence": 0.9,
            "source": "yolo", "mask_provenance": {"file": "evidence.png"},
        }],
    }
    entries = create_approved_crops(image_path, cards, document, margin=5)
    crop = np.array(Image.open(cards / "page_mask_layer_0.png"))
    assert entries[0]["page_bbox"] == [20, 10, 60, 50]
    assert entries[0]["crop_bbox"] == [15, 5, 65, 55]
    assert crop.shape == (50, 50, 3)
    assert np.array_equal(crop, page[5:55, 15:65])


def test_project_extraction_uses_only_approved_boxes_and_keeps_reviewer_columns(tmp_path):
    project = tmp_path / "project"
    images, masks, cards = project / "images", project / "masks", project / "cards"
    images.mkdir(parents=True)
    masks.mkdir()
    cards.mkdir()
    Image.new("RGB", (160, 120), "white").save(images / "page.png")
    detected = reconcile_yolo_detections(
        masks, "page", [160, 120],
        [_detection([20, 20, 70, 90]), _detection([80, 20, 140, 90], index=1)],
    )
    left, right = detected["detections"]
    reviewed = apply_box_review(masks, "page", [160, 120], [
        {**left, "approved": True}, {**right, "approved": False},
    ])
    extractor = MaskExtractor(MaskExtractionConfig(
        pdfimg_output_dir=tmp_path, pred_output_dir=tmp_path, crop_margin_pixels=6))
    result = extractor.extract_masks_from_project(str(masks), str(cards))
    assert result == "Successfully extracted 1 approved vessel crops"
    assert (cards / f"{left['vessel_id']}.png").exists()
    assert not (cards / f"{right['vessel_id']}.png").exists()
    info = pd.read_csv(cards / "mask_info.csv", dtype=str, keep_default_na=False)
    assert info["mask_file"].tolist() == [left["vessel_id"]]

    info["Notes"] = ["researcher correction"]
    info.to_csv(cards / "mask_info.csv", index=False)
    moved = [{**item} for item in reviewed["detections"]]
    moved[0]["reviewed_bbox"] = [22, 22, 72, 92]
    apply_box_review(masks, "page", [160, 120], moved)
    extractor.extract_masks_from_project(str(masks), str(cards), crop_margin_pixels=8)
    refreshed = pd.read_csv(cards / "mask_info.csv", dtype=str, keep_default_na=False)
    assert refreshed.loc[0, "Notes"] == "researcher correction"
    with open(cards / "vessel_crops.json", encoding="utf-8") as handle:
        manifest = json.load(handle)
    assert manifest["crop_margin_pixels"] == 8
    assert manifest["vessels"][0]["page_bbox"] == [22, 22, 72, 92]

    current = read_vessel_boxes(masks, "page")["detections"]
    approve_both = [{**item, "approved": True, "status": "approved"} for item in current]
    apply_box_review(masks, "page", [160, 120], approve_both)
    extractor.extract_masks_from_project(str(masks), str(cards))
    assert (cards / f"{right['vessel_id']}.png").exists()
    delete_right = [{**item} for item in read_vessel_boxes(masks, "page")["detections"]]
    delete_right[1].update({"approved": False, "status": "deleted", "deleted": True})
    apply_box_review(masks, "page", [160, 120], delete_right)
    extractor.extract_masks_from_project(str(masks), str(cards))
    assert not (cards / f"{right['vessel_id']}.png").exists()


def test_legacy_review_migration_preserves_mask_file_identity_and_bbox(tmp_path):
    masks = tmp_path / "masks"
    cards = tmp_path / "cards"
    masks.mkdir()
    cards.mkdir()
    with open(cards / "mask_info_annots.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["bbox", "mask_file"])
        writer.writeheader()
        writer.writerow({"bbox": "(11, 22, 101, 122)",
                         "mask_file": "page_mask_layer_7.png"})
    document = migrate_legacy_review(masks, cards, "page", [500, 400])
    item = document["detections"][0]
    assert item["vessel_id"] == "page_mask_layer_7"
    assert item["reviewed_bbox"] == [11, 22, 101, 122]
    assert item["approved"] is True
    assert read_vessel_boxes(masks, "page")["detections"][0]["detection_id"] == item["detection_id"]
