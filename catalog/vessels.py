"""Stable per-vessel detection, review, and crop persistence.

YOLO instances are stored independently.  The page mask remains optional visual
evidence; it is deliberately not used to recover instances for new detections.
"""

from __future__ import annotations

import csv
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


BOX_SIDECAR_SUFFIX = "_vessel_boxes.json"
BOX_SCHEMA_VERSION = 2
CROP_MANIFEST = "vessel_crops.json"
_ID_NAMESPACE = uuid.UUID("650bd64b-d8f4-4f4b-b5a7-84563fd6e960")


class VesselBoxError(ValueError):
    """Raised when reviewed vessel-box data is unsafe or inconsistent."""


def normalize_bbox(bbox: Iterable[Any], image_size: Iterable[Any] | None = None) -> list[int]:
    """Return a clipped ``[x1, y1, x2, y2]`` box with positive area."""
    try:
        values = [int(round(float(value))) for value in bbox]
    except (TypeError, ValueError) as exc:
        raise VesselBoxError("A vessel box must contain four numeric coordinates") from exc
    if len(values) != 4:
        raise VesselBoxError("A vessel box must contain exactly four coordinates")
    x1, y1, x2, y2 = values
    if image_size is not None:
        size = [int(value) for value in image_size]
        if len(size) != 2 or min(size) <= 0:
            raise VesselBoxError("The source image size is invalid")
        width, height = size
        x1, x2 = max(0, min(width, x1)), max(0, min(width, x2))
        y1, y2 = max(0, min(height, y1)), max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        raise VesselBoxError("A vessel box must have positive width and height")
    return [x1, y1, x2, y2]


def display_bbox_to_original(bbox: Iterable[Any], display_size: Iterable[Any],
                             original_size: Iterable[Any]) -> list[int]:
    """Translate a browser display-space box to original page coordinates."""
    display_width, display_height = [float(value) for value in display_size]
    original_width, original_height = [int(value) for value in original_size]
    if display_width <= 0 or display_height <= 0:
        raise VesselBoxError("The display size must be positive")
    x1, y1, x2, y2 = [float(value) for value in bbox]
    return normalize_bbox([
        x1 * original_width / display_width,
        y1 * original_height / display_height,
        x2 * original_width / display_width,
        y2 * original_height / display_height,
    ], [original_width, original_height])


def expanded_crop_bbox(bbox: Iterable[Any], image_size: Iterable[Any], margin: int) -> list[int]:
    """Expand a reviewed box by a clipped, non-negative pixel margin."""
    width, height = [int(value) for value in image_size]
    x1, y1, x2, y2 = normalize_bbox(bbox, [width, height])
    margin = max(0, int(margin))
    return [max(0, x1 - margin), max(0, y1 - margin),
            min(width, x2 + margin), min(height, y2 + margin)]


def bbox_iou(left: Iterable[Any], right: Iterable[Any]) -> float:
    ax1, ay1, ax2, ay2 = normalize_bbox(left)
    bx1, by1, bx2, by2 = normalize_bbox(right)
    intersection = max(0, min(ax2, bx2) - max(ax1, bx1)) * \
        max(0, min(ay2, by2) - max(ay1, by1))
    if not intersection:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / union if union else 0.0


def _sidecar_path(masks_dir: Path, base_filename: str) -> Path:
    return Path(masks_dir) / f"{base_filename}{BOX_SIDECAR_SUFFIX}"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _stable_detection_id(base_filename: str, vessel_id: str) -> str:
    return str(uuid.uuid5(_ID_NAMESPACE, f"{base_filename}:{vessel_id}"))


def _record(base_filename: str, vessel_id: str, bbox: Iterable[Any], *,
            source: str, confidence: float | None = None,
            approved: bool = False, mask_provenance: dict[str, Any] | None = None,
            detection_id: str | None = None) -> dict[str, Any]:
    box = normalize_bbox(bbox)
    return {
        "detection_id": detection_id or str(uuid.uuid4()),
        "vessel_id": vessel_id,
        "source": source,
        "detected_bbox": list(box),
        "reviewed_bbox": list(box),
        "confidence": None if confidence is None else round(float(confidence), 6),
        "approved": bool(approved),
        "status": "approved" if approved else "pending",
        "mask_provenance": mask_provenance,
        "stale_detection": False,
    }


def _validate_document(document: dict[str, Any], base_filename: str | None = None) -> dict[str, Any]:
    image_size = document.get("image_size") or []
    if len(image_size) != 2 or min(int(value) for value in image_size) <= 0:
        raise VesselBoxError("The vessel-box sidecar has no valid source image size")
    image_size = [int(image_size[0]), int(image_size[1])]
    seen_detection_ids: set[str] = set()
    seen_vessel_ids: set[str] = set()
    records = []
    for raw in document.get("detections", []):
        item = dict(raw)
        detection_id = str(item.get("detection_id") or "").strip()
        vessel_id = str(item.get("vessel_id") or "").strip()
        if not detection_id or not vessel_id:
            raise VesselBoxError("Every vessel needs both an immutable detection ID and vessel ID")
        if detection_id in seen_detection_ids or vessel_id in seen_vessel_ids:
            raise VesselBoxError("Duplicate vessel identities were found in the vessel-box sidecar")
        seen_detection_ids.add(detection_id)
        seen_vessel_ids.add(vessel_id)
        item["detected_bbox"] = normalize_bbox(
            item.get("detected_bbox") or item.get("reviewed_bbox"), image_size)
        item["reviewed_bbox"] = normalize_bbox(
            item.get("reviewed_bbox") or item["detected_bbox"], image_size)
        item["approved"] = bool(item.get("approved", False))
        if item.get("status") not in {"pending", "approved", "deleted"}:
            item["status"] = "approved" if item["approved"] else "pending"
        if item["status"] == "deleted":
            item["approved"] = False
        records.append(item)
    return {
        "schema_version": BOX_SCHEMA_VERSION,
        "image": base_filename or str(document.get("image") or ""),
        "image_size": image_size,
        "detections": records,
    }


def read_vessel_boxes(masks_dir: Path, base_filename: str) -> dict[str, Any] | None:
    path = _sidecar_path(Path(masks_dir), base_filename)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as handle:
        return _validate_document(json.load(handle), base_filename)


def write_vessel_boxes(masks_dir: Path, base_filename: str,
                       document: dict[str, Any]) -> Path:
    clean = _validate_document(document, base_filename)
    path = _sidecar_path(Path(masks_dir), base_filename)
    _atomic_json(path, clean)
    return path


def _next_vessel_id(base_filename: str, records: Iterable[dict[str, Any]]) -> str:
    pattern = re.compile(rf"^{re.escape(base_filename)}_mask_layer_(\d+)$")
    used = []
    for item in records:
        match = pattern.match(str(item.get("vessel_id", "")))
        if match:
            used.append(int(match.group(1)))
    return f"{base_filename}_mask_layer_{max(used, default=-1) + 1}"


def reconcile_yolo_detections(masks_dir: Path, base_filename: str,
                              image_size: Iterable[Any], detections: list[dict[str, Any]],
                              match_iou: float = 0.5) -> dict[str, Any]:
    """Persist independent model detections while retaining reviewed identities."""
    size = [int(value) for value in image_size]
    old_document = read_vessel_boxes(masks_dir, base_filename)
    old = old_document["detections"] if old_document else []
    available = [
        item for item in old
        if item.get("source") not in {"manual", "legacy_polygon"}
    ]
    matched_ids: set[str] = set()
    updated: list[dict[str, Any]] = []

    for detection in detections:
        detected_bbox = normalize_bbox(detection["bbox"], size)
        candidates = [item for item in available if item["detection_id"] not in matched_ids]
        match = max(candidates, key=lambda item: bbox_iou(item["detected_bbox"], detected_bbox),
                    default=None)
        if match is not None and bbox_iou(match["detected_bbox"], detected_bbox) >= match_iou:
            item = dict(match)
            matched_ids.add(item["detection_id"])
            item["detected_bbox"] = detected_bbox
            if not item.get("approved") and item.get("status") != "deleted":
                item["reviewed_bbox"] = list(detected_bbox)
            item["confidence"] = round(float(detection.get("confidence", 0.0)), 6)
            item["mask_provenance"] = detection.get("mask_provenance")
            item["stale_detection"] = False
        else:
            vessel_id = _next_vessel_id(base_filename, [*old, *updated])
            item = _record(base_filename, vessel_id, detected_bbox, source="yolo",
                           confidence=detection.get("confidence"),
                           mask_provenance=detection.get("mask_provenance"))
        updated.append(item)

    # Never silently discard a researcher's reviewed, deleted, or manual record.
    for item in old:
        if item.get("detection_id") in matched_ids:
            continue
        retained = dict(item)
        if retained.get("source") == "yolo":
            retained["stale_detection"] = True
        updated.append(retained)

    document = {"schema_version": BOX_SCHEMA_VERSION, "image": base_filename,
                "image_size": size, "detections": updated}
    write_vessel_boxes(masks_dir, base_filename, document)
    return _validate_document(document, base_filename)


def migrate_legacy_review(masks_dir: Path, cards_dir: Path, base_filename: str,
                          image_size: Iterable[Any], polygons: list | None = None,
                          fallback_boxes: list[Iterable[Any]] | None = None) -> dict[str, Any]:
    """Create box records once from legacy reviewed cards/polygons/page masks."""
    existing = read_vessel_boxes(masks_dir, base_filename)
    if existing is not None:
        return existing
    size = [int(value) for value in image_size]
    records: list[dict[str, Any]] = []
    annots_path = Path(cards_dir) / "mask_info_annots.csv"
    if annots_path.exists():
        with open(annots_path, newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                vessel_id = Path(str(row.get("mask_file", ""))).stem
                if not vessel_id.startswith(f"{base_filename}_mask_layer_"):
                    continue
                numbers = re.findall(r"-?\d+(?:\.\d+)?", str(row.get("bbox", "")))
                if len(numbers) != 4:
                    continue
                records.append(_record(
                    base_filename, vessel_id, numbers, source="legacy_review",
                    approved=True, detection_id=_stable_detection_id(base_filename, vessel_id),
                    mask_provenance={"kind": "legacy_page_mask"},
                ))
    for polygon in polygons or []:
        if not isinstance(polygon, list) or len(polygon) < 3:
            continue
        xs = [point[0] for point in polygon]
        ys = [point[1] for point in polygon]
        box = normalize_bbox([min(xs), min(ys), max(xs), max(ys)], size)
        if any(bbox_iou(item["reviewed_bbox"], box) >= 0.95 for item in records):
            continue
        vessel_id = _next_vessel_id(base_filename, records)
        records.append(_record(
            base_filename, vessel_id, box, source="legacy_polygon", approved=True,
            detection_id=_stable_detection_id(base_filename, vessel_id),
            mask_provenance={"kind": "reviewer_polygon", "polygon": polygon},
        ))
    if not records:
        for box in fallback_boxes or []:
            vessel_id = _next_vessel_id(base_filename, records)
            records.append(_record(
                base_filename, vessel_id, box, source="legacy_mask_migration", approved=True,
                detection_id=_stable_detection_id(base_filename, vessel_id),
                mask_provenance={"kind": "legacy_connected_component"},
            ))
    document = {"schema_version": BOX_SCHEMA_VERSION, "image": base_filename,
                "image_size": size, "detections": records}
    write_vessel_boxes(masks_dir, base_filename, document)
    return _validate_document(document, base_filename)


def legacy_mask_component_boxes(mask_path: Path, image_size: Iterable[Any],
                                min_area_ratio: float = 0.0002) -> list[list[int]]:
    """One-time compatibility migration for masks created before schema v2.

    New YOLO output never passes through this function.
    """
    import cv2
    import numpy as np

    image = np.array(Image.open(mask_path))
    alpha = image[:, :, 3] if image.ndim == 3 and image.shape[2] >= 4 else image
    binary = (alpha > 0).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    target_width, target_height = [int(value) for value in image_size]
    scale_x, scale_y = target_width / binary.shape[1], target_height / binary.shape[0]
    minimum_area = binary.shape[0] * binary.shape[1] * float(min_area_ratio)
    boxes = []
    for label_index in range(1, count):
        x, y, width, height, area = stats[label_index]
        if area < minimum_area or width <= 0 or height <= 0:
            continue
        boxes.append(normalize_bbox([
            x * scale_x, y * scale_y, (x + width) * scale_x, (y + height) * scale_y,
        ], [target_width, target_height]))
    return boxes


def apply_box_review(masks_dir: Path, base_filename: str, image_size: Iterable[Any],
                     submitted: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply UI edits without allowing identity reassignment or silent omission."""
    size = [int(value) for value in image_size]
    existing_document = read_vessel_boxes(masks_dir, base_filename)
    existing = existing_document["detections"] if existing_document else []
    by_detection = {item["detection_id"]: item for item in existing}
    by_vessel = {item["vessel_id"]: item for item in existing}
    updated: list[dict[str, Any]] = []
    touched: set[str] = set()

    for raw in submitted:
        detection_id = str(raw.get("detection_id") or "").strip()
        vessel_id = str(raw.get("vessel_id") or "").strip()
        prior = by_detection.get(detection_id) if detection_id else None
        if prior is None and vessel_id:
            prior = by_vessel.get(vessel_id)
        if prior is not None:
            # The persisted pair is authoritative; a browser cannot rebind IDs.
            item = dict(prior)
            detection_id = item["detection_id"]
            touched.add(detection_id)
        else:
            vessel_id = _next_vessel_id(base_filename, [*existing, *updated])
            item = _record(base_filename, vessel_id, raw.get("reviewed_bbox") or raw.get("bbox"),
                           source="manual")
            detection_id = item["detection_id"]
            touched.add(detection_id)
        item["reviewed_bbox"] = normalize_bbox(
            raw.get("reviewed_bbox") or raw.get("bbox") or item["reviewed_bbox"], size)
        deleted = bool(raw.get("deleted")) or raw.get("status") == "deleted"
        item["approved"] = bool(raw.get("approved")) and not deleted
        item["status"] = "deleted" if deleted else ("approved" if item["approved"] else "pending")
        item["review_source"] = "researcher"
        updated.append(item)

    # Missing browser rows are retained. Deletion must always be explicit.
    updated.extend(dict(item) for item in existing if item["detection_id"] not in touched)
    document = {"schema_version": BOX_SCHEMA_VERSION, "image": base_filename,
                "image_size": size, "detections": updated}
    write_vessel_boxes(masks_dir, base_filename, document)
    return _validate_document(document, base_filename)


def create_approved_crops(image_path: Path, cards_dir: Path, document: dict[str, Any],
                          margin: int) -> list[dict[str, Any]]:
    """Write unmasked original-resolution rectangular crops for approved boxes."""
    image = Image.open(image_path).convert("RGB")
    size = [image.width, image.height]
    cards_dir = Path(cards_dir)
    cards_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for item in document.get("detections", []):
        if not item.get("approved") or item.get("status") == "deleted":
            continue
        reviewed_bbox = normalize_bbox(item["reviewed_bbox"], size)
        crop_bbox = expanded_crop_bbox(reviewed_bbox, size, margin)
        crop_file = f"{item['vessel_id']}.png"
        image.crop(tuple(crop_bbox)).save(cards_dir / crop_file)
        manifest.append({
            "vessel_id": item["vessel_id"],
            "detection_id": item["detection_id"],
            "image": document.get("image"),
            "crop_file": crop_file,
            "page_bbox": reviewed_bbox,
            "crop_bbox": crop_bbox,
            "crop_margin_pixels": max(0, int(margin)),
            "confidence": item.get("confidence"),
            "source": item.get("source"),
            "mask_provenance": item.get("mask_provenance"),
        })
    return manifest


def write_crop_manifest(cards_dir: Path, entries: list[dict[str, Any]], margin: int) -> Path:
    path = Path(cards_dir) / CROP_MANIFEST
    _atomic_json(path, {"schema_version": 1, "crop_margin_pixels": max(0, int(margin)),
                        "vessels": entries})
    return path


__all__ = [
    "BOX_SIDECAR_SUFFIX", "CROP_MANIFEST", "VesselBoxError", "apply_box_review",
    "bbox_iou", "create_approved_crops", "display_bbox_to_original", "expanded_crop_bbox",
    "legacy_mask_component_boxes", "migrate_legacy_review", "normalize_bbox", "read_vessel_boxes",
    "reconcile_yolo_detections", "write_crop_manifest", "write_vessel_boxes",
]
