"""Diagnostic side-profile segmentation for approved SherdScope crops.

This module proposes an isolated profile mask from an already-approved vessel
crop.  It keeps the proposal separate from the original crop so reviewer edits
can stay authoritative and reruns never destroy evidence.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.ndimage import (
    binary_dilation,
    binary_fill_holes,
    label as component_labels,
    distance_transform_edt,
)
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import binary_opening, disk, remove_small_objects


PROFILE_SCHEMA_VERSION = 1
PROFILE_ALGORITHM_VERSION = "thick-profile-v3"
PROFILE_SIDECAR = "profile_review.json"
PROFILE_DIR = "profiles"
AUTO_DIR = "auto"
ACCEPTED_DIR = "accepted"
SOURCE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ProfileProposal:
    bbox: list[int] | None
    confidence: float
    reasons: list[str]
    candidate_count: int
    ink_area: int
    profile_area: int
    algorithm_version: str = PROFILE_ALGORITHM_VERSION

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def profile_review_path(cards_dir: Path) -> Path:
    return Path(cards_dir) / PROFILE_SIDECAR


def profile_mask_path(cards_dir: Path, filename: str, kind: str) -> Path:
    if kind not in {AUTO_DIR, ACCEPTED_DIR}:
        raise ValueError("Profile mask kind must be 'auto' or 'accepted'")
    stem = Path(filename).stem
    return Path(cards_dir) / PROFILE_DIR / kind / f"{stem}_profile.png"


def list_card_files(cards_dir: Path) -> list[Path]:
    cards_dir = Path(cards_dir)

    def natural_key(path: Path) -> list[Any]:
        import re

        return [int(part) if part.isdigit() else part.lower()
                for part in re.split(r"(\d+)", path.name)]

    return sorted(
        [path for path in cards_dir.iterdir()
         if path.is_file() and path.suffix.lower() in SOURCE_EXTENSIONS],
        key=natural_key,
    )


def read_profile_review(cards_dir: Path) -> dict[str, Any]:
    path = profile_review_path(cards_dir)
    if not path.exists():
        return {"schema_version": PROFILE_SCHEMA_VERSION, "profiles": {}}
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    profiles = data.get("profiles") if isinstance(data, dict) else {}
    return {"schema_version": PROFILE_SCHEMA_VERSION,
            "profiles": profiles if isinstance(profiles, dict) else {}}


def write_profile_review(cards_dir: Path, document: dict[str, Any]) -> Path:
    clean = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "profiles": document.get("profiles", {}),
    }
    path = profile_review_path(cards_dir)
    _atomic_json(path, clean)
    return path


def _ink_mask(image: Image.Image, *, loose: bool = False) -> tuple[np.ndarray, int]:
    gray = np.asarray(image.convert("L"))
    if gray.size == 0:
        return np.zeros_like(gray, dtype=bool), 0
    try:
        otsu = int(threshold_otsu(gray))
    except ValueError:
        otsu = 200
    threshold = min(238 if loose else 215, max(120 if loose else 90, otsu + (58 if loose else 0)))
    ink = gray < threshold
    min_area = max(12, int(gray.size * 0.00008))
    ink = remove_small_objects(ink, min_size=min_area)
    return np.asarray(ink, dtype=bool), threshold


def _bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _restore_profile_outline(seed: np.ndarray, strict_mask: np.ndarray, loose_mask: np.ndarray,
                             dist: np.ndarray) -> np.ndarray:
    """Recover the profile boundary after the thick center has been selected.

    The detector uses strict ink to avoid long construction lines.  The final
    mask then expands into a looser ink mask, but only close to the selected
    thick region, so anti-aliased profile outlines are retained without pulling
    in an entire horizontal drawing line.
    """
    height, width = strict_mask.shape
    positive = dist[dist > 0]
    max_dist = float(positive.max()) if positive.size else 1.0
    strict_reach = max(3, min(24, int(round(max_dist * 1.2)) + 2))
    outline_reach = max(strict_reach + 2, min(42, int(round(max_dist * 2.4)) + 4))

    strict_body = binary_dilation(seed, iterations=strict_reach) & strict_mask
    strict_body = binary_fill_holes(strict_body)
    search_area = binary_dilation(strict_body, iterations=outline_reach)
    nearby_loose_ink = loose_mask & search_area

    labels = label(nearby_loose_ink, connectivity=2)
    touching = binary_dilation(strict_body, iterations=2)
    restored = np.zeros_like(strict_mask, dtype=bool)
    for region in regionprops(labels):
        component = labels == region.label
        if (component & touching).any():
            restored |= component

    if not restored.any():
        restored = strict_body

    restored = binary_fill_holes(restored)
    restored &= binary_dilation(strict_body, iterations=outline_reach)

    # Force the accepted proposal to be supported by a real thick body.  This
    # removes one- and two-pixel construction branches even when they touch the
    # profile, then lets only the immediate edge around that thick body return.
    opening_radius = 2 if max_dist >= 3.0 else 1
    thick_supported = binary_opening(restored, footprint=disk(opening_radius))
    if thick_supported.any():
        thick_supported = binary_fill_holes(thick_supported)
        edge_reach = max(2, min(7, int(round(max_dist * 0.35)) + 1))
        restored = restored & binary_dilation(thick_supported, iterations=edge_reach)

    body_distance = distance_transform_edt(~binary_dilation(thick_supported if thick_supported.any() else strict_body, iterations=1))
    branch_cutoff = max(2.0, min(5.0, strict_reach * 0.32))
    thin_line_band = (dist <= 2.2) & (body_distance > branch_cutoff)
    restored &= ~thin_line_band
    restored = binary_fill_holes(restored)
    return np.asarray(restored, dtype=bool)


def _component_candidates(core: np.ndarray, ink: np.ndarray, loose_ink: np.ndarray,
                          dist: np.ndarray) -> list[dict[str, Any]]:
    labels = label(core, connectivity=2)
    height, width = ink.shape
    candidates = []
    for region in regionprops(labels):
        if region.area < max(4, int(ink.size * 0.00004)):
            continue
        seed = labels == region.label
        grown = _restore_profile_outline(seed, ink, loose_ink, dist)
        bbox = _bbox(grown)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        area = int(grown.sum())
        box_area = max(1, (x2 - x1) * (y2 - y1))
        mean_thickness = float(dist[grown].mean()) if area else 0.0
        solidity = area / box_area
        x_center = (x1 + x2) / 2
        edge_score = max(x_center / max(1, width), (width - x_center) / max(1, width))
        vertical_score = min(1.0, (y2 - y1) / max(1, height * 0.55))
        score = (
            area * 1.0
            + mean_thickness * area * 0.55
            + solidity * area * 0.35
            + edge_score * area * 0.2
            + vertical_score * area * 0.2
        )
        candidates.append({
            "mask": np.asarray(grown, dtype=bool),
            "bbox": bbox,
            "area": area,
            "mean_thickness": mean_thickness,
            "solidity": solidity,
            "edge_score": edge_score,
            "score": float(score),
        })
    return sorted(candidates, key=lambda item: item["score"], reverse=True)


def propose_profile_mask(image: Image.Image) -> tuple[np.ndarray, ProfileProposal]:
    """Return a binary profile proposal and compact evidence.

    The proposal favors thick filled ink regions.  Thin construction lines can
    be present, even connected, without automatically becoming part of the mask.
    """
    image = image.convert("RGB")
    ink, threshold = _ink_mask(image)
    loose_ink, loose_threshold = _ink_mask(image, loose=True)
    reasons: list[str] = [f"ink_threshold={threshold}"]
    if loose_threshold != threshold:
        reasons.append(f"outline_threshold={loose_threshold}")
    if not ink.any():
        empty = np.zeros((image.height, image.width), dtype=np.uint8)
        return empty, ProfileProposal(
            bbox=None, confidence=0.0, reasons=["no_visible_ink"],
            candidate_count=0, ink_area=0, profile_area=0)

    dist = distance_transform_edt(ink)
    positive = dist[dist > 0]
    max_dist = float(positive.max()) if positive.size else 0.0
    thick_cutoff = max(2.0, min(max_dist * 0.58, float(np.percentile(positive, 84))))
    if max_dist < 2.5:
        thick_cutoff = max(1.0, max_dist * 0.55)
        reasons.append("thin_ink_fallback")
    core = ink & (dist >= thick_cutoff)
    core = remove_small_objects(core, min_size=max(3, int(ink.size * 0.00002)))

    if not core.any():
        labels = label(ink, connectivity=2)
        props = sorted(regionprops(labels), key=lambda region: region.area, reverse=True)
        if not props:
            empty = np.zeros((image.height, image.width), dtype=np.uint8)
            return empty, ProfileProposal(
                bbox=None, confidence=0.0, reasons=["no_profile_candidate"],
                candidate_count=0, ink_area=int(ink.sum()), profile_area=0)
        chosen = labels == props[0].label
        reasons.append("connected_component_fallback")
        bbox = _bbox(chosen)
        confidence = 0.35 if props[0].area > 0 else 0.0
        return (chosen.astype(np.uint8) * 255), ProfileProposal(
            bbox=bbox, confidence=confidence, reasons=reasons,
            candidate_count=len(props), ink_area=int(ink.sum()),
            profile_area=int(chosen.sum()))

    candidates = _component_candidates(core, ink, loose_ink, dist)
    if not candidates:
        empty = np.zeros((image.height, image.width), dtype=np.uint8)
        return empty, ProfileProposal(
            bbox=None, confidence=0.0, reasons=["no_profile_candidate"],
            candidate_count=0, ink_area=int(ink.sum()), profile_area=0)

    chosen = candidates[0]
    second_score = candidates[1]["score"] if len(candidates) > 1 else 0.0
    dominance = chosen["score"] / max(1.0, second_score)
    area_share = chosen["area"] / max(1, int(ink.sum()))
    confidence = 0.42
    confidence += min(0.28, (dominance - 1.0) * 0.16)
    confidence += min(0.18, area_share * 0.35)
    confidence += min(0.12, chosen["mean_thickness"] / max(1.0, max_dist) * 0.2)
    confidence = round(float(max(0.0, min(0.98, confidence))), 3)

    x1, y1, x2, y2 = chosen["bbox"]
    if len(candidates) > 1 and dominance < 1.35:
        reasons.append("several_similar_candidates")
        confidence = min(confidence, 0.72)
    if area_share < 0.12:
        reasons.append("selected_profile_is_small_part_of_ink")
        confidence = min(confidence, 0.68)
    if x1 <= 1 or y1 <= 1 or x2 >= image.width - 1 or y2 >= image.height - 1:
        reasons.append("profile_touches_crop_boundary")
        confidence = min(confidence, 0.82)

    return chosen["mask"].astype(np.uint8) * 255, ProfileProposal(
        bbox=chosen["bbox"], confidence=confidence, reasons=reasons,
        candidate_count=len(candidates), ink_area=int(ink.sum()),
        profile_area=int(chosen["area"]))


def recover_profile_mask(
    image: Image.Image,
    current_mask: np.ndarray | Image.Image,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Recover thick profile ink connected to a reviewed mask.

    This is the in-app version of the training curator's ``G`` recovery tool.
    It grows only from a thick ink core that touches the current mask, so long
    construction lines and isolated side branches are not pulled in.
    """
    image = image.convert("RGB")
    if isinstance(current_mask, Image.Image):
        current = np.asarray(current_mask.convert("L")) > 32
    else:
        current = np.asarray(current_mask) > 0
    if current.shape != (image.height, image.width):
        raise ValueError("Current profile mask dimensions do not match the crop")

    proposed_raw, evidence = propose_profile_mask(image)
    proposed = np.asarray(proposed_raw) > 0
    loose_ink, _threshold = _ink_mask(image, loose=True)
    guided = np.zeros_like(current, dtype=bool)
    if loose_ink.any() and current.any():
        thickness = distance_transform_edt(loose_ink)
        # Scale the curator's high-resolution radius to ordinary SherdScope crops while
        # retaining a thick-core requirement that rejects hairline guides.
        thick_radius = max(3.0, min(6.0, min(image.size) * 0.012))
        thick_core = loose_ink & (thickness >= thick_radius)
        labels, count = component_labels(
            thick_core, structure=np.ones((3, 3), dtype=np.uint8)
        )
        if count:
            contact = binary_dilation(current, iterations=3)
            touching_labels = np.unique(labels[contact & (labels > 0)])
            if touching_labels.size:
                selected_core = np.isin(labels, touching_labels)
                guided = loose_ink & binary_dilation(selected_core, iterations=5)
                guided = np.asarray(binary_fill_holes(guided), dtype=bool)

    recovery = guided if current.any() else proposed
    if not recovery.any():
        raise ValueError("No thick profile ink was found near the current mask")
    agreement = 1.0
    if current.any():
        overlap = int((current & recovery).sum())
        agreement = overlap / max(1, min(int(current.sum()), int(recovery.sum())))
        if agreement < 0.45:
            raise ValueError(
                "The recovery proposal does not agree with this mask enough to apply safely"
            )
        recovered = current | recovery
    else:
        recovered = recovery
    return recovered.astype(np.uint8) * 255, {
        "added_pixels": int((recovered & ~current).sum()),
        "agreement": round(float(agreement), 4),
        "proposal_confidence": round(float(evidence.confidence), 4),
        "proposal_reasons": evidence.reasons,
    }


def save_profile_mask(path: Path, mask: np.ndarray | Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(mask, Image.Image):
        output = mask.convert("L")
    else:
        output = Image.fromarray((np.asarray(mask) > 0).astype(np.uint8) * 255, mode="L")
    output.save(path)


def generate_profile_proposals(
    cards_dir: Path,
    force: bool = False,
    target_filenames: set[str] | None = None,
) -> dict[str, Any]:
    cards_dir = Path(cards_dir)
    document = read_profile_review(cards_dir)
    profiles = document["profiles"]
    generated = 0
    skipped_reviewed = 0
    skipped_out_of_scope = 0
    failed = 0

    for card_path in list_card_files(cards_dir):
        filename = card_path.name
        if target_filenames is not None and filename not in target_filenames:
            skipped_out_of_scope += 1
            continue
        record = dict(profiles.get(filename) or {})
        reviewed = record.get("review_status") in {"approved", "edited", "no_profile"}
        if reviewed and not force:
            skipped_reviewed += 1
            continue
        try:
            mask, proposal = propose_profile_mask(Image.open(card_path))
            auto_path = profile_mask_path(cards_dir, filename, AUTO_DIR)
            save_profile_mask(auto_path, mask)
            # Automatic proposals and accepted reviewer masks are separate
            # artifacts. Even an explicit auto-regeneration must never replace
            # a reviewed accepted mask or return it to the pending queue.
            if not record.get("accepted_mask") or not reviewed:
                accepted_path = profile_mask_path(cards_dir, filename, ACCEPTED_DIR)
                save_profile_mask(accepted_path, mask)
                record["accepted_mask"] = accepted_path.relative_to(cards_dir).as_posix()
            record.update({
                "filename": filename,
                "auto_mask": auto_path.relative_to(cards_dir).as_posix(),
                "proposal": proposal.as_dict(),
                "review_status": record.get("review_status") or "pending",
                "review_note": record.get("review_note") or "",
            })
            profiles[filename] = record
            generated += 1
        except Exception as exc:
            failed += 1
            profiles[filename] = {
                **record,
                "filename": filename,
                "review_status": record.get("review_status") or "failed",
                "review_note": record.get("review_note") or "",
                "proposal": {
                    "bbox": None,
                    "confidence": 0.0,
                    "reasons": [f"proposal_failed: {exc}"],
                    "candidate_count": 0,
                    "ink_area": 0,
                    "profile_area": 0,
                    "algorithm_version": PROFILE_ALGORITHM_VERSION,
                },
            }

    write_profile_review(cards_dir, document)
    return {
        "total_cards": len(list_card_files(cards_dir)),
        "generated": generated,
        "skipped_reviewed": skipped_reviewed,
        "skipped_out_of_scope": skipped_out_of_scope,
        "failed": failed,
    }


__all__ = [
    "ACCEPTED_DIR",
    "AUTO_DIR",
    "PROFILE_ALGORITHM_VERSION",
    "PROFILE_DIR",
    "PROFILE_SCHEMA_VERSION",
    "PROFILE_SIDECAR",
    "ProfileProposal",
    "generate_profile_proposals",
    "list_card_files",
    "profile_mask_path",
    "propose_profile_mask",
    "recover_profile_mask",
    "read_profile_review",
    "save_profile_mask",
    "write_profile_review",
]
