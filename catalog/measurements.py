"""Deterministic scale-bar and rim-diameter measurements for Hesban drawings."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import pandas as pd
from PIL import Image


DETECTOR_VERSION = "hesban-scale-diameter-v2"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:20]


def _gray_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"))


def _binary_ink(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 35, 13,
    )


def _long_row_coverage(binary: np.ndarray) -> float:
    if binary.size == 0:
        return 0.0
    return float(np.max(np.count_nonzero(binary, axis=1))) / max(1, binary.shape[1])


def _divider_count(binary: np.ndarray) -> int:
    """Count strong vertical divider groups inside a candidate ruler."""
    if binary.size == 0:
        return 0
    projection = np.count_nonzero(binary, axis=0)
    threshold = max(2, int(round(binary.shape[0] * 0.55)))
    active = projection >= threshold
    groups = 0
    inside = False
    for value in active:
        if value and not inside:
            groups += 1
            inside = True
        elif not value:
            inside = False
    return groups


def _alternation_score(binary: np.ndarray, segments: int = 5) -> float:
    if binary.shape[1] < segments * 2 or binary.shape[0] < 3:
        return 0.0
    inner = binary[max(0, binary.shape[0] // 4):max(1, binary.shape[0] * 3 // 4)]
    values = []
    for index in range(segments):
        left = round(index * inner.shape[1] / segments)
        right = round((index + 1) * inner.shape[1] / segments)
        cell = inner[:, left:right]
        values.append(float(np.count_nonzero(cell)) / max(1, cell.size))
    if len(values) < 2:
        return 0.0
    differences = [abs(values[index] - values[index - 1])
                   for index in range(1, len(values))]
    return min(1.0, float(np.mean(differences)) * 2.5)


def _straighten_candidate(binary: np.ndarray) -> tuple[np.ndarray, tuple[float, float],
                                                        tuple[float, float]]:
    """Deskew a ruler candidate and return its long-axis endpoints.

    Hesban scans are sometimes rotated by a few degrees.  Measuring horizontal
    row coverage on the uncorrected crop rejects those otherwise valid rulers.
    The minimum-area ink rectangle gives us a deterministic local correction
    without rotating or resampling the full publication page.
    """
    ys, xs = np.nonzero(binary)
    if len(xs) < 2:
        return binary, (0.0, 0.0), (0.0, 0.0)
    points = np.column_stack((xs, ys)).astype(np.float32)
    rectangle = cv2.minAreaRect(points)
    box = cv2.boxPoints(rectangle)
    edges = [(box[(index + 1) % 4] - box[index], index) for index in range(4)]
    edge, index = max(edges, key=lambda item: float(np.linalg.norm(item[0])))
    length = float(np.linalg.norm(edge))
    if length <= 0:
        return binary, (0.0, 0.0), (0.0, 0.0)
    unit = edge / length
    center = np.asarray(rectangle[0], dtype=np.float32)
    p1 = tuple(float(value) for value in center - unit * length / 2)
    p2 = tuple(float(value) for value in center + unit * length / 2)
    angle = math.degrees(math.atan2(float(edge[1]), float(edge[0])))
    padded = cv2.copyMakeBorder(binary, binary.shape[0], binary.shape[0],
                                binary.shape[1] // 4, binary.shape[1] // 4,
                                cv2.BORDER_CONSTANT, value=0)
    padded_center = (padded.shape[1] / 2, padded.shape[0] / 2)
    rotated = cv2.warpAffine(
        padded, cv2.getRotationMatrix2D(padded_center, angle, 1.0),
        (padded.shape[1], padded.shape[0]), flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    rotated_ys, rotated_xs = np.nonzero(rotated)
    if len(rotated_xs):
        rotated = rotated[rotated_ys.min():rotated_ys.max() + 1,
                          rotated_xs.min():rotated_xs.max() + 1]
    return rotated, p1, p2


def detect_hesban_scale(image_path: Path) -> dict[str, Any]:
    """Find the standard segmented 0--10 CM ruler anywhere on the page."""
    image_path = Path(image_path)
    gray = _gray_image(image_path)
    height, width = gray.shape
    binary = _binary_ink(gray)
    # Some Hesban plates use only part of a publication page, placing their
    # ruler well above the lower-page area. Search the complete rendered page;
    # the divider and alternating-block checks below identify the ruler by its
    # structure rather than by an assumed vertical position.
    search_top = 0
    search = binary
    # Close small scan/dropout gaps before finding the connected ruler body.
    # Structural validation below prevents this wider join from accepting
    # ordinary vessel lines.
    join_width = max(5, int(round(width * 0.006)))
    joined = cv2.morphologyEx(
        search, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (join_width, 3)),
    )
    contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[dict[str, Any]] = []
    for contour in contours:
        x, local_y, candidate_width, candidate_height = cv2.boundingRect(contour)
        y = local_y + search_top
        if not (width * 0.018 <= candidate_width <= width * 0.28):
            continue
        if not (2 <= candidate_height <= max(12, height * 0.045)):
            continue
        if candidate_width / max(1, candidate_height) < 5.0:
            continue
        pad_x = max(2, int(candidate_width * 0.025))
        pad_y = max(2, int(candidate_height * 0.5))
        x1, x2 = max(0, x - pad_x), min(width, x + candidate_width + pad_x)
        y1, y2 = max(0, y - pad_y), min(height, y + candidate_height + pad_y)
        structural_crop = binary[y:y + candidate_height, x:x + candidate_width]
        structural_crop, local_p1, local_p2 = _straighten_candidate(structural_crop)
        coverage = _long_row_coverage(structural_crop)
        dividers = _divider_count(structural_crop)
        alternation = _alternation_score(structural_crop)
        if coverage < 0.62 or not (3 <= dividers <= 9) or alternation < 0.18:
            continue
        score = min(1.0, 0.45 * coverage + 0.30 * min(dividers, 6) / 6 +
                    0.25 * alternation)
        # Use ink extents, not the contour/padded crop bounds, as ruler endpoints.
        if math.dist(local_p1, local_p2) <= 0:
            continue
        ruler_p1 = [round(local_p1[0] + x, 3), round(local_p1[1] + y, 3)]
        ruler_p2 = [round(local_p2[0] + x, 3), round(local_p2[1] + y, 3)]
        candidates.append({
            "p1": ruler_p1, "p2": ruler_p2,
            "real_cm": 10.0, "zone": None, "method": "automatic",
            "status": "verified_automatic", "confidence": round(score, 4),
            "evidence_bounds": [x1, y1, x2, y2],
            "detector_version": DETECTOR_VERSION,
            "page_fingerprint": _fingerprint(image_path),
            "image_size": [width, height], "detected_at": _now(),
        })
    candidates.sort(key=lambda item: item["confidence"], reverse=True)
    if not candidates:
        return {
            "status": "unresolved", "method": "automatic",
            "real_cm": 10.0, "confidence": 0.0,
            "detector_version": DETECTOR_VERSION,
            "page_fingerprint": _fingerprint(image_path),
            "image_size": [width, height], "detected_at": _now(),
            "warning": "scale_not_found",
        }
    best = candidates[0]
    if len(candidates) > 1 and candidates[1]["confidence"] >= best["confidence"] - 0.08:
        best = dict(best)
        best["status"] = "unresolved"
        best["warning"] = "multiple_scale_candidates"
    best["px_per_cm"] = round(
        math.dist(best["p1"], best["p2"]) / float(best["real_cm"]), 6)
    return best


def normalize_calibration(calibration: dict[str, Any]) -> dict[str, Any]:
    result = dict(calibration or {})
    result.setdefault("real_cm", 10.0)
    result.setdefault("method", "manual" if result.get("p1") and result.get("p2") else "automatic")
    result.setdefault("status", "verified_manual" if result.get("method") == "manual"
                      else "verified_automatic")
    if result.get("status") == "verified":
        result["status"] = "verified_manual" if result.get("method") == "manual" else "verified_automatic"
    elif result.get("status") == "suggested" and result.get("method") == "automatic":
        result["status"] = "verified_automatic"
    if result.get("p1") and result.get("p2") and float(result.get("real_cm", 0) or 0) > 0:
        result["px_per_cm"] = round(
            math.dist(result["p1"], result["p2"]) / float(result["real_cm"]), 6)
    return result


def manual_calibration(image_path: Path, p1: Iterable[float], p2: Iterable[float],
                       real_cm: float = 10.0) -> dict[str, Any]:
    image_path = Path(image_path)
    points = [[float(value) for value in point] for point in (p1, p2)]
    if (any(len(point) != 2 for point in points) or
            any(not math.isfinite(value) for point in points for value in point) or
            math.dist(*points) < 5):
        raise ValueError("Scale endpoints must be at least five pixels apart")
    if not math.isfinite(float(real_cm)) or float(real_cm) <= 0:
        raise ValueError("Scale distance must be positive")
    with Image.open(image_path) as image:
        width, height = image.size
    if any(point[0] < 0 or point[1] < 0 or point[0] > width or point[1] > height
           for point in points):
        raise ValueError("Scale endpoints must lie inside the source page")
    return normalize_calibration({
        "p1": points[0], "p2": points[1], "real_cm": float(real_cm),
        "zone": None, "method": "manual", "status": "verified_manual",
        "confidence": 1.0, "detector_version": DETECTOR_VERSION,
        "page_fingerprint": _fingerprint(image_path),
        "image_size": [width, height], "verified_at": _now(),
    })


def detect_rim_diameter(image_path: Path, bbox: Iterable[int],
                        calibration: dict[str, Any]) -> dict[str, Any]:
    """Measure detected rim ink; the card bounding-box width is never the result."""
    image_path = Path(image_path)
    gray = _gray_image(image_path)
    page_height, page_width = gray.shape
    try:
        coordinates = [float(value) for value in bbox]
    except (TypeError, ValueError):
        coordinates = []
    if (len(coordinates) != 4 or any(not math.isfinite(value) for value in coordinates) or
            coordinates[2] <= coordinates[0] or coordinates[3] <= coordinates[1]):
        return {
            "status": "unresolved", "method": "automatic",
            "detector_version": DETECTOR_VERSION, "measured_at": _now(),
            "image_size": [page_width, page_height], "warning": "invalid_drawing_bbox",
        }
    x1, y1, x2, y2 = [int(value) for value in coordinates]
    margin_x = max(8, int((x2 - x1) * 0.025))
    margin_y = max(8, int((y2 - y1) * 0.05))
    left, top = max(0, x1 - margin_x), max(0, y1 - margin_y)
    right, bottom = min(page_width, x2 + margin_x), min(page_height, y2 + margin_y)
    crop_gray = gray[top:bottom, left:right]
    binary = _binary_ink(crop_gray)
    height, width = binary.shape
    base = {
        "status": "unresolved", "method": "automatic",
        "detector_version": DETECTOR_VERSION, "measured_at": _now(),
        "crop": [left, top, right, bottom],
        "image_size": [page_width, page_height],
    }
    try:
        px_per_cm = float(calibration.get("px_per_cm", 0) or 0)
    except (TypeError, ValueError):
        px_per_cm = 0.0
    calibration_usable = calibration.get("status") in {
        "suggested", "verified", "verified_automatic", "verified_manual"}
    if (not calibration_usable or not math.isfinite(px_per_cm) or px_per_cm <= 0 or
            width < 20 or height < 10):
        return {**base, "warning": "missing_scale_calibration"}

    # Hesban profiles consistently place the rim at the top of the drawing.
    # Search only the top 10% of the original card box (not the expanded crop)
    # so lower reconstruction lines and vessel tails cannot become the rim.
    bbox_top = max(0, y1 - top)
    bbox_height = max(1, y2 - y1)
    search_bottom = min(height, bbox_top + max(6, int(math.ceil(bbox_height * 0.10))))
    rim_band = binary[bbox_top:search_bottom, :]
    joined_band = cv2.morphologyEx(
        rim_band, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, width // 120), 1)),
    )
    horizontal = cv2.morphologyEx(
        joined_band, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(12, width // 9), 1)),
    )
    rows: list[tuple[int, int, int, int]] = []
    for local_y in range(horizontal.shape[0]):
        xs = np.flatnonzero(horizontal[local_y])
        if len(xs) < max(10, width * 0.18):
            continue
        rows.append((int(xs.max() - xs.min()), bbox_top + local_y,
                     int(xs.min()), int(xs.max())))
    if not rows:
        return {**base, "warning": "rim_span_not_found"}
    # The topmost credible stroke is the rim. Printed/scanned lines can vary by
    # a few pixels across their length, so union the first few rows belonging to
    # that same stroke rather than measuring one razor-thin pixel row.
    first_y = min(row[1] for row in rows)
    row_tolerance = max(4, int(math.ceil(bbox_height * 0.02)))
    rim_rows = [row for row in rows if row[1] <= first_y + row_tolerance]
    rim_y = int(round(float(np.median([row[1] for row in rim_rows]))))
    rim_left = min(row[2] for row in rim_rows)
    rim_right = max(row[3] for row in rim_rows)
    span_width = rim_right - rim_left
    if span_width <= 0:
        return {**base, "warning": "rim_span_not_found"}

    vertical = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, height // 5))),
    )
    axis_candidates: list[tuple[int, int, int, int]] = []
    candidate_left = max(int(width * 0.20), rim_left + max(4, span_width // 8))
    candidate_right = min(int(width * 0.80), rim_right - max(4, span_width // 8))
    for x in range(candidate_left, max(candidate_right, candidate_left + 1)):
        ys = np.flatnonzero(vertical[:, x])
        touches_rim = len(ys) and int(ys.min()) <= rim_y + max(5, height // 30)
        if touches_rim and len(ys) >= max(8, height * 0.16):
            axis_candidates.append((len(ys), x, int(ys.min()), int(ys.max())))
    if not axis_candidates:
        axis_x = None
        centreline = None
        axis_diameter_px = None
    else:
        target_mid = (rim_left + rim_right) / 2
        # The reconstruction axis is the vertical line nearest the midpoint of
        # the illustrated rim.  Sorting by length first can select a long outer
        # vessel wall and produce a convincing but wrong diameter.
        candidates = sorted(axis_candidates,
                            key=lambda item: (abs(item[1] - target_mid), -item[0]))
        _, axis_x, axis_top, axis_bottom = candidates[0]
        centreline = [left + axis_x, top + axis_top, left + axis_x, top + axis_bottom]
        # The left rim edge is physically connected to the reconstruction line
        # in Hesban drawings. The right black profile is commonly separated by
        # a small publication gap, so infer the full diameter by mirroring the
        # reliable connected left radius around the centreline.
        connected_radius = axis_x - rim_left
        axis_diameter_px = 2.0 * float(connected_radius) if connected_radius > 0 else None

    observed_span_px = float(rim_right - rim_left)
    agreement = (abs(observed_span_px - axis_diameter_px) / axis_diameter_px
                 if axis_diameter_px is not None and observed_span_px else None)
    # A small difference between the observed rim stroke and the mirrored
    # centreline estimate is normal in scanned drawings. Keep 5--15% as an
    # accepted advisory range; only larger disagreement requires attention.
    ideal_agreement = agreement is not None and agreement <= 0.05
    acceptable_agreement = agreement is not None and agreement <= 0.15
    confidence = 0.9 if ideal_agreement else 0.75 if acceptable_agreement else 0.55
    status = "verified_automatic" if acceptable_agreement else "unresolved"
    suggested_cm = axis_diameter_px / px_per_cm if axis_diameter_px else None
    inferred_right = (rim_left + axis_diameter_px
                      if axis_diameter_px is not None else rim_right)
    bbox_width = float(x2 - x1)
    bbox_height = float(y2 - y1)
    allowed_x_margin = bbox_width * 0.15
    allowed_y_margin = bbox_height * 0.15
    inferred_page_endpoints = [
        [left + rim_left, top + rim_y],
        [left + inferred_right, top + rim_y],
    ]
    exceeds_bbox_limit = any(
        point_x < x1 - allowed_x_margin or point_x > x2 + allowed_x_margin
        or point_y < y1 - allowed_y_margin or point_y > y2 + allowed_y_margin
        for point_x, point_y in inferred_page_endpoints
    )
    if exceeds_bbox_limit:
        status = "unresolved"
        confidence = min(confidence, 0.25)
        suggested_cm = None
    result = {
        **base, "status": status, "confidence": confidence,
        "suggested_cm": suggested_cm,
        "verified_cm": suggested_cm if status == "verified_automatic" else None,
        "diameter_px": axis_diameter_px, "axis_diameter_px": axis_diameter_px,
        "observed_span_px": observed_span_px,
        "agreement": agreement,
        "rim_endpoints": inferred_page_endpoints,
        "observed_rim_endpoints": [[left + rim_left, top + rim_y],
                                   [left + rim_right, top + rim_y]],
        "connected_radius_endpoints": ([[left + rim_left, top + rim_y],
                                         [left + axis_x, top + rim_y]]
                                        if axis_x is not None else None),
        "centreline": centreline, "scale_px_per_cm": px_per_cm,
        "rim_search_region": [left, top + bbox_top, right, top + search_bottom],
        "scale_page_fingerprint": calibration.get("page_fingerprint", ""),
    }
    if exceeds_bbox_limit:
        result["warning"] = "rim_endpoints_exceed_drawing_bbox"
    elif status == "verified_automatic" and not ideal_agreement:
        result["warning"] = "diameter_estimators_minor_disagreement"
    elif status == "unresolved":
        result["warning"] = "diameter_estimators_disagree" if axis_diameter_px else "centreline_not_found"
    return result


def verified_measurement(measurement: dict[str, Any], value_cm: float | None = None,
                         rim_endpoints: list[list[float]] | None = None,
                         px_per_cm: float | None = None) -> dict[str, Any]:
    result = dict(measurement or {})
    if rim_endpoints is not None:
        if (len(rim_endpoints) != 2 or any(len(point) != 2 for point in rim_endpoints) or
                any(not math.isfinite(float(value))
                    for point in rim_endpoints for value in point)):
            raise ValueError("Rim endpoints must contain two finite x/y points")
        result["rim_endpoints"] = rim_endpoints
    if px_per_cm is not None:
        result["scale_px_per_cm"] = float(px_per_cm)
    if value_cm is None and result.get("rim_endpoints") and result.get("scale_px_per_cm"):
        value_cm = math.dist(*result["rim_endpoints"]) / float(result["scale_px_per_cm"])
    if value_cm is None or not math.isfinite(float(value_cm)) or float(value_cm) <= 0:
        raise ValueError("A verified diameter must be a positive number")
    verified_at = _now()
    result.update({
        "verified_cm": float(value_cm), "status": "verified_manual",
        "method": "manual", "verified_at": verified_at,
    })
    result.setdefault("reviewer_history", []).append({
        "action": "diameter_verified", "at": verified_at,
        "verified_cm": float(value_cm),
    })
    return result


def _manifest_pages(project_path: Path) -> list[dict[str, Any]]:
    manifest_path = Path(project_path) / "page_manifest.json"
    if not manifest_path.exists():
        return []
    try:
        return list(json.loads(manifest_path.read_text(encoding="utf-8")).get("pages", []))
    except (OSError, ValueError, TypeError):
        return []


def _read_scale_sidecar(project_path: Path, image_name: str) -> dict[str, Any]:
    sidecar = Path(project_path) / "masks" / f"{Path(image_name).stem}_scale.json"
    if not sidecar.exists():
        return {}
    try:
        scales = json.loads(sidecar.read_text(encoding="utf-8")).get("scales", [])
        normalized = [normalize_calibration(scale) for scale in scales if isinstance(scale, dict)]
        preferred = [scale for scale in normalized
                     if not scale.get("zone") and scale.get("status") != "unresolved"]
        # A regional calibration is valid only for drawings whose centroid lies
        # in that region. It cannot safely stand in for the page-wide ruler.
        return preferred[0] if preferred else {}
    except (OSError, ValueError, TypeError, IndexError):
        return {}


def persist_figure_measurements(project_path: Path, figure: dict[str, Any]) -> None:
    """Keep sidecars and card CSV synchronized with linkage/approval writes."""
    # Import lazily to avoid the module-level linker -> measurement dependency.
    # Reviewer edits are allowed while later figures are processing, so this
    # must share the same lock as linkage state and CSV approval writes.
    from catalog.linkage import _LINKAGE_STATE_LOCK
    with _LINKAGE_STATE_LOCK:
        _persist_figure_measurements(project_path, figure)


def _persist_figure_measurements(project_path: Path, figure: dict[str, Any]) -> None:
    project_path = Path(project_path)
    masks_path = project_path / "masks"
    masks_path.mkdir(parents=True, exist_ok=True)
    calibrations = figure.get("scale_calibrations", {})
    for image_name, calibration in calibrations.items():
        if not calibration.get("p1") or not calibration.get("p2"):
            continue
        sidecar = masks_path / f"{Path(image_name).stem}_scale.json"
        existing_payload: dict[str, Any] = {}
        if sidecar.exists():
            try:
                existing_payload = json.loads(sidecar.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                existing_payload = {}
        scales = [scale for scale in existing_payload.get("scales", [])
                  if isinstance(scale, dict)]
        # The linker owns the page-wide (zone-less) calibration only. Preserve
        # manually defined regional calibrations used by card extraction.
        global_index = next((index for index, scale in enumerate(scales)
                             if not scale.get("zone")), None)
        if global_index is None:
            scales.append(calibration)
        else:
            scales[global_index] = calibration
        payload = {**existing_payload, "image": Path(image_name).stem, "scales": scales}
        temporary = sidecar.with_suffix(sidecar.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(sidecar)

    info_path = project_path / "cards" / "mask_info.csv"
    if not info_path.exists():
        return
    frame = pd.read_csv(info_path, dtype=str, keep_default_na=False)
    if "mask_file" not in frame.columns:
        return
    if "px_per_cm" not in frame.columns:
        frame["px_per_cm"] = ""
    for drawing in figure.get("drawings", []):
        calibration = calibrations.get(str(drawing.get("image_name", "")), {})
        key = Path(str(drawing.get("mask_file", ""))).stem
        target = frame["mask_file"].map(lambda value: Path(str(value)).stem).eq(key)
        # Clear stale ratios when a formerly usable calibration becomes
        # ambiguous or fails the project-median check.
        frame.loc[target, "px_per_cm"] = ""
        ratio = calibration.get("px_per_cm")
        if calibration.get("status") in {"suggested", "verified", "verified_automatic", "verified_manual"} and ratio:
            frame.loc[target, "px_per_cm"] = str(ratio)
    temporary = info_path.with_suffix(info_path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(info_path)


def measure_figure(project_path: Path, figure: dict[str, Any], *,
                   project_ratios: Iterable[float] = (), persist: bool = True) -> dict[str, Any]:
    """Populate page calibrations and per-drawing suggestions in-place."""
    project_path = Path(project_path)
    images_path = project_path / "images"
    calibrations = dict(figure.get("scale_calibrations", {}))
    manifest = _manifest_pages(project_path)
    manifest_by_name = {str(page.get("image_name", "")): page for page in manifest}
    drawing_names = [str(page.get("image_name", ""))
                     for page in figure.get("drawing_pages", []) if page.get("image_name")]

    for image_name in drawing_names:
        existing = normalize_calibration(
            calibrations.get(image_name, {}) or _read_scale_sidecar(project_path, image_name))
        image_path = images_path / Path(image_name).name
        if (existing.get("status") in {"verified", "verified_manual"} and existing.get("p1") and
                existing.get("p2") and not existing.get("page_fingerprint") and
                image_path.exists()):
            existing["page_fingerprint"] = _fingerprint(image_path)
            with Image.open(image_path) as image:
                existing["image_size"] = list(image.size)
        evidence_path = images_path / Path(str(existing.get("evidence_image") or image_name)).name
        same_page = (existing.get("page_fingerprint") == _fingerprint(evidence_path)
                     if evidence_path.exists() and existing.get("page_fingerprint") else False)
        if existing.get("status") in {"verified", "verified_manual"} and same_page:
            calibrations[image_name] = existing
            continue
        if not image_path.exists():
            calibrations[image_name] = {"status": "unresolved", "warning": "image_not_found"}
            continue
        detected = detect_hesban_scale(image_path)
        if detected.get("status") == "unresolved":
            page = manifest_by_name.get(image_name, {})
            siblings = [candidate for candidate in manifest
                        if candidate.get("source_pdf") == page.get("source_pdf") and
                        candidate.get("pdf_page_index") == page.get("pdf_page_index") and
                        candidate.get("image_name") != image_name]
            for sibling in siblings:
                sibling_path = images_path / Path(str(sibling.get("image_name", ""))).name
                if not sibling_path.exists():
                    continue
                sibling_scale = detect_hesban_scale(sibling_path)
                if (sibling_scale.get("status") in {"suggested", "verified", "verified_automatic", "verified_manual"} and
                        sibling_scale.get("px_per_cm")):
                    detected = {
                        **sibling_scale, "method": "sibling_page",
                        "status": "verified_automatic", "evidence_image": sibling_path.name,
                        "applies_to_image": image_name,
                    }
                    break
        calibrations[image_name] = detected

    ratios = []
    for value in project_ratios:
        try:
            ratio = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(ratio) and ratio > 0:
            ratios.append(ratio)
    # Prefer genuinely independent pages. If this is the first measured
    # figure, fall back to the within-figure median for consistency checks.
    reference_ratios = ratios or [
        float(calibration.get("px_per_cm"))
        for calibration in calibrations.values()
        if calibration.get("status") in {"suggested", "verified", "verified_automatic", "verified_manual"} and
        calibration.get("px_per_cm")
    ]
    median = float(np.median(reference_ratios)) if reference_ratios else 0.0
    for image_name, calibration in calibrations.items():
        calibration = normalize_calibration(calibration)
        ratio = float(calibration.get("px_per_cm", 0) or 0)
        if (median and ratio and abs(ratio - median) / median > 0.03 and
                calibration.get("status") not in {"verified", "verified_manual"}):
            calibration["status"] = "unresolved"
            calibration["warning"] = "scale_median_disagreement"
            calibration["project_median_px_per_cm"] = median
        calibrations[image_name] = calibration
    figure["scale_calibrations"] = calibrations

    for drawing in figure.get("drawings", []):
        image_name = str(drawing.get("image_name", ""))
        image_path = images_path / Path(image_name).name
        calibration = calibrations.get(image_name, {})
        previous = dict(drawing.get("measurement", {}))
        same_geometry = previous.get("drawing_fingerprint") == drawing.get("fingerprint")
        same_scale = previous.get("scale_page_fingerprint") == calibration.get("page_fingerprint")
        if previous.get("status") in {"verified", "verified_manual"} and same_geometry and same_scale:
            continue
        if not image_path.exists():
            measurement = {"status": "unresolved", "warning": "image_not_found"}
        else:
            measurement = detect_rim_diameter(image_path, drawing.get("bbox", []), calibration)
        measurement["drawing_fingerprint"] = drawing.get("fingerprint", "")
        drawing["measurement"] = measurement
    if persist:
        persist_figure_measurements(project_path, figure)
    return figure
