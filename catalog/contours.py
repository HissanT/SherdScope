"""Canonical three-curve geometry for reviewed SherdScope profile masks."""

from __future__ import annotations

import hashlib
import csv
import json
import math
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_fill_holes, gaussian_filter1d
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from skimage.filters import threshold_otsu
from skimage.measure import find_contours, label, regionprops
from skimage.morphology import skeletonize

from catalog.profile_segmentation import (
    ACCEPTED_DIR,
    list_card_files,
    profile_mask_path,
    read_profile_review,
)


CONTOUR_SCHEMA_VERSION = 1
CONTOUR_ALGORITHM_VERSION = "reference-two-wall-v1"
MATCHER_DIR = "matcher"
CONTOUR_DIR = "contours"
MANIFEST_NAME = "manifest.json"
RESOLVED_REVIEW_STATUSES = {"approved", "edited", "no_profile"}
MATCHABLE_REVIEW_STATUSES = {"approved", "edited"}
DEFAULT_SAMPLES = 96


class ContourError(ValueError):
    """Raised when a useful canonical contour cannot be extracted."""


def card_stem(value: Any) -> str:
    """Remove only a real image extension from an extracted card filename."""
    text = str(value or "")
    suffix = Path(text).suffix.lower()
    return text[:-len(suffix)] if suffix in {".png", ".jpg", ".jpeg"} else text


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def matcher_root(project_path: Path) -> Path:
    return Path(project_path) / MATCHER_DIR


def contour_root(project_path: Path) -> Path:
    return matcher_root(project_path) / CONTOUR_DIR


def manifest_path(project_path: Path) -> Path:
    return contour_root(project_path) / MANIFEST_NAME


def read_manifest(project_path: Path) -> dict[str, Any]:
    path = manifest_path(project_path)
    if not path.exists():
        return {
            "schema_version": CONTOUR_SCHEMA_VERSION,
            "algorithm_version": CONTOUR_ALGORITHM_VERSION,
            "references": {},
        }
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        data = {}
    references = data.get("references") if isinstance(data, dict) else {}
    return {
        "schema_version": CONTOUR_SCHEMA_VERSION,
        "algorithm_version": CONTOUR_ALGORITHM_VERSION,
        "references": references if isinstance(references, dict) else {},
    }


def write_manifest(project_path: Path, document: dict[str, Any]) -> Path:
    clean = {
        "schema_version": CONTOUR_SCHEMA_VERSION,
        "algorithm_version": CONTOUR_ALGORITHM_VERSION,
        "references": document.get("references", {}),
    }
    path = manifest_path(project_path)
    _atomic_json(path, clean)
    return path


def source_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_stem(filename: str) -> str:
    value = "".join(char if char.isalnum() or char in "-_." else "_" for char in card_stem(filename))
    return value[:180] or hashlib.sha1(filename.encode("utf-8")).hexdigest()


def artifact_paths(project_path: Path, filename: str) -> dict[str, Path]:
    root = contour_root(project_path)
    stem = _safe_stem(filename)
    return {
        "json": root / f"{stem}.json",
        "preview": root / f"{stem}_preview.png",
        "overlay": root / f"{stem}_overlay.png",
        "clean_mask": root / f"{stem}_clean_mask.png",
    }


def library_status(project_path: Path) -> dict[str, Any]:
    cards_dir = Path(project_path) / "cards"
    document = read_profile_review(cards_dir)
    records = document.get("profiles", {})
    cards = list_card_files(cards_dir) if cards_dir.exists() else []
    manifest = read_manifest(project_path)
    entries = manifest["references"]

    counts = {
        "total": len(cards),
        "resolved": 0,
        "pending": 0,
        "no_profile": 0,
        "matchable": 0,
        "built": 0,
        "stale": 0,
        "flagged": 0,
        "unresolved_flags": 0,
        "failed": 0,
    }
    unresolved: list[str] = []
    for card in cards:
        record = records.get(card.name, {})
        status = record.get("review_status") or "not_generated"
        if status in RESOLVED_REVIEW_STATUSES:
            counts["resolved"] += 1
        else:
            counts["pending"] += 1
            unresolved.append(card.name)
        if status == "no_profile":
            counts["no_profile"] += 1
            continue
        if status not in MATCHABLE_REVIEW_STATUSES:
            continue
        counts["matchable"] += 1
        mask_path = profile_mask_path(cards_dir, card.name, ACCEPTED_DIR)
        entry = entries.get(card.name)
        if not entry or not mask_path.exists():
            counts["stale"] += 1
            continue
        current_hash = source_fingerprint(mask_path)
        artifact = contour_root(project_path) / str(entry.get("artifact", ""))
        if (
            entry.get("source_fingerprint") != current_hash
            or entry.get("algorithm_version") != CONTOUR_ALGORITHM_VERSION
            or not artifact.exists()
        ):
            counts["stale"] += 1
            continue
        if entry.get("state") == "failed":
            counts["failed"] += 1
        else:
            counts["built"] += 1
        if entry.get("flagged"):
            counts["flagged"] += 1
            if entry.get("review_resolution") not in {"accepted", "minimal"}:
                counts["unresolved_flags"] += 1

    ready_to_build = counts["total"] > 0 and counts["pending"] == 0
    ready_to_match = (
        ready_to_build
        and counts["matchable"] > 0
        and counts["built"] == counts["matchable"]
        and counts["stale"] == 0
        and counts["failed"] == 0
        and counts["unresolved_flags"] == 0
    )
    return {
        **counts,
        "ready_to_build": ready_to_build,
        "ready_to_match": ready_to_match,
        "unresolved": unresolved,
        "algorithm_version": CONTOUR_ALGORITHM_VERSION,
    }


def _foreground_mask(image: Image.Image) -> tuple[np.ndarray, dict[str, Any]]:
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    if gray.ndim != 2 or min(gray.shape) < 4:
        raise ContourError("Image is too small for contour extraction")
    border_width = max(1, min(5, min(gray.shape) // 20))
    border = np.concatenate(
        (
            gray[:border_width].ravel(),
            gray[-border_width:].ravel(),
            gray[:, :border_width].ravel(),
            gray[:, -border_width:].ravel(),
        )
    )
    background = float(np.median(border))
    try:
        threshold = float(threshold_otsu(gray))
    except ValueError:
        threshold = 127.5
    foreground_dark = background > 127.5
    binary = gray <= threshold if foreground_dark else gray > threshold
    labelled = label(binary, connectivity=2)
    regions = sorted(regionprops(labelled), key=lambda region: region.area, reverse=True)
    if not regions:
        raise ContourError("No foreground silhouette was found")
    main = labelled == regions[0].label
    second_fraction = float(regions[1].area / regions[0].area) if len(regions) > 1 else 0.0

    filled = binary_fill_holes(main)
    holes = filled & ~main
    hole_labels = label(holes, connectivity=2)
    max_small_hole = max(16, int(main.sum() * 0.001))
    retained_holes = np.zeros_like(main)
    large_hole_area = 0
    for region in regionprops(hole_labels):
        component = hole_labels == region.label
        if region.area <= max_small_hole:
            main[component] = True
        else:
            retained_holes[component] = True
            large_hole_area += int(region.area)
    main[retained_holes] = False

    ys, xs = np.nonzero(main)
    if len(xs) < 24:
        raise ContourError("Foreground silhouette contains too few pixels")
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    cropped = main[bbox[1] : bbox[3], bbox[0] : bbox[2]]
    return cropped, {
        "image_size": [int(gray.shape[1]), int(gray.shape[0])],
        "bbox": bbox,
        "background_gray": background,
        "threshold": threshold,
        "foreground_dark": bool(foreground_dark),
        "component_count": len(regions),
        "second_component_fraction": second_fraction,
        "large_hole_area": large_hole_area,
        "foreground_area": int(main.sum()),
    }


def _deduplicate(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return points
    keep = np.ones(len(points), dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-8
    result = points[keep]
    if len(result) > 2 and np.linalg.norm(result[0] - result[-1]) < 1e-8:
        result = result[:-1]
    return result


def _ordered_outline(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(float), 1)
    contours = find_contours(padded, 0.5, fully_connected="high")
    if not contours:
        raise ContourError("No closed sub-pixel boundary was found")
    contour = max(contours, key=len)
    points = np.column_stack((contour[:, 1] - 1.0, contour[:, 0] - 1.0))
    points = _deduplicate(points)
    if len(points) < 24:
        raise ContourError("Extracted boundary contains too few points")
    return points


def _skeleton_terminal_points(mask: np.ndarray) -> tuple[np.ndarray | None, int]:
    skeleton = skeletonize(mask)
    coords = np.column_stack(np.nonzero(skeleton))
    if len(coords) < 2:
        return None, 0
    lookup = {tuple(coord): index for index, coord in enumerate(coords)}
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    degrees = np.zeros(len(coords), dtype=int)
    for index, (row, col) in enumerate(coords):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                other = lookup.get((int(row + dr), int(col + dc)))
                if other is None:
                    continue
                rows.append(index)
                cols.append(other)
                data.append(math.sqrt(2.0) if dr and dc else 1.0)
                degrees[index] += 1
    graph = csr_matrix((data, (rows, cols)), shape=(len(coords), len(coords)))
    endpoints = np.flatnonzero(degrees == 1)
    endpoint_count = int(len(endpoints))
    candidates = endpoints if len(endpoints) >= 2 else np.arange(len(coords))
    best_pair: tuple[int, int] | None = None
    best_distance = -1.0
    for start in candidates:
        distances = dijkstra(graph, directed=False, indices=int(start))
        finite_candidates = candidates[np.isfinite(distances[candidates])]
        if not len(finite_candidates):
            continue
        end = int(finite_candidates[np.argmax(distances[finite_candidates])])
        distance = float(distances[end])
        if distance > best_distance:
            best_distance = distance
            best_pair = (int(start), end)
    if best_pair is None:
        return None, endpoint_count
    result = coords[list(best_pair)][:, ::-1].astype(float)
    return result, endpoint_count


def _pca_terminals(outline: np.ndarray) -> np.ndarray:
    centred = outline - outline.mean(axis=0)
    covariance = np.cov(centred.T)
    values, vectors = np.linalg.eigh(covariance)
    axis = vectors[:, int(np.argmax(values))]
    projection = centred @ axis
    return np.vstack((outline[int(np.argmin(projection))], outline[int(np.argmax(projection))]))


def _nearest_outline_indices(outline: np.ndarray, terminals: np.ndarray) -> tuple[int, int]:
    indices = []
    for terminal in terminals:
        indices.append(int(np.argmin(np.sum((outline - terminal) ** 2, axis=1))))
    first, second = indices
    if first == second:
        raise ContourError("Could not separate the two profile end caps")
    return (first, second) if first < second else (second, first)


def _resample_curve(points: np.ndarray, count: int) -> np.ndarray:
    points = _deduplicate(np.asarray(points, dtype=float))
    if len(points) < 2:
        raise ContourError("A curve contains fewer than two unique points")
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(distances)))
    if cumulative[-1] <= 1e-8:
        raise ContourError("A curve has zero arc length")
    target = np.linspace(0.0, cumulative[-1], int(count))
    return np.column_stack(
        (
            np.interp(target, cumulative, points[:, 0]),
            np.interp(target, cumulative, points[:, 1]),
        )
    )


def _split_boundary_roles(
    outline: np.ndarray, terminals: np.ndarray, count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    first, second = _nearest_outline_indices(outline, terminals)
    arc_a = outline[first : second + 1]
    arc_b = np.vstack((outline[second:], outline[: first + 1]))[::-1]
    if min(len(arc_a), len(arc_b)) < 8:
        raise ContourError("The two wall curves could not be separated reliably")
    dense_count = max(count * 4, 192)
    wall_a = _resample_curve(arc_a, dense_count)
    wall_b = _resample_curve(arc_b, dense_count)
    direct = np.linalg.norm(wall_a[0] - wall_b[0]) + np.linalg.norm(wall_a[-1] - wall_b[-1])
    crossed = np.linalg.norm(wall_a[0] - wall_b[-1]) + np.linalg.norm(wall_a[-1] - wall_b[0])
    if crossed < direct:
        wall_b = wall_b[::-1]

    role_qc = {
        "wall_a_source_range": [0, int(len(wall_a))],
        "wall_b_source_range": [0, int(len(wall_b))],
        "automatic_fracture_segmentation": False,
    }
    return wall_a, wall_b, np.empty((0, 2)), np.empty((0, 2)), role_qc


def _smooth_curve(points: np.ndarray, max_displacement: float, strength: float) -> tuple[np.ndarray, dict[str, float]]:
    points = np.asarray(points, dtype=float)
    step = float(np.median(np.linalg.norm(np.diff(points, axis=0), axis=1)))
    sigma = max(0.65, min(5.0, (1.25 / max(step, 0.2)) * strength))
    candidate = np.column_stack(
        (
            gaussian_filter1d(points[:, 0], sigma=sigma, mode="nearest"),
            gaussian_filter1d(points[:, 1], sigma=sigma, mode="nearest"),
        )
    )
    displacement = np.linalg.norm(candidate - points, axis=1)
    raw_max = float(displacement.max(initial=0.0))
    if raw_max > max_displacement and raw_max > 0:
        candidate = points + (candidate - points) * (max_displacement / raw_max)
        displacement = np.linalg.norm(candidate - points, axis=1)
    return candidate, {
        "sigma_samples": float(sigma),
        "unclamped_max_displacement": raw_max,
        "max_displacement": float(displacement.max(initial=0.0)),
        "median_displacement": float(np.median(displacement)),
    }


def _curvature(points: np.ndarray, sigma: float = 0.0) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    x = points[:, 0]
    y = points[:, 1]
    if sigma:
        x = gaussian_filter1d(x, sigma=sigma, mode="nearest")
        y = gaussian_filter1d(y, sigma=sigma, mode="nearest")
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    denominator = np.maximum((dx * dx + dy * dy) ** 1.5, 1e-8)
    return (dx * ddy - dy * ddx) / denominator


def _salient_indices(points: np.ndarray) -> tuple[list[int], list[list[float]]]:
    values = np.vstack([_curvature(points, sigma) for sigma in (1.0, 2.5, 5.0)])
    magnitudes = np.abs(values)
    broad = magnitudes[1:].mean(axis=0)
    threshold = float(np.quantile(broad, 0.88)) if len(broad) else 0.0
    candidates = [
        index
        for index in range(2, len(points) - 2)
        if broad[index] >= threshold
        and broad[index] >= broad[index - 1]
        and broad[index] >= broad[index + 1]
    ]
    selected: list[int] = []
    minimum_gap = max(3, len(points) // 20)
    for index in sorted(candidates, key=lambda item: broad[item], reverse=True):
        if all(abs(index - kept) >= minimum_gap for kept in selected):
            selected.append(index)
        if len(selected) >= 8:
            break
    selected.sort()
    return selected, values.T.tolist()


def _normalise_curves(
    wall_a: np.ndarray, wall_b: np.ndarray, centre: np.ndarray
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    all_points = np.vstack((wall_a, wall_b, centre))
    centroid = all_points.mean(axis=0)
    centred = all_points - centroid
    count = len(wall_a)
    wall_a_centred = centred[:count]
    wall_b_centred = centred[count : count * 2]
    centre_centred = centred[count * 2 :]
    centre_length = float(np.linalg.norm(np.diff(centre_centred, axis=0), axis=1).sum())
    if centre_length <= 1e-8:
        raise ContourError("Centreline has zero length")
    curves = {
        "wall_a": wall_a_centred / centre_length,
        "wall_b": wall_b_centred / centre_length,
        "centreline": centre_centred / centre_length,
    }
    return curves, {
        "centroid": centroid.tolist(),
        "rotation_radians": 0.0,
        "scale": centre_length,
    }


def _centreline_reliability(
    wall_a: np.ndarray, wall_b: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    """Estimate where endpoint wall pairing still describes vessel thickness.

    This inspects smoothed *measurements* of the existing walls but never
    changes either wall curve. Only contiguous unstable terminal regions are
    downweighted; interior lips and bumps therefore remain matcher evidence.
    """
    count = len(wall_a)
    if count < 8:
        return np.ones(count, dtype=float), {
            "reliable_start": 0,
            "reliable_end": count,
            "reliable_fraction": 1.0,
        }

    def tangent(points: np.ndarray) -> np.ndarray:
        measured = np.column_stack(
            (
                gaussian_filter1d(points[:, 0], sigma=1.0, mode="nearest"),
                gaussian_filter1d(points[:, 1], sigma=1.0, mode="nearest"),
            )
        )
        values = np.gradient(measured, axis=0)
        return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)

    tangent_a = tangent(wall_a)
    tangent_b = tangent(wall_b)
    agreement = np.clip(np.sum(tangent_a * tangent_b, axis=1), -1.0, 1.0)
    connector = wall_b - wall_a
    thickness = np.linalg.norm(connector, axis=1)
    connector /= np.maximum(thickness[:, None], 1e-8)
    mean_tangent = tangent_a + tangent_b
    mean_tangent /= np.maximum(
        np.linalg.norm(mean_tangent, axis=1, keepdims=True), 1e-8
    )
    orthogonality = 1.0 - np.abs(np.sum(connector * mean_tangent, axis=1))
    smooth_thickness = gaussian_filter1d(thickness, sigma=1.0, mode="nearest")
    thickness_step = np.abs(np.gradient(smooth_thickness)) / max(
        float(np.median(smooth_thickness)), 1e-8
    )
    stable = (
        (agreement >= math.cos(math.radians(55.0)))
        & (orthogonality >= 0.35)
        & (thickness_step <= 0.12)
    )
    window = min(5, max(3, count // 12))

    def first_stable(values: np.ndarray) -> int:
        for index in range(0, len(values) - window + 1):
            if int(np.count_nonzero(values[index : index + window])) >= window - 1:
                return index
        return max(0, len(values) // 4)

    left = first_stable(stable)
    right = count - first_stable(stable[::-1])
    if right - left < max(window, count // 3):
        left = min(left, count // 4)
        right = max(right, count - count // 4)

    confidence = np.ones(count, dtype=float)
    confidence[:left] = 0.0
    confidence[right:] = 0.0
    fade = min(4, max(1, (right - left) // 8))
    for offset in range(fade):
        value = float((offset + 1) / (fade + 1))
        if left + offset < right:
            confidence[left + offset] = min(confidence[left + offset], value)
        if right - 1 - offset >= left:
            confidence[right - 1 - offset] = min(
                confidence[right - 1 - offset], value
            )
    return confidence, {
        "reliable_start": int(left),
        "reliable_end": int(right),
        "reliable_fraction": float(np.mean(confidence > 0.0)),
        "fully_weighted_fraction": float(np.mean(confidence >= 0.999)),
        "terminal_trim_start": int(left),
        "terminal_trim_end": int(count - right),
        "wall_tangent_agreement_min": float(agreement.min()),
        "cross_section_orthogonality_min": float(orthogonality.min()),
        "relative_thickness_step_max": float(thickness_step.max()),
    }
def _draw_polyline(draw: ImageDraw.ImageDraw, points: np.ndarray, fill: tuple[int, ...], width: int) -> None:
    if len(points) > 1:
        draw.line([tuple(map(float, point)) for point in points], fill=fill, width=width, joint="curve")


def _draw_weighted_centreline(
    draw: ImageDraw.ImageDraw, points: np.ndarray, confidence: np.ndarray, width: int
) -> None:
    for index in range(len(points) - 1):
        reliable = min(float(confidence[index]), float(confidence[index + 1]))
        if reliable > 0.05:
            green = int(105 + 45 * reliable)
            color = (35, green, 85)
            line_width = max(2, int(round(width * (0.45 + 0.55 * reliable))))
            draw.line(
                [tuple(map(float, points[index])), tuple(map(float, points[index + 1]))],
                fill=color,
                width=line_width,
            )
        elif index % 2 == 0:
            draw.line(
                [tuple(map(float, points[index])), tuple(map(float, points[index + 1]))],
                fill=(170, 175, 185),
                width=max(2, width // 2),
            )


def _canvas_transform(curves: Iterable[np.ndarray], size: int = 1024, margin: int = 72):
    all_points = np.vstack(list(curves))
    minimum = all_points.min(axis=0)
    maximum = all_points.max(axis=0)
    extent = np.maximum(maximum - minimum, 1e-8)
    scale = min((size - 2 * margin) / extent[0], (size - 2 * margin) / extent[1])
    offset = np.array([margin, margin]) + ((size - 2 * margin) - extent * scale) / 2.0

    def transform(points: np.ndarray) -> np.ndarray:
        result = (points - minimum) * scale + offset
        return result

    return transform


def render_previews(
    artifact: dict[str, Any], preview_path: Path, overlay_path: Path
) -> None:
    raw = np.asarray(artifact["raw_outline"], dtype=float)
    curves = artifact["curves"]
    wall_a = np.asarray(curves["wall_a"], dtype=float)
    wall_b = np.asarray(curves["wall_b"], dtype=float)
    fracture_start = np.asarray(curves.get("fracture_start", []), dtype=float)
    fracture_end = np.asarray(curves.get("fracture_end", []), dtype=float)
    fracture = np.asarray(curves.get("fracture", []), dtype=float)
    centre = np.asarray(curves["centreline"], dtype=float)

    scale = 4
    size = 1024
    shown = [wall_a, wall_b]
    if len(fracture_start):
        shown.append(fracture_start)
    if len(fracture_end):
        shown.append(fracture_end)
    if len(fracture):
        shown.append(fracture)
    transform = _canvas_transform(shown, size * scale, 72 * scale)
    image = Image.new("RGB", (size * scale, size * scale), "white")
    draw = ImageDraw.Draw(image)
    _draw_polyline(draw, transform(wall_a.copy()), (20, 110, 190), 8)
    _draw_polyline(draw, transform(wall_b.copy()), (225, 90, 45), 8)
    if len(fracture_start):
        _draw_polyline(draw, transform(fracture_start.copy()), (145, 70, 185), 7)
    if len(fracture_end):
        _draw_polyline(draw, transform(fracture_end.copy()), (145, 70, 185), 7)
    if len(fracture):
        _draw_polyline(draw, transform(fracture.copy()), (145, 70, 185), 7)
    image.resize((size, size), Image.Resampling.LANCZOS).save(preview_path, format="PNG")

    normalization = artifact["normalization"]
    theta = float(normalization["rotation_radians"])
    rotation = np.array(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]],
        dtype=float,
    )
    raw_normalized = (
        (raw - np.asarray(normalization["centroid"], dtype=float)) @ rotation.T
        / max(float(normalization["scale"]), 1e-8)
    )
    overlay_transform = _canvas_transform(
        (raw_normalized, *shown), size * scale, 72 * scale
    )
    overlay = Image.new("RGB", (size * scale, size * scale), "white")
    overlay_draw = ImageDraw.Draw(overlay)
    _draw_polyline(overlay_draw, overlay_transform(raw_normalized.copy()), (175, 180, 190), 5)
    _draw_polyline(overlay_draw, overlay_transform(wall_a.copy()), (20, 110, 190), 8)
    _draw_polyline(overlay_draw, overlay_transform(wall_b.copy()), (225, 90, 45), 8)
    if len(fracture_start):
        _draw_polyline(
            overlay_draw,
            overlay_transform(fracture_start.copy()),
            (145, 70, 185),
            7,
        )
    if len(fracture_end):
        _draw_polyline(
            overlay_draw,
            overlay_transform(fracture_end.copy()),
            (145, 70, 185),
            7,
        )
    if len(fracture):
        _draw_polyline(
            overlay_draw,
            overlay_transform(fracture.copy()),
            (145, 70, 185),
            7,
        )
    overlay.resize((size, size), Image.Resampling.LANCZOS).save(overlay_path, format="PNG")


def render_clean_mask(
    artifact: dict[str, Any], path: Path, *, size: int = 1024
) -> None:
    """Render the canonical vector walls as a centred antialiased binary mask."""
    curves = artifact["curves"]
    wall_a = np.asarray(curves["wall_a"], dtype=float)
    wall_b = np.asarray(curves["wall_b"], dtype=float)
    fracture_start = np.asarray(curves.get("fracture_start", []), dtype=float)
    fracture_end = np.asarray(curves.get("fracture_end", []), dtype=float)
    scale = 4
    canvas_size = size * scale
    components = [wall_a]
    if len(fracture_end):
        components.append(fracture_end)
    components.append(wall_b[::-1])
    if len(fracture_start):
        components.append(fracture_start[::-1])
    transform = _canvas_transform(components, canvas_size, 72 * scale)
    polygon = np.vstack([transform(component.copy()) for component in components])
    image = Image.new("L", (canvas_size, canvas_size), 0)
    ImageDraw.Draw(image).polygon(
        [tuple(map(float, point)) for point in polygon],
        fill=255,
    )
    image.resize((size, size), Image.Resampling.LANCZOS).save(path, format="PNG")


def build_contour_artifact(
    image: Image.Image,
    *,
    reference_id: str,
    source_filename: str,
    source_hash: str | None = None,
    samples: int = DEFAULT_SAMPLES,
    smoothing_mode: str = "cleaned",
) -> dict[str, Any]:
    mask, extraction = _foreground_mask(image)
    outline = _ordered_outline(mask)
    terminals, skeleton_endpoints = _skeleton_terminal_points(mask)
    # Outline extremes give unambiguous shared anchors for the two boundary
    # paths. Skeleton endpoints sit inside the silhouette and can be equally
    # close to either wall, causing both wall labels to collapse onto one side.
    terminals = _pca_terminals(outline)
    (
        wall_a_dense,
        wall_b_dense,
        fracture_start_dense,
        fracture_end_dense,
        boundary_role_qc,
    ) = _split_boundary_roles(outline, terminals, samples)

    strength = 0.35 if smoothing_mode == "minimal" else 1.0
    wall_a_smooth, smooth_a = _smooth_curve(wall_a_dense, 0.75, strength)
    wall_b_smooth, smooth_b = _smooth_curve(wall_b_dense, 0.75, strength)
    wall_a = _resample_curve(wall_a_smooth, samples)
    wall_b = _resample_curve(wall_b_smooth, samples)
    centre = (wall_a + wall_b) / 2.0
    centreline_confidence, centreline_qc = _centreline_reliability(wall_a, wall_b)
    curves, transform = _normalise_curves(wall_a, wall_b, centre)
    thickness = np.linalg.norm(wall_a - wall_b, axis=1) / transform["scale"]
    reference_master = _deduplicate(
        np.vstack((wall_a_smooth[::-1], wall_b_smooth[1:]))
    )
    master_steps = np.linalg.norm(np.diff(reference_master, axis=0), axis=1)
    master_cumulative = np.concatenate(
        (np.array([0.0]), np.cumsum(master_steps))
    )
    master_length = max(float(master_cumulative[-1]), 1e-8)
    master_seam_index = len(wall_a_smooth) - 1
    master_seam_fraction = float(
        master_cumulative[master_seam_index] / master_length
    )
    reference_master_normalised = (
        reference_master - np.asarray(transform["centroid"], dtype=float)
    ) / max(float(transform["scale"]), 1e-8)

    salient: dict[str, list[int]] = {}
    curvature: dict[str, list[list[float]]] = {}
    for name in ("wall_a", "wall_b"):
        points = curves[name]
        indices, values = _salient_indices(points)
        salient[name] = indices
        curvature[name] = values

    warnings: list[str] = []
    bbox_width = extraction["bbox"][2] - extraction["bbox"][0]
    bbox_height = extraction["bbox"][3] - extraction["bbox"][1]
    if min(bbox_width, bbox_height) < 48:
        warnings.append("low_source_resolution")
    if extraction["second_component_fraction"] > 0.01:
        warnings.append("multiple_material_components")
    if extraction["large_hole_area"] > 0:
        warnings.append("large_internal_hole")
    # Small terminal forks are normal at jagged break edges.  The longest
    # geodesic path is stable through up to four endpoints; larger branch
    # counts need human review.
    if skeleton_endpoints > 4 or skeleton_endpoints < 2:
        warnings.append("ambiguous_skeleton_terminals")
    unclamped = max(
        smooth_a["unclamped_max_displacement"],
        smooth_b["unclamped_max_displacement"],
    )
    if unclamped > 0.95:
        warnings.append("smoothing_displacement_clamped")

    qc = {
        **extraction,
        "skeleton_endpoint_count": skeleton_endpoints,
        "wall_a_smoothing": smooth_a,
        "wall_b_smoothing": smooth_b,
        "max_displacement": max(smooth_a["max_displacement"], smooth_b["max_displacement"]),
        "median_displacement": float(
            np.median([smooth_a["median_displacement"], smooth_b["median_displacement"]])
        ),
        "thickness_min": float(thickness.min()),
        "thickness_median": float(np.median(thickness)),
        "thickness_max": float(thickness.max()),
        "centreline_reliability": centreline_qc,
        "boundary_roles": boundary_role_qc,
        "warnings": warnings,
    }
    return {
        "schema_version": CONTOUR_SCHEMA_VERSION,
        "algorithm_version": CONTOUR_ALGORITHM_VERSION,
        "reference_id": reference_id,
        "source_filename": source_filename,
        "source_fingerprint": source_hash,
        "smoothing_mode": smoothing_mode,
        "sample_count": int(samples),
        "raw_outline": outline.tolist(),
        "curves": {name: value.tolist() for name, value in curves.items()},
        "centreline_confidence": centreline_confidence.tolist(),
        "boundary_roles": {
            "wall_a": "vessel_wall",
            "wall_b": "vessel_wall",
            "centreline": "audit_only",
        },
        "thickness": thickness.tolist(),
        "curvature": curvature,
        "salient_indices": salient,
        "normalization": transform,
        "reference_master_boundary": {
            "source_points": reference_master.tolist(),
            "points": reference_master_normalised.tolist(),
            "nominal_seam_fraction": master_seam_fraction,
            "semantics": "continuous_fracture_to_rim_to_fracture_boundary",
            "split_before_wall_resampling": True,
        },
        "qc": qc,
        "flagged": bool(warnings),
    }


def build_manual_query_artifact(
    image: Image.Image,
    *,
    reference_id: str,
    source_filename: str,
    manual_curves: dict[str, Any],
    samples: int = DEFAULT_SAMPLES,
) -> dict[str, Any]:
    """Build query geometry from three user-authored source-pixel polylines.

    The fracture is retained for display and endpoint validation only. It is
    deliberately absent from curvature, salience, and matcher descriptors.
    """
    required = ("exterior", "interior", "fracture")
    width, height = image.size
    source: dict[str, np.ndarray] = {}
    for name in required:
        try:
            points = np.asarray(manual_curves[name], dtype=float)
        except (KeyError, TypeError, ValueError):
            raise ContourError(f"Draw the query {name} curve before preparing it")
        minimum = 8 if name != "fracture" else 3
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < minimum:
            raise ContourError(
                f"The query {name} curve needs at least {minimum} drawn points"
            )
        if not np.all(np.isfinite(points)):
            raise ContourError(f"The query {name} curve contains invalid coordinates")
        if (
            np.any(points[:, 0] < 0)
            or np.any(points[:, 0] > width)
            or np.any(points[:, 1] < 0)
            or np.any(points[:, 1] > height)
        ):
            raise ContourError(f"The query {name} curve falls outside the PNG")
        source[name] = _deduplicate(points)

    exterior = source["exterior"]
    interior = source["interior"]
    fracture = source["fracture"]
    fracture_ends = (fracture[0], fracture[-1])
    alternatives = []
    for exterior_end in (0, -1):
        for interior_end in (0, -1):
            direct = (
                np.linalg.norm(exterior[exterior_end] - fracture_ends[0])
                + np.linalg.norm(interior[interior_end] - fracture_ends[1])
            )
            crossed = (
                np.linalg.norm(exterior[exterior_end] - fracture_ends[1])
                + np.linalg.norm(interior[interior_end] - fracture_ends[0])
            )
            alternatives.append(
                (min(direct, crossed), exterior_end, interior_end, crossed < direct)
            )
    connection, exterior_end, interior_end, reverse_fracture = min(alternatives)
    diagonal = max(math.hypot(width, height), 1.0)
    if connection / 2.0 > diagonal * 0.08:
        raise ContourError(
            "The fracture line must touch the fracture-end of both wall curves"
        )
    if exterior_end == 0:
        exterior = exterior[::-1]
    if interior_end == 0:
        interior = interior[::-1]
    if reverse_fracture:
        fracture = fracture[::-1]

    wall_a = _resample_curve(exterior, samples)
    wall_b = _resample_curve(interior, samples)
    centre = (wall_a + wall_b) / 2.0
    curves, transform = _normalise_curves(wall_a, wall_b, centre)
    centroid = np.asarray(transform["centroid"], dtype=float)
    scale = max(float(transform["scale"]), 1e-8)
    curves["fracture"] = (fracture - centroid) / scale

    # Preserve the continuous fracture-to-rim-to-fracture vessel boundary.
    # Seam hypotheses must be cut from this master *before* either wall is
    # independently resampled; otherwise moving the gold point changes point
    # density and curvature descriptors even though the physical outline did
    # not change.
    supplied_master = manual_curves.get("master_boundary")
    master_boundary = None
    if supplied_master is not None:
        try:
            candidate = _deduplicate(np.asarray(supplied_master, dtype=float))
        except (TypeError, ValueError):
            candidate = np.empty((0, 2), dtype=float)
        if (
            candidate.ndim == 2
            and candidate.shape[1:] == (2,)
            and len(candidate) >= 16
            and np.all(np.isfinite(candidate))
            and np.all(candidate[:, 0] >= 0)
            and np.all(candidate[:, 0] <= width)
            and np.all(candidate[:, 1] >= 0)
            and np.all(candidate[:, 1] <= height)
        ):
            endpoints = (candidate[0], candidate[-1])
            expected = (exterior[-1], interior[-1])
            direct = np.linalg.norm(endpoints[0] - expected[0]) + np.linalg.norm(
                endpoints[1] - expected[1]
            )
            crossed = np.linalg.norm(endpoints[0] - expected[1]) + np.linalg.norm(
                endpoints[1] - expected[0]
            )
            if min(direct, crossed) / 2.0 <= diagonal * 0.04:
                master_boundary = candidate if direct <= crossed else candidate[::-1]
    if master_boundary is None:
        master_boundary = _deduplicate(
            np.vstack((exterior[::-1], interior[1:]))
        )

    cumulative = np.concatenate(
        (
            np.array([0.0]),
            np.cumsum(np.linalg.norm(np.diff(master_boundary, axis=0), axis=1)),
        )
    )
    master_length = max(float(cumulative[-1]), 1e-8)
    rim_target = (exterior[0] + interior[0]) / 2.0
    seam_index = int(
        np.argmin(np.sum((master_boundary - rim_target) ** 2, axis=1))
    )
    annotated_seam_fraction = float(cumulative[seam_index] / master_length)
    master_uniform = _resample_curve(
        master_boundary, max(256, min(1024, len(master_boundary)))
    )
    master_centroid = master_uniform.mean(axis=0)
    # Half of the complete two-wall arc is the seam-independent analogue of
    # centreline length used by reference normalization.
    master_scale = max(master_length / 2.0, 1e-8)
    normalised_master = (master_boundary - master_centroid) / master_scale
    rim_annotation = None
    if "rim_point" in manual_curves:
        rim_point = np.asarray(manual_curves["rim_point"], dtype=float)
        if rim_point.shape == (2,) and np.all(np.isfinite(rim_point)):
            rim_annotation = {
                "source_point": rim_point.tolist(),
                "normalised_point": ((rim_point - centroid) / scale).tolist(),
                "semantics": "coarse_rim_region_hint_not_hard_anchor",
                "search_half_width_fraction": 0.125,
            }
    thickness = np.linalg.norm(wall_a - wall_b, axis=1) / scale

    salient: dict[str, list[int]] = {}
    curvature: dict[str, list[list[float]]] = {}
    for name in ("wall_a", "wall_b"):
        indices, values = _salient_indices(curves[name])
        salient[name] = indices
        curvature[name] = values
    raw_outline = np.vstack((exterior, fracture, interior[::-1]))
    connection_distances = [
        float(np.linalg.norm(exterior[-1] - fracture[0])),
        float(np.linalg.norm(interior[-1] - fracture[-1])),
    ]
    return {
        "schema_version": CONTOUR_SCHEMA_VERSION,
        "algorithm_version": CONTOUR_ALGORITHM_VERSION,
        "reference_id": reference_id,
        "source_filename": source_filename,
        "source_fingerprint": None,
        "smoothing_mode": "user_drawn",
        "sample_count": int(samples),
        "raw_outline": raw_outline.tolist(),
        "curves": {name: value.tolist() for name, value in curves.items()},
        "boundary_roles": {
            "wall_a": "user_exterior_vessel_wall",
            "wall_b": "user_interior_vessel_wall",
            "fracture": "user_cutoff_not_scored",
            "centreline": "audit_only",
        },
        "thickness": thickness.tolist(),
        "curvature": curvature,
        "salient_indices": salient,
        "normalization": transform,
        "rim_annotation": rim_annotation,
        "query_master_boundary": {
            "source_points": master_boundary.tolist(),
            "points": normalised_master.tolist(),
            # Matching starts from the seam-independent midpoint and searches
            # around it. The user's gold point remains a coarse annotation and
            # never becomes the mathematical origin of the wall sampling.
            "nominal_seam_fraction": 0.5,
            "annotated_seam_fraction": annotated_seam_fraction,
            "normalization": {
                "centroid": master_centroid.tolist(),
                "scale": master_scale,
            },
            "semantics": "continuous_fracture_to_rim_to_fracture_boundary",
            "split_before_wall_resampling": True,
        },
        "qc": {
            "warnings": [],
            "manual_annotation": True,
            "fracture_scored": False,
            "fracture_connection_distances_px": connection_distances,
            "max_displacement": 0.0,
            "median_displacement": 0.0,
        },
        "flagged": False,
    }


def auto_query_wall_curves_from_fracture(
    image: Image.Image,
    manual_curves: dict[str, Any],
) -> dict[str, list[list[float]]]:
    """Infer query wall curves from the silhouette outline, fracture line, and rim split point."""
    width, height = image.size
    try:
        fracture = np.asarray(manual_curves["fracture"], dtype=float)
    except (KeyError, TypeError, ValueError):
        raise ContourError("Draw the query fracture curve before auto-tracing walls")
    if fracture.ndim != 2 or fracture.shape[1] != 2 or len(fracture) < 3:
        raise ContourError("The query fracture curve needs at least 3 drawn points")
    if not np.all(np.isfinite(fracture)):
        raise ContourError("The query fracture curve contains invalid coordinates")
    if (
        np.any(fracture[:, 0] < 0)
        or np.any(fracture[:, 0] > width)
        or np.any(fracture[:, 1] < 0)
        or np.any(fracture[:, 1] > height)
    ):
        raise ContourError("The query fracture curve falls outside the PNG")
    try:
        rim_point = np.asarray(manual_curves["rim_point"], dtype=float)
    except (KeyError, TypeError, ValueError):
        raise ContourError("Place the rim split point before auto-tracing walls")
    if rim_point.shape != (2,) or not np.all(np.isfinite(rim_point)):
        raise ContourError("The rim split point contains invalid coordinates")
    if rim_point[0] < 0 or rim_point[0] > width or rim_point[1] < 0 or rim_point[1] > height:
        raise ContourError("The rim split point falls outside the PNG")

    mask, foreground = _foreground_mask(image)
    outline = _ordered_outline(mask)
    bbox_x, bbox_y = foreground["bbox"][0], foreground["bbox"][1]
    outline = outline + np.array([bbox_x, bbox_y], dtype=float)
    rim_terminal = outline[int(np.argmin(np.sum((outline - rim_point) ** 2, axis=1)))]
    first = int(np.argmin(np.sum((outline - fracture[0]) ** 2, axis=1)))
    second = int(np.argmin(np.sum((outline - fracture[-1]) ** 2, axis=1)))
    if first == second:
        raise ContourError("The fracture endpoints touch the same outline point")

    def forward_path(start: int, end: int) -> np.ndarray:
        if start <= end:
            return outline[start : end + 1]
        return np.vstack((outline[start:], outline[: end + 1]))

    paths = [forward_path(first, second), forward_path(second, first)]
    rim_distances = [
        float(np.min(np.sum((path - rim_terminal) ** 2, axis=1))) for path in paths
    ]
    if abs(rim_distances[0] - rim_distances[1]) > 1e-8:
        wall_path = paths[int(np.argmin(rim_distances))]
    else:
        wall_path = max(paths, key=len)
    # Smooth and densify the complete physical boundary once, before using the
    # gold point to divide it. This makes every gold-point placement operate on
    # the same underlying geometry.
    smooth_count = max(DEFAULT_SAMPLES * 4, len(wall_path))
    master_boundary, _ = _smooth_curve(
        _resample_curve(wall_path, smooth_count), 0.75, 0.9
    )
    rim_position = int(
        np.argmin(np.sum((master_boundary - rim_terminal) ** 2, axis=1))
    )
    arc_a = master_boundary[: rim_position + 1]
    arc_b = master_boundary[rim_position:][::-1]
    if min(len(arc_a), len(arc_b)) < 8:
        raise ContourError("The fracture split leaves one wall curve too short")

    # Keep the left-leaning arc as exterior for ordinary profile drawings. The
    # matcher can still swap wall labels later if the assignment is wrong.
    if float(np.mean(arc_b[:, 0])) < float(np.mean(arc_a[:, 0])):
        arc_a, arc_b = arc_b, arc_a
    arc_a_smooth = arc_a
    arc_b_smooth = arc_b
    stored_master = _deduplicate(
        np.vstack((arc_a_smooth, arc_b_smooth[-2::-1]))
    )
    return {
        "exterior": _resample_curve(arc_a_smooth, DEFAULT_SAMPLES).tolist(),
        "interior": _resample_curve(arc_b_smooth, DEFAULT_SAMPLES).tolist(),
        "fracture": _deduplicate(fracture).tolist(),
        "rim_point": rim_point.tolist(),
        "master_boundary": stored_master.tolist(),
    }


def build_reference_library(
    project_path: Path,
    *,
    progress=None,
    source_pdf: str | None = None,
) -> dict[str, Any]:
    project_path = Path(project_path)
    status = library_status(project_path)
    if not status["ready_to_build"]:
        raise ContourError(
            f"{status['pending']} profile mask(s) still need review before contour cleaning"
        )
    cards_dir = project_path / "cards"
    records = read_profile_review(cards_dir).get("profiles", {})
    manifest = read_manifest(project_path)
    references = manifest["references"]
    targets = [
        card
        for card in list_card_files(cards_dir)
        if records.get(card.name, {}).get("review_status") in MATCHABLE_REVIEW_STATUSES
    ]
    if source_pdf:
        # Restrict the work queue to cards whose crop manifest points to the
        # selected PDF. Existing entries from every other PDF remain untouched.
        try:
            with open(cards_dir / "vessel_crops.json", encoding="utf-8") as handle:
                crop_document = json.load(handle)
            with open(project_path / "page_manifest.json", encoding="utf-8") as handle:
                page_document = json.load(handle)
            page_sources = {
                Path(str(page.get("image_name") or "")).stem: str(page.get("source_pdf") or "")
                for page in page_document.get("pages", []) if isinstance(page, dict)
            }
            source_cards = {
                str(crop.get("crop_file"))
                for crop in crop_document.get("vessels", [])
                if isinstance(crop, dict)
                and page_sources.get(str(crop.get("image") or "")) == source_pdf
            }
            targets = [card for card in targets if card.name in source_cards]
        except (OSError, json.JSONDecodeError):
            targets = []
    target_names = {card.name for card in targets}
    # A source-scoped build must never delete entries belonging to the other
    # PDFs in the project. Full builds may still remove genuinely stale cards.
    if not source_pdf:
        for stale_name in set(references) - target_names:
            references.pop(stale_name, None)
    built = flagged = failed = skipped_existing = 0
    for index, card in enumerate(targets):
        if progress:
            progress(index, len(targets), f"Cleaning {card.name}")
        mask_path = profile_mask_path(cards_dir, card.name, ACCEPTED_DIR)
        paths = artifact_paths(project_path, card.name)
        paths["json"].parent.mkdir(parents=True, exist_ok=True)
        source_hash = source_fingerprint(mask_path)
        prior = references.get(card.name, {})
        current_artifacts = all(
            paths[key].exists() for key in ("json", "preview", "overlay", "clean_mask")
        )
        if (
            prior.get("source_fingerprint") == source_hash
            and prior.get("algorithm_version") == CONTOUR_ALGORITHM_VERSION
            and prior.get("state") == "ready"
            and current_artifacts
        ):
            skipped_existing += 1
            if progress:
                progress(index + 1, len(targets), f"Preserved {card.name}")
            continue
        preserve_minimal = bool(
            prior.get("source_fingerprint") == source_hash
            and prior.get("review_resolution") == "minimal"
        )
        try:
            with Image.open(mask_path) as image:
                artifact = build_contour_artifact(
                    image,
                    reference_id=card_stem(card.name),
                    source_filename=card.name,
                    source_hash=source_hash,
                    smoothing_mode=(
                        "minimal" if preserve_minimal else "cleaned"
                    ),
                )
            _atomic_json(paths["json"], artifact)
            render_previews(artifact, paths["preview"], paths["overlay"])
            render_clean_mask(artifact, paths["clean_mask"])
            prior_resolution = (
                prior.get("review_resolution")
                if prior.get("source_fingerprint") == source_hash
                else None
            )
            entry = {
                "reference_id": artifact["reference_id"],
                "source_filename": card.name,
                "source_fingerprint": source_hash,
                "algorithm_version": CONTOUR_ALGORITHM_VERSION,
                "artifact": paths["json"].name,
                "preview": paths["preview"].name,
                "overlay": paths["overlay"].name,
                "clean_mask": paths["clean_mask"].name,
                "flagged": artifact["flagged"],
                "warnings": artifact["qc"]["warnings"],
                "review_resolution": (
                    prior_resolution
                    if artifact["flagged"] and prior_resolution in {"accepted", "minimal"}
                    else None if artifact["flagged"] else "accepted"
                ),
                "state": "ready",
            }
            built += 1
            if artifact["flagged"]:
                flagged += 1
        except Exception as exc:
            entry = {
                "reference_id": card_stem(card.name),
                "source_filename": card.name,
                "source_fingerprint": source_hash,
                "algorithm_version": CONTOUR_ALGORITHM_VERSION,
                "artifact": paths["json"].name,
                "flagged": True,
                "warnings": ["extraction_failed"],
                "review_resolution": None,
                "state": "failed",
                "error": str(exc),
            }
            failed += 1
        references[card.name] = entry
        write_manifest(project_path, manifest)
    if progress:
        progress(len(targets), len(targets), "Contour library complete")
    return {
        "built_now": built,
        "preserved_existing": skipped_existing,
        "flagged_now": flagged,
        "failed_now": failed,
        "source_pdf": source_pdf,
        **library_status(project_path),
    }


def flagged_contours(project_path: Path) -> list[dict[str, Any]]:
    manifest = read_manifest(project_path)
    result = []
    for filename, entry in manifest["references"].items():
        if not entry.get("flagged") or entry.get("review_resolution") in {"accepted", "minimal"}:
            continue
        item = dict(entry)
        item["filename"] = filename
        if entry.get("state") != "failed":
            artifact_path = contour_root(project_path) / entry["artifact"]
            try:
                with open(artifact_path, encoding="utf-8") as handle:
                    artifact = json.load(handle)
                item["qc"] = artifact.get("qc", {})
            except (OSError, json.JSONDecodeError):
                item["qc"] = {}
        result.append(item)
    return result


def resolve_flag(project_path: Path, reference_id: str, action: str) -> dict[str, Any]:
    if action not in {"accept", "minimal"}:
        raise ContourError("Flag action must be 'accept' or 'minimal'")
    manifest = read_manifest(project_path)
    match = next(
        (
            (filename, entry)
            for filename, entry in manifest["references"].items()
            if entry.get("reference_id") == reference_id
        ),
        None,
    )
    if not match:
        raise ContourError("Flagged reference was not found")
    filename, entry = match
    if entry.get("state") == "failed":
        raise ContourError("Fix the accepted profile mask before rebuilding this failed contour")
    if action == "minimal":
        cards_dir = Path(project_path) / "cards"
        mask_path = profile_mask_path(cards_dir, filename, ACCEPTED_DIR)
        paths = artifact_paths(project_path, filename)
        with Image.open(mask_path) as image:
            artifact = build_contour_artifact(
                image,
                reference_id=entry["reference_id"],
                source_filename=filename,
                source_hash=source_fingerprint(mask_path),
                smoothing_mode="minimal",
            )
        _atomic_json(paths["json"], artifact)
        render_previews(artifact, paths["preview"], paths["overlay"])
        render_clean_mask(artifact, paths["clean_mask"])
        entry["warnings"] = artifact["qc"]["warnings"]
        entry["clean_mask"] = paths["clean_mask"].name
    entry["review_resolution"] = "minimal" if action == "minimal" else "accepted"
    manifest["references"][filename] = entry
    write_manifest(project_path, manifest)
    return {"entry": entry, "status": library_status(project_path)}


def approve_all_flags(project_path: Path) -> dict[str, Any]:
    """Accept every unresolved, successfully built contour in one atomic save."""
    manifest = read_manifest(project_path)
    approved = 0
    skipped_failed = 0
    for entry in manifest["references"].values():
        if (
            not entry.get("flagged")
            or entry.get("review_resolution") in {"accepted", "minimal"}
        ):
            continue
        if entry.get("state") != "ready":
            skipped_failed += 1
            continue
        entry["review_resolution"] = "accepted"
        approved += 1
    if approved:
        write_manifest(project_path, manifest)
    return {
        "approved": approved,
        "skipped_failed": skipped_failed,
        "status": library_status(project_path),
    }


def load_ready_artifacts(project_path: Path) -> list[dict[str, Any]]:
    status = library_status(project_path)
    if not status["ready_to_match"]:
        raise ContourError("The canonical contour library is incomplete, stale, or has unresolved flags")
    project_path = Path(project_path)
    root = contour_root(project_path)
    records = read_profile_review(project_path / "cards").get("profiles", {})
    metadata_path = Path(project_path) / "cards" / "mask_info.csv"
    linked_stems = set()
    if metadata_path.exists():
        with open(metadata_path, newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if (str(row.get("Figure") or "").strip() and
                        str(row.get("No.") or row.get("Number") or "").strip()):
                    linked_stems.add(card_stem(row.get("mask_file") or row.get("file") or ""))
    artifacts = []
    for filename, entry in read_manifest(project_path)["references"].items():
        if records.get(filename, {}).get("review_status") not in MATCHABLE_REVIEW_STATUSES:
            continue
        if metadata_path.exists() and card_stem(filename) not in linked_stems:
            # Do not return an unlinked profile as a cryptic filename in Match
            # Query. It can remain in the contour library for later linking,
            # but it is not a usable publication reference yet.
            continue
        if entry.get("state") != "ready":
            continue
        with open(root / entry["artifact"], encoding="utf-8") as handle:
            artifacts.append(json.load(handle))
    return artifacts


def find_contours_by_citation(
    project_path: Path, figure: str, item: str
) -> list[dict[str, Any]]:
    """Find canonical contour assets using the saved Figure and No. metadata."""
    def normalise(value: Any, prefixes: tuple[str, ...]) -> str:
        text = str(value or "").strip().lower()
        for prefix in prefixes:
            if text.startswith(prefix):
                text = text[len(prefix) :].strip()
                break
        if text.endswith(".0"):
            text = text[:-2]
        return "".join(text.split())

    figure_key = normalise(figure, ("figure", "fig.", "fig"))
    item_key = normalise(item, ("item", "no.", "no"))
    if not figure_key or not item_key:
        raise ContourError("Enter both a figure number and an item number")
    metadata_path = Path(project_path) / "cards" / "mask_info.csv"
    if not metadata_path.exists():
        raise ContourError("Saved sherd metadata was not found")
    rows: dict[str, dict[str, str]] = {}
    with open(metadata_path, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if (
                normalise(row.get("Figure"), ("figure", "fig.", "fig"))
                != figure_key
                or normalise(row.get("No.") or row.get("Number"), ("item", "no.", "no"))
                != item_key
            ):
                continue
            stem = card_stem(row.get("mask_file") or row.get("file") or "")
            if stem:
                rows[stem] = row
    manifest = read_manifest(project_path)
    matches = []
    for filename, entry in manifest["references"].items():
        row = rows.get(card_stem(filename))
        if row is None:
            continue
        matches.append(
            {
                "figure": str(row.get("Figure") or "").strip(),
                "item": str(row.get("No.") or row.get("Number") or "").strip(),
                "citation_label": (
                    f"Figure {str(row.get('Figure') or '').strip()} "
                    f"Item {str(row.get('No.') or row.get('Number') or '').strip()}"
                ),
                "source_filename": filename,
                "reference_id": entry.get("reference_id", card_stem(filename)),
                "preview": entry.get("preview"),
                "overlay": entry.get("overlay"),
                "clean_mask": entry.get("clean_mask"),
                "state": entry.get("state"),
                "flagged": bool(entry.get("flagged")),
                "warnings": list(entry.get("warnings") or []),
                "review_resolution": entry.get("review_resolution"),
            }
        )
    return matches
