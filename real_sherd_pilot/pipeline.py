"""Preparation, U-Net inference, gold-mask processing, and evaluation."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import math
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from scipy.ndimage import distance_transform_edt


SCHEMA_VERSION = 1
DEFAULT_INPUT_SIZE = 640
DEFAULT_THRESHOLD = 0.90
DEFAULT_SMOOTHING_SIGMA = 1.0
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


class PilotError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def atomic_image(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp.png")
    image.save(temporary, format="PNG", optimize=True)
    os.replace(temporary, path)


def choose_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(checkpoint: Path, device: str):
    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:
        raise PilotError(
            "segmentation_models_pytorch is unavailable. Run this workflow with the "
            "same Python installation used by DL ArchProject (normally: python)."
        ) from exc
    model = smp.Unet(
        encoder_name="resnet50", encoder_weights=None,
        in_channels=3, classes=2, activation=None,
    )
    try:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location=device)
    try:
        model.load_state_dict(state)
    except RuntimeError as exc:
        raise PilotError(f"Checkpoint is not the expected ResNet-50 U-Net: {checkpoint}") from exc
    return model.to(device).eval()


def open_oriented(path: Path) -> Image.Image:
    try:
        with Image.open(path) as source:
            return ImageOps.exif_transpose(source).convert("RGB")
    except (OSError, ValueError) as exc:
        raise PilotError(f"Could not read image: {path}") from exc


def predict_probability(model, image: Image.Image, device: str,
                        input_size: int = DEFAULT_INPUT_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """Return model-resolution probability and RGB input without changing aspect protocol.

    The historical U-Net was trained with square resizing, so the evaluation
    deliberately preserves that preprocessing rather than silently changing
    the model at test time.
    """
    resized = image.resize((input_size, input_size), Image.Resampling.LANCZOS)
    array = np.asarray(resized, dtype=np.uint8)
    tensor = torch.from_numpy(array.copy()).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=device == "cuda"):
        logits = model(tensor)
        probability = torch.softmax(logits, dim=1)[0, 1].float().cpu().numpy()
    return probability, array


def threshold_prediction(probability: np.ndarray, threshold: float) -> np.ndarray:
    binary = (probability > threshold).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1).astype(bool)


def smooth_mask(mask: np.ndarray, sigma: float = DEFAULT_SMOOTHING_SIGMA) -> np.ndarray:
    """Apply only sub-pixel edge smoothing; do not simplify archaeological features."""
    binary = np.asarray(mask, dtype=bool)
    if not binary.any() or sigma <= 0:
        return binary.copy()
    blurred = cv2.GaussianBlur(binary.astype(np.float32), (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
    return blurred >= 0.5


def _largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if count <= 1:
        return mask.astype(bool)
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == label


def automatic_roi(full_mask: np.ndarray, original_size: tuple[int, int]) -> list[int]:
    """Derive a padded proposal from the largest full-frame prediction."""
    height, width = full_mask.shape
    selected = _largest_component(full_mask)
    ys, xs = np.nonzero(selected)
    original_width, original_height = original_size
    if not len(xs) or len(xs) < full_mask.size * 0.0002:
        return [0, 0, original_width, original_height]
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    padding = max(12, round(max(x1 - x0, y1 - y0) * 0.28))
    x0, x1 = max(0, x0 - padding), min(width, x1 + padding)
    y0, y1 = max(0, y0 - padding), min(height, y1 + padding)
    return [
        round(x0 * original_width / width), round(y0 * original_height / height),
        round(x1 * original_width / width), round(y1 * original_height / height),
    ]


def validate_roi(roi: list[Any], image_size: tuple[int, int]) -> list[int]:
    if not isinstance(roi, list) or len(roi) != 4:
        raise PilotError("Crop must contain four coordinates")
    width, height = image_size
    x0, y0, x1, y1 = [int(round(float(value))) for value in roi]
    x0, x1 = sorted((max(0, min(width, x0)), max(0, min(width, x1))))
    y0, y1 = sorted((max(0, min(height, y0)), max(0, min(height, y1))))
    if x1 - x0 < 64 or y1 - y0 < 64:
        raise PilotError("Crop is too small; draw a rectangle around the complete visible sherd")
    return [x0, y0, x1, y1]


def _save_prediction(workspace: Path, entry: dict[str, Any], image: Image.Image,
                     model, device: str, threshold: float, input_size: int) -> None:
    roi = validate_roi(entry["roi"], image.size)
    crop = image.crop(tuple(roi))
    probability, _ = predict_probability(model, crop, device, input_size)
    raw_640 = threshold_prediction(probability, threshold)
    width, height = crop.size
    probability_full = Image.fromarray(np.clip(probability * 255, 0, 255).astype(np.uint8), "L").resize(
        (width, height), Image.Resampling.BILINEAR)
    raw_full = Image.fromarray(raw_640.astype(np.uint8) * 255, "L").resize(
        (width, height), Image.Resampling.NEAREST)
    # Restore the probability field before thresholding the operational mask.
    # This avoids enlarging a 640 px binary staircase with nearest-neighbour
    # interpolation, while retaining the unmodified threshold mask above for
    # reproducibility and comparison.
    probability_full_array = np.asarray(probability_full, dtype=np.float32) / 255.0
    operational_full = (probability_full_array >= threshold).astype(np.uint8)
    operational_full = cv2.morphologyEx(
        operational_full, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8)
    ).astype(bool)
    restored_sigma = min(2.0, max(DEFAULT_SMOOTHING_SIGMA, min(width, height) / input_size))
    smooth_full = Image.fromarray(
        smooth_mask(operational_full, restored_sigma).astype(np.uint8) * 255, "L"
    )
    stem = entry["id"]
    paths = {
        "crop": workspace / "crops" / f"{stem}.png",
        "probability": workspace / "unet" / "probabilities" / f"{stem}.png",
        "unet_raw": workspace / "unet" / "raw_masks" / f"{stem}.png",
        "unet_smooth": workspace / "unet" / "smooth_masks" / f"{stem}.png",
    }
    atomic_image(paths["crop"], crop)
    atomic_image(paths["probability"], probability_full)
    atomic_image(paths["unet_raw"], raw_full)
    atomic_image(paths["unet_smooth"], smooth_full)
    for key, path in paths.items():
        entry[key] = path.relative_to(workspace).as_posix()
    entry["crop_size"] = [width, height]
    entry["prediction_updated_at"] = utc_now()
    entry["metrics"] = None
    entry["status"] = "pending"


def prepare_workspace(image_paths: list[Path], workspace: Path, checkpoint: Path,
                      *, threshold: float = DEFAULT_THRESHOLD,
                      input_size: int = DEFAULT_INPUT_SIZE,
                      force: bool = False,
                      progress: Callable[[int, int, str], None] | None = None) -> dict[str, Any]:
    workspace = Path(workspace).resolve()
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file():
        raise PilotError(f"U-Net checkpoint does not exist: {checkpoint}")
    clean_paths = [Path(path).expanduser().resolve() for path in image_paths]
    if len(clean_paths) != 5:
        raise PilotError(f"This pilot expects exactly five photographs; received {len(clean_paths)}")
    if len({path.name.lower() for path in clean_paths}) != len(clean_paths):
        raise PilotError("Photograph filenames must be unique")
    for path in clean_paths:
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise PilotError(f"Unsupported or missing photograph: {path}")

    manifest_path = workspace / "manifest.json"
    existing = {}
    if manifest_path.is_file():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = {}
    entries = existing.get("entries") if isinstance(existing.get("entries"), dict) else {}
    device = choose_device()
    model = None
    for index, source in enumerate(sorted(clean_paths, key=lambda p: p.name.lower()), 1):
        stem = source.stem
        if progress:
            progress(index, len(clean_paths), stem)
        original_target = workspace / "originals" / source.name
        original_target.parent.mkdir(parents=True, exist_ok=True)
        if force or not original_target.is_file() or sha256(original_target) != sha256(source):
            shutil.copy2(source, original_target)
        image = open_oriented(original_target)
        normalized = workspace / "normalized_photos" / f"{stem}.png"
        entry = dict(entries.get(stem) or {})
        original_hash = sha256(original_target)
        source_changed = entry.get("original_sha256") != original_hash
        if force or source_changed or not normalized.is_file():
            atomic_image(normalized, image)
        entry.update({
            "id": stem, "source_filename": source.name,
            "original": original_target.relative_to(workspace).as_posix(),
            "normalized_photo": normalized.relative_to(workspace).as_posix(),
            "original_sha256": original_hash, "original_size": list(image.size),
        })
        if force or source_changed or not entry.get("roi"):
            if model is None:
                model = load_model(checkpoint, device)
            probability, _ = predict_probability(model, image, device, input_size)
            full_mask = threshold_prediction(probability, threshold)
            entry["roi"] = automatic_roi(full_mask, image.size)
            entry["roi_source"] = "unet_full_frame_proposal"
            _save_prediction(workspace, entry, image, model, device, threshold, input_size)
        elif not (workspace / str(entry.get("crop") or "")).is_file():
            if model is None:
                model = load_model(checkpoint, device)
            _save_prediction(workspace, entry, image, model, device, threshold, input_size)
        entries[stem] = entry
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": existing.get("created_at") or utc_now(), "updated_at": utc_now(),
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "device": device, "threshold": threshold, "input_size": input_size,
        "scale_policy": "ignored_unreliable_decorative_scale",
        "segmentation_target": "visible pottery sherd; exclude glove, background, scale, and shadow",
        "entries": entries,
    }
    atomic_json(manifest_path, manifest)
    return {"workspace": workspace, "manifest": manifest_path, "device": device, "count": len(entries)}


def rerun_roi(workspace: Path, entry_id: str, roi: list[Any]) -> dict[str, Any]:
    manifest = load_manifest(workspace)
    entry = manifest["entries"].get(entry_id)
    if not entry:
        raise PilotError("Photograph not found")
    image = open_oriented(workspace / entry["original"])
    entry["roi"] = validate_roi(roi, image.size)
    entry["roi_source"] = "manual"
    model = load_model(Path(manifest["checkpoint"]), manifest.get("device") or choose_device())
    _save_prediction(workspace, entry, image, model, manifest.get("device") or choose_device(),
                     float(manifest["threshold"]), int(manifest["input_size"]))
    entry.pop("gold_raw", None); entry.pop("gold_smooth", None); entry.pop("overlay", None)
    manifest["updated_at"] = utc_now()
    atomic_json(workspace / "manifest.json", manifest)
    return entry


def load_manifest(workspace: Path) -> dict[str, Any]:
    try:
        value = json.loads((Path(workspace) / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PilotError(f"Could not read pilot manifest: {exc}") from exc
    if not isinstance(value.get("entries"), dict):
        raise PilotError("Pilot manifest has no entries")
    return value


def decode_mask(data_url: str, expected_size: tuple[int, int]) -> np.ndarray:
    encoded = data_url.split(",", 1)[1] if "," in data_url else data_url
    try:
        payload = base64.b64decode(encoded, validate=True)
        with Image.open(io.BytesIO(payload)) as source:
            mask = source.convert("L")
    except Exception as exc:
        raise PilotError("Gold mask is not a valid PNG") from exc
    if mask.size != expected_size:
        raise PilotError("Gold mask dimensions do not match the saved crop")
    return np.asarray(mask) > 32


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0


def segmentation_metrics(prediction: np.ndarray, truth: np.ndarray,
                         tolerance: int = 3) -> dict[str, float | int]:
    pred, gold = prediction.astype(bool), truth.astype(bool)
    tp = int((pred & gold).sum()); fp = int((pred & ~gold).sum())
    fn = int((~pred & gold).sum()); tn = int((~pred & ~gold).sum())
    precision = tp / max(1, tp + fp); recall = tp / max(1, tp + fn)
    dice = 2 * tp / max(1, 2 * tp + fp + fn); iou = tp / max(1, tp + fp + fn)
    pb, gb = mask_boundary(pred), mask_boundary(gold)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tolerance + 1,) * 2)
    if pb.any() and gb.any():
        p = float((pb & (cv2.dilate(gb.astype(np.uint8), kernel) > 0)).sum() / pb.sum())
        r = float((gb & (cv2.dilate(pb.astype(np.uint8), kernel) > 0)).sum() / gb.sum())
        boundary_f1 = 2 * p * r / max(1e-12, p + r)
        d_to_gold = distance_transform_edt(~gb)[pb]
        d_to_pred = distance_transform_edt(~pb)[gb]
        distances = np.concatenate((d_to_gold, d_to_pred))
        hd95 = float(np.percentile(distances, 95))
        mean_surface = float(np.mean(distances))
    else:
        boundary_f1 = 0.0; hd95 = float(math.hypot(*pred.shape)); mean_surface = hd95
    diagonal = math.hypot(*pred.shape)
    return {
        "true_positive": tp, "false_positive": fp, "false_negative": fn, "true_negative": tn,
        "precision": precision, "recall": recall, "dice": dice, "iou": iou,
        "boundary_f1_at_3px": boundary_f1, "hausdorff95_px": hd95,
        "hausdorff95_percent_diagonal": 100 * hd95 / max(1, diagonal),
        "mean_surface_distance_px": mean_surface,
        "area_error_percent_of_gold": 100 * abs(int(pred.sum()) - int(gold.sum())) / max(1, int(gold.sum())),
    }


def standardized(mask: np.ndarray, size: int = DEFAULT_INPUT_SIZE) -> np.ndarray:
    image = Image.fromarray(mask.astype(np.uint8) * 255, "L").resize((size, size), Image.Resampling.NEAREST)
    return np.asarray(image) > 0


def comparison_overlay(image: Image.Image, prediction: np.ndarray, gold: np.ndarray) -> Image.Image:
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    output = base.copy()
    overlap = prediction & gold; missed = gold & ~prediction; extra = prediction & ~gold
    output[overlap] = output[overlap] * .42 + np.array([34, 197, 94]) * .58
    output[missed] = output[missed] * .35 + np.array([37, 99, 235]) * .65
    output[extra] = output[extra] * .35 + np.array([239, 68, 68]) * .65
    return Image.fromarray(np.clip(output, 0, 255).astype(np.uint8), "RGB")


def _query_blob(mask: np.ndarray) -> Image.Image:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise PilotError("Cannot export an empty mask")
    pad = max(12, round(max(xs.max() - xs.min(), ys.max() - ys.min()) * .06))
    x0, x1 = max(0, int(xs.min()) - pad), min(mask.shape[1], int(xs.max()) + pad + 1)
    y0, y1 = max(0, int(ys.min()) - pad), min(mask.shape[0], int(ys.max()) + pad + 1)
    crop = mask[y0:y1, x0:x1]
    rgb = np.full((*crop.shape, 4), 255, dtype=np.uint8)
    rgb[crop, :3] = 0
    return Image.fromarray(rgb, "RGBA")


def save_gold(workspace: Path, entry_id: str, mask_data: str,
              *, smoothing_sigma: float = DEFAULT_SMOOTHING_SIGMA,
              outline: list[list[float]] | None = None) -> dict[str, Any]:
    workspace = Path(workspace).resolve(); manifest = load_manifest(workspace)
    entry = manifest["entries"].get(entry_id)
    if not entry:
        raise PilotError("Photograph not found")
    crop_path = workspace / entry["crop"]
    with Image.open(crop_path) as source:
        crop = source.convert("RGB")
    raw = decode_mask(mask_data, crop.size)
    if raw.sum() < 64:
        raise PilotError("Gold mask is empty or too small")
    smooth = smooth_mask(raw, max(0.0, min(3.0, float(smoothing_sigma))))
    with Image.open(workspace / entry["unet_raw"]) as source:
        unet_raw = np.asarray(source.convert("L")) > 0
    with Image.open(workspace / entry["unet_smooth"]) as source:
        unet_smooth = np.asarray(source.convert("L")) > 0
    raw_metrics = segmentation_metrics(standardized(unet_raw), standardized(smooth))
    smooth_metrics = segmentation_metrics(standardized(unet_smooth), standardized(smooth))
    paths = {
        "gold_raw": workspace / "gold" / "raw_masks" / f"{entry_id}.png",
        "gold_smooth": workspace / "gold" / "smooth_masks" / f"{entry_id}.png",
        "overlay": workspace / "comparisons" / f"{entry_id}_overlay.png",
        "manual_query": workspace / "exports" / "manual_gold_queries" / f"{entry_id}_manual.png",
        "unet_query": workspace / "exports" / "unet_queries" / f"{entry_id}_unet.png",
    }
    atomic_image(paths["gold_raw"], Image.fromarray(raw.astype(np.uint8) * 255, "L"))
    atomic_image(paths["gold_smooth"], Image.fromarray(smooth.astype(np.uint8) * 255, "L"))
    atomic_image(paths["overlay"], comparison_overlay(crop, unet_smooth, smooth))
    atomic_image(paths["manual_query"], _query_blob(smooth))
    atomic_image(paths["unet_query"], _query_blob(unet_smooth))
    for key, path in paths.items(): entry[key] = path.relative_to(workspace).as_posix()
    entry["outline"] = outline or []
    entry["smoothing_sigma_px"] = float(smoothing_sigma)
    entry["metrics"] = {"unet_raw": raw_metrics, "unet_smoothed": smooth_metrics}
    entry["status"] = "completed"; entry["completed_at"] = utc_now()
    manifest["updated_at"] = utc_now(); atomic_json(workspace / "manifest.json", manifest)
    write_metric_reports(workspace, manifest)
    return entry


def write_metric_reports(workspace: Path, manifest: dict[str, Any]) -> None:
    completed = [entry for entry in manifest["entries"].values() if entry.get("metrics")]
    metric_names = ["precision", "recall", "dice", "iou", "boundary_f1_at_3px",
                    "hausdorff95_px", "hausdorff95_percent_diagonal",
                    "mean_surface_distance_px", "area_error_percent_of_gold"]
    rows = []
    for entry in sorted(completed, key=lambda item: item["id"]):
        for variant in ("unet_raw", "unet_smoothed"):
            rows.append({"image": entry["id"], "variant": variant,
                         **{name: entry["metrics"][variant][name] for name in metric_names}})
    report_dir = workspace / "reports"; report_dir.mkdir(parents=True, exist_ok=True)
    with (report_dir / "segmentation_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "variant", *metric_names])
        writer.writeheader(); writer.writerows(rows)
    aggregate: dict[str, Any] = {"completed": len(completed), "total": len(manifest["entries"]), "variants": {}}
    for variant in ("unet_raw", "unet_smoothed"):
        variant_rows = [row for row in rows if row["variant"] == variant]
        aggregate["variants"][variant] = {
            name: {"mean": float(np.mean([row[name] for row in variant_rows])),
                   "median": float(np.median([row[name] for row in variant_rows])),
                   "std": float(np.std([row[name] for row in variant_rows]))}
            for name in metric_names
        } if variant_rows else {}
    atomic_json(report_dir / "segmentation_metrics.json", {"rows": rows, "aggregate": aggregate})
