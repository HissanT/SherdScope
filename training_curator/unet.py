"""Compact, non-destructive U-Net training and prediction for profile masks."""

from __future__ import annotations

import hashlib
import html
import json
import math
import os
import random
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_dilation, label as component_labels
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .workspace import CuratorWorkspaceError, _atomic_json, utc_now


MODEL_SCHEMA_VERSION = 1
DEFAULT_IMAGE_SIZE = 320
DEFAULT_EPOCHS = 24
DEFAULT_BATCH_SIZE = 4


class DoubleConv(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.block(value)


class ProfileUNet(nn.Module):
    """Small U-Net suitable for a 6-GB consumer GPU."""

    def __init__(self, base_channels: int = 16):
        super().__init__()
        widths = [base_channels * (2**index) for index in range(5)]
        self.encoder = nn.ModuleList(
            [DoubleConv(1, widths[0])]
            + [DoubleConv(widths[index - 1], widths[index]) for index in range(1, 5)]
        )
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ModuleList(
            [nn.ConvTranspose2d(widths[index], widths[index - 1], 2, stride=2)
             for index in range(4, 0, -1)]
        )
        self.decoder = nn.ModuleList(
            [DoubleConv(widths[index - 1] * 2, widths[index - 1])
             for index in range(4, 0, -1)]
        )
        self.output = nn.Conv2d(widths[0], 1, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        skips = []
        for index, block in enumerate(self.encoder):
            value = block(value)
            if index < len(self.encoder) - 1:
                skips.append(value)
                value = self.pool(value)
        for up, block, skip in zip(self.up, self.decoder, reversed(skips)):
            value = up(value)
            value = torch.cat((skip, value), dim=1)
            value = block(value)
        return self.output(value)


@dataclass(frozen=True)
class Letterbox:
    original_size: tuple[int, int]
    resized_size: tuple[int, int]
    offset: tuple[int, int]
    target_size: int


def _letterbox(image: Image.Image, size: int, *, mask: bool) -> tuple[np.ndarray, Letterbox]:
    image = image.convert("L")
    width, height = image.size
    scale = min(size / max(1, width), size / max(1, height))
    resized = (max(1, round(width * scale)), max(1, round(height * scale)))
    method = Image.Resampling.NEAREST if mask else Image.Resampling.BILINEAR
    transformed = image.resize(resized, method)
    offset = ((size - resized[0]) // 2, (size - resized[1]) // 2)
    fill = 0 if mask else 255
    canvas = Image.new("L", (size, size), fill)
    canvas.paste(transformed, offset)
    values = np.asarray(canvas, dtype=np.float32) / 255.0
    if mask:
        values = (values > 0.5).astype(np.float32)
    else:
        values = 1.0 - values
    return values, Letterbox((width, height), resized, offset, size)


def _restore_probability(probability: np.ndarray, box: Letterbox) -> np.ndarray:
    x, y = box.offset
    width, height = box.resized_size
    clipped = np.clip(probability[y:y + height, x:x + width], 0, 1)
    source = Image.fromarray((clipped * 255).astype(np.uint8), mode="L")
    restored = source.resize(box.original_size, Image.Resampling.BILINEAR)
    return np.asarray(restored, dtype=np.float32) / 255.0


def deterministic_split(training_ids: list[str]) -> dict[str, list[str]]:
    """Stable 80/10/10 assignment unaffected by manifest ordering."""
    groups = {"train": [], "validation": [], "test": []}
    for training_id in sorted(training_ids):
        bucket = int(hashlib.sha256(training_id.encode("utf-8")).hexdigest()[:8], 16) % 10
        group = "train" if bucket < 8 else "validation" if bucket == 8 else "test"
        groups[group].append(training_id)
    if len(training_ids) >= 10:
        for name in ("validation", "test"):
            if not groups[name]:
                donor = max(groups, key=lambda key: len(groups[key]))
                groups[name].append(groups[donor].pop())
    return groups


def _load_manifest(root: Path) -> dict[str, Any]:
    path = root / "manifest.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CuratorWorkspaceError(f"Could not read training manifest: {exc}") from exc
    if not isinstance(value.get("entries"), dict):
        raise CuratorWorkspaceError("Training manifest has no entries")
    return value


def _approved_records(root: Path, manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    output = {}
    for training_id, entry in manifest["entries"].items():
        image = root / str(entry.get("approved_image") or "")
        mask = root / str(entry.get("approved_mask") or "")
        if entry.get("decision") == "approved" and image.is_file() and mask.is_file():
            draft = root / str(entry.get("draft_mask") or "")
            output[training_id] = {
                "image": image,
                "mask": mask,
                "draft": draft if draft.is_file() else mask,
            }
    return output


class ProfileDataset(Dataset):
    def __init__(self, records: dict[str, dict[str, Path]], ids: list[str], size: int,
                 *, augment: bool = False):
        self.records = records
        self.ids = ids
        self.size = size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int):
        training_id = self.ids[index]
        record = self.records[training_id]
        with Image.open(record["image"]) as source:
            image, _ = _letterbox(source, self.size, mask=False)
        with Image.open(record["mask"]) as source:
            mask, _ = _letterbox(source, self.size, mask=True)
        if self.augment:
            gain = random.uniform(0.88, 1.12)
            bias = random.uniform(-0.04, 0.04)
            image = np.clip(image * gain + bias, 0, 1)
            if random.random() < 0.35:
                image = np.clip(image + np.random.normal(0, 0.012, image.shape), 0, 1)
        return (
            torch.from_numpy(np.ascontiguousarray(image[None])).float(),
            torch.from_numpy(np.ascontiguousarray(mask[None])).float(),
            training_id,
        )


def _dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    intersection = (probability * target).sum(dim=(1, 2, 3))
    denominator = probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1 - (2 * intersection + 1) / (denominator + 1)).mean()


def _metrics(logits: torch.Tensor, target: torch.Tensor, threshold: float) -> tuple[float, float]:
    prediction = torch.sigmoid(logits) >= threshold
    truth = target >= 0.5
    intersection = (prediction & truth).sum(dim=(1, 2, 3)).float()
    predicted = prediction.sum(dim=(1, 2, 3)).float()
    actual = truth.sum(dim=(1, 2, 3)).float()
    union = (prediction | truth).sum(dim=(1, 2, 3)).float()
    dice = ((2 * intersection + 1) / (predicted + actual + 1)).mean().item()
    iou = ((intersection + 1) / (union + 1)).mean().item()
    return dice, iou


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
              threshold: float = 0.5) -> dict[str, float]:
    model.eval()
    dice_values, iou_values = [], []
    with torch.inference_mode():
        for images, masks, _ids in loader:
            logits = model(images.to(device))
            dice, iou = _metrics(logits, masks.to(device), threshold)
            dice_values.append(dice)
            iou_values.append(iou)
    return {
        "dice": float(np.mean(dice_values)) if dice_values else 0.0,
        "iou": float(np.mean(iou_values)) if iou_values else 0.0,
    }


def _best_threshold(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, dict[str, float]]:
    candidates = [round(value, 2) for value in np.arange(0.30, 0.76, 0.05)]
    scored = [(threshold, _evaluate(model, loader, device, threshold)) for threshold in candidates]
    return max(scored, key=lambda item: item[1]["dice"])


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _atomic_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def train_unet(root: Path, *, image_size: int = DEFAULT_IMAGE_SIZE,
               epochs: int = DEFAULT_EPOCHS, batch_size: int = DEFAULT_BATCH_SIZE,
               learning_rate: float = 1e-3, seed: int = 20260731,
               patience: int = 5,
               progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    root = Path(root).resolve()
    manifest = _load_manifest(root)
    records = _approved_records(root, manifest)
    if len(records) < 30:
        raise CuratorWorkspaceError("At least 30 approved masks are required for training")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    splits = deterministic_split(list(records))
    train_set = ProfileDataset(records, splits["train"], image_size, augment=True)
    validation_set = ProfileDataset(records, splits["validation"], image_size)
    test_set = ProfileDataset(records, splits["test"], image_size)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0,
                              pin_memory=torch.cuda.is_available())
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False,
                                   num_workers=0, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0,
                             pin_memory=torch.cuda.is_available())
    device = _device()
    model = ProfileUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )
    positive = 0.0
    pixels = 0
    for _image, mask, _training_id in train_set:
        positive += float(mask.sum())
        pixels += mask.numel()
    ratio = positive / max(1, pixels)
    positive_weight = min(10.0, max(1.0, (1 - ratio) / max(ratio, 1e-6)))
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([positive_weight], device=device))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_dice = -1.0
    best_state = None
    history = []
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for images, masks, _ids in train_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(images)
                loss = 0.55 * bce(logits, masks) + 0.45 * _dice_loss(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        validation = _evaluate(model, validation_loader, device)
        scheduler.step(validation["dice"])
        row = {
            "epoch": epoch,
            "loss": round(float(np.mean(losses)), 6),
            "validation_dice": round(validation["dice"], 6),
            "validation_iou": round(validation["iou"], 6),
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        if progress:
            progress(
                f"Epoch {epoch:02d}/{epochs}: loss={row['loss']:.4f} "
                f"val Dice={row['validation_dice']:.4f} IoU={row['validation_iou']:.4f}"
            )
        if validation["dice"] > best_dice + 0.001:
            best_dice = validation["dice"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                if progress:
                    progress(f"Early stopping after epoch {epoch}; best validation Dice={best_dice:.4f}")
                break
    if best_state is None:
        raise CuratorWorkspaceError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    threshold, validation = _best_threshold(model, validation_loader, device)
    test = _evaluate(model, test_loader, device, threshold)
    model_dir = root / "models" / "unet_v1"
    checkpoint = model_dir / "best.pt"
    report = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "created_at": utc_now(),
        "approved_examples": len(records),
        "splits": splits,
        "image_size": image_size,
        "base_channels": 16,
        "batch_size": batch_size,
        "requested_epochs": epochs,
        "completed_epochs": len(history),
        "learning_rate": learning_rate,
        "positive_fraction": ratio,
        "positive_weight": positive_weight,
        "threshold": threshold,
        "validation": validation,
        "test": test,
        "history": history,
        "device": str(device),
        "torch_version": torch.__version__,
    }
    _atomic_checkpoint(checkpoint, {
        "schema_version": MODEL_SCHEMA_VERSION,
        "state_dict": best_state,
        "config": {"image_size": image_size, "base_channels": 16, "threshold": threshold},
        "training": report,
    })
    _atomic_json(model_dir / "training_report.json", report)
    preview = build_preview(root, checkpoint, splits["test"][:12])
    return {"checkpoint": checkpoint, "report": model_dir / "training_report.json",
            "preview": preview, **report}


def _load_model(checkpoint: Path) -> tuple[ProfileUNet, dict[str, Any], torch.device]:
    device = _device()
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    config = payload.get("config") or {}
    model = ProfileUNet(int(config.get("base_channels") or 16)).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, config, device


def _predict_probability(model: nn.Module, image: Image.Image, size: int,
                         device: torch.device) -> np.ndarray:
    values, box = _letterbox(image, size, mask=False)
    tensor = torch.from_numpy(values[None, None]).float().to(device)
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        probability = torch.sigmoid(model(tensor))[0, 0].float().cpu().numpy()
    return _restore_probability(probability, box)


def _anchor_prediction(prediction: np.ndarray, anchor: Image.Image,
                       threshold: float) -> np.ndarray:
    """Keep predicted components that touch the existing migrated mask.

    The migrated mask may need edge correction, but it is strong location
    evidence and prevents the network from selecting another drawing in a wide
    archaeological figure crop.
    """
    binary = np.asarray(prediction >= threshold, dtype=bool)
    anchor_image = anchor.convert("L")
    expected_size = (binary.shape[1], binary.shape[0])
    if anchor_image.size != expected_size:
        anchor_image = anchor_image.resize(expected_size, Image.Resampling.NEAREST)
    anchor_values = np.asarray(anchor_image) > 0
    if not binary.any() or not anchor_values.any():
        return binary
    labels, count = component_labels(
        binary, structure=np.ones((3, 3), dtype=np.uint8)
    )
    if not count:
        return binary
    reach = max(3, min(12, round(min(binary.shape) * 0.012)))
    support = binary_dilation(anchor_values, iterations=reach)
    touching = np.unique(labels[support & (labels > 0)])
    return np.isin(labels, touching) if touching.size else binary


def _preview_composite(image: Image.Image, truth: Image.Image,
                       predicted: np.ndarray) -> Image.Image:
    base = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    truth_values = np.asarray(truth.convert("L")) > 0
    overlay = base.astype(np.float32)
    only_truth = truth_values & ~predicted
    only_prediction = predicted & ~truth_values
    overlap = truth_values & predicted
    overlay[overlap] = overlay[overlap] * 0.45 + np.array([30, 190, 90]) * 0.55
    overlay[only_truth] = overlay[only_truth] * 0.35 + np.array([35, 105, 245]) * 0.65
    overlay[only_prediction] = overlay[only_prediction] * 0.35 + np.array([235, 70, 55]) * 0.65
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8), mode="RGB")


def build_preview(root: Path, checkpoint: Path, ids: list[str]) -> Path:
    root = Path(root).resolve()
    manifest = _load_manifest(root)
    records = _approved_records(root, manifest)
    model, config, device = _load_model(checkpoint)
    size = int(config.get("image_size") or DEFAULT_IMAGE_SIZE)
    threshold = float(config.get("threshold") or 0.5)
    preview_root = checkpoint.parent / "preview"
    preview_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, training_id in enumerate(ids[:12], start=1):
        record = records.get(training_id)
        if not record:
            continue
        with Image.open(record["image"]) as source:
            image = source.convert("RGB")
        with Image.open(record["mask"]) as source:
            truth = source.convert("L")
        with Image.open(record["draft"]) as source:
            anchor = source.convert("L")
        probability = _predict_probability(model, image, size, device)
        anchored = _anchor_prediction(probability, anchor, threshold)
        predicted = Image.fromarray(anchored.astype(np.uint8) * 255, mode="L")
        composite = _preview_composite(image, truth, anchored)
        stem = f"{index:02d}_{training_id}"
        original_path = preview_root / f"{stem}_image.png"
        truth_path = preview_root / f"{stem}_truth.png"
        predicted_path = preview_root / f"{stem}_prediction.png"
        overlay_path = preview_root / f"{stem}_overlay.png"
        image.save(original_path, format="PNG", optimize=True)
        truth.save(truth_path, format="PNG", optimize=True)
        predicted.save(predicted_path, format="PNG", optimize=True)
        composite.save(overlay_path, format="PNG", optimize=True)
        rows.append((training_id, original_path.name, truth_path.name,
                     predicted_path.name, overlay_path.name))
    cards = "\n".join(
        f"<article><h2>{html.escape(training_id)}</h2><div>"
        f"<figure><figcaption>Original</figcaption><img src='{original}'></figure>"
        f"<figure><figcaption>Ground truth</figcaption><img src='{truth}'></figure>"
        f"<figure><figcaption>Prediction</figcaption><img src='{prediction}'></figure>"
        f"<figure><figcaption>Overlay: green agreement, blue missed, red extra</figcaption><img src='{overlay}'></figure>"
        "</div></article>"
        for training_id, original, truth, prediction, overlay in rows
    )
    report_path = preview_root / "index.html"
    report_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>U-Net holdout preview</title>"
        "<style>body{font:14px system-ui;margin:22px;background:#f3f5f8;color:#172033}"
        "article{background:white;padding:14px;margin-bottom:18px;border-radius:10px}"
        "article div{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}"
        "figure{margin:0}figcaption{font-weight:700;margin-bottom:5px}"
        "img{width:100%;height:300px;object-fit:contain;border:1px solid #d8dee8}"
        "@media(max-width:1000px){article div{grid-template-columns:repeat(2,1fr)}}"
        "</style></head><body><h1>U-Net unseen holdout preview</h1>"
        f"<p>Threshold: {threshold:.2f}. These examples were not used for fitting.</p>{cards}</body></html>",
        encoding="utf-8",
    )
    return report_path


def predict_pending(root: Path, checkpoint: Path, *, force: bool = False,
                    progress: Callable[[int, int, str], None] | None = None) -> dict[str, Any]:
    root = Path(root).resolve()
    manifest = _load_manifest(root)
    model, config, device = _load_model(Path(checkpoint))
    size = int(config.get("image_size") or DEFAULT_IMAGE_SIZE)
    threshold = float(config.get("threshold") or 0.5)
    pending = [(training_id, entry) for training_id, entry in manifest["entries"].items()
               if entry.get("decision") == "pending"]
    output_root = root / "predictions" / "unet_v1"
    generated = reused = 0
    for index, (training_id, entry) in enumerate(pending, start=1):
        if progress:
            progress(index, len(pending), training_id)
        image_path = root / str(entry.get("candidate_image") or "")
        target = output_root / "masks" / f"{training_id}.png"
        if not image_path.is_file():
            continue
        if target.is_file() and not force:
            reused += 1
        else:
            with Image.open(image_path) as source:
                probability = _predict_probability(model, source.convert("RGB"), size, device)
            draft_path = root / str(entry.get("draft_mask") or "")
            if draft_path.is_file():
                with Image.open(draft_path) as source:
                    anchor = source.convert("L")
                binary = _anchor_prediction(probability, anchor, threshold)
            else:
                binary = probability >= threshold
            mask = Image.fromarray(binary.astype(np.uint8) * 255, mode="L")
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp.png")
            mask.save(temporary, format="PNG", optimize=True)
            os.replace(temporary, target)
            generated += 1
        entry["model_prediction"] = target.relative_to(root).as_posix()
        entry["model_prediction_checkpoint"] = Path(checkpoint).resolve().relative_to(root).as_posix()
        entry["model_prediction_at"] = utc_now()
    manifest["updated_at"] = utc_now()
    _atomic_json(root / "manifest.json", manifest)
    _atomic_json(output_root / "prediction_report.json", {
        "schema_version": 1,
        "created_at": utc_now(),
        "checkpoint": str(Path(checkpoint).resolve()),
        "threshold": threshold,
        "pending": len(pending),
        "generated": generated,
        "reused": reused,
    })
    return {"pending": len(pending), "generated": generated, "reused": reused,
            "output": output_root, "threshold": threshold}


__all__ = ["ProfileUNet", "deterministic_split", "train_unet", "predict_pending",
           "build_preview"]
