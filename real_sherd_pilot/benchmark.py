"""Fair external benchmark for real-sherd segmentation methods."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_fill_holes

from .pipeline import atomic_image, atomic_json, load_manifest, segmentation_metrics


METRIC_NAMES = (
    "precision", "recall", "dice", "iou", "boundary_f1_at_3px",
    "hausdorff95_px", "hausdorff95_percent_diagonal",
    "mean_surface_distance_px", "area_error_percent_of_gold",
)


def read_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 127


def standardized_metrics(prediction: np.ndarray, gold: np.ndarray) -> dict[str, float | int]:
    pred = cv2.resize(prediction.astype(np.uint8), (640, 640), interpolation=cv2.INTER_NEAREST) > 0
    truth = cv2.resize(gold.astype(np.uint8), (640, 640), interpolation=cv2.INTER_NEAREST) > 0
    return segmentation_metrics(pred, truth, tolerance=3)


def fill_enclosed_holes(mask: np.ndarray) -> np.ndarray:
    """Fill only enclosed background; never move the outer predicted contour."""
    return binary_fill_holes(mask).astype(bool)


def unet_prompt(mask: np.ndarray) -> tuple[list[int], list[int]]:
    """Create a reproducible SAM prompt without consulting the gold mask."""
    ys, xs = np.where(mask)
    if not len(xs):
        h, w = mask.shape
        return [0, 0, w - 1, h - 1], [w // 2, h // 2]
    box = [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    y, x = np.unravel_index(int(np.argmax(distance)), distance.shape)
    return box, [int(x), int(y)]


def sam2_masks(images: dict[str, np.ndarray], prompts: dict[str, tuple[list[int], list[int]]],
               checkpoint: Path, device: str) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    from ultralytics import SAM

    model = SAM(str(checkpoint))
    output: dict[str, np.ndarray] = {}
    timings: dict[str, float] = {}
    for name, image in images.items():
        box, point = prompts[name]
        started = time.perf_counter()
        result = model(image, bboxes=[box], points=[point], labels=[1],
                       device=0 if device == "cuda" else "cpu", verbose=False)[0]
        timings[name] = time.perf_counter() - started
        if result.masks is None or not len(result.masks.data):
            output[name] = np.zeros(image.shape[:2], dtype=bool)
        else:
            output[name] = result.masks.data[0].detach().cpu().numpy() > 0.5
    return output, timings


def load_deeplab(checkpoint: Path, device: str):
    import segmentation_models_pytorch as smp

    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = smp.DeepLabV3Plus(
        encoder_name="resnet50", encoder_weights=None, in_channels=3,
        classes=1, activation=None,
    )
    model.load_state_dict(payload["model_state"])
    return model.to(device).eval(), float(payload.get("threshold", 0.5))


def deeplab_masks(images: dict[str, np.ndarray], checkpoint: Path,
                  device: str) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    model, threshold = load_deeplab(checkpoint, device)
    output: dict[str, np.ndarray] = {}
    timings: dict[str, float] = {}
    for name, image in images.items():
        h, w = image.shape[:2]
        resized = cv2.resize(image, (640, 640), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)
        started = time.perf_counter()
        with torch.inference_mode(), torch.autocast(
            device_type=device, dtype=torch.float16, enabled=device == "cuda"
        ):
            probability = torch.sigmoid(model(tensor))[0, 0].float().cpu().numpy()
        timings[name] = time.perf_counter() - started
        probability = cv2.resize(probability, (w, h), interpolation=cv2.INTER_LINEAR)
        output[name] = probability >= threshold
    return output, timings


def comparison_overlay(image: np.ndarray, prediction: np.ndarray, gold: np.ndarray) -> Image.Image:
    display = image.copy()
    colour = np.zeros_like(display)
    colour[prediction & gold] = (25, 180, 80)
    colour[~prediction & gold] = (30, 105, 230)
    colour[prediction & ~gold] = (225, 55, 55)
    active = prediction | gold
    display[active] = (0.48 * display[active] + 0.52 * colour[active]).astype(np.uint8)
    return Image.fromarray(display)


def write_gallery(output: Path, methods: list[str], names: list[str]) -> None:
    columns = "".join(f"<th>{method}</th>" for method in methods)
    rows = []
    for name in names:
        cells = "".join(
            f'<td><img src="overlays/{method}/{name}.png"><img class="mask" '
            f'src="masks/{method}/{name}.png"></td>' for method in methods
        )
        rows.append(f"<tr><th>{name}</th>{cells}</tr>")
    html = f"""<!doctype html><meta charset="utf-8"><title>Sherd segmentation benchmark</title>
<style>body{{font:15px system-ui;margin:20px;background:#f5f7fb;color:#172033}}table{{border-collapse:collapse}}
th,td{{border:1px solid #ccd3df;padding:8px;vertical-align:top;background:white}}img{{width:280px;max-height:320px;object-fit:contain;display:block}}
img.mask{{margin-top:6px;background:#111}}.key span{{padding:2px 7px;color:white;margin-right:5px}}.g{{background:#19a957}}.b{{background:#1e69df}}.r{{background:#df3737}}</style>
<h1>Five-photo external segmentation benchmark</h1><p>No method receives information from the manual gold mask during prediction.</p>
<p class="key"><span class="g">agreement</span><span class="b">missed</span><span class="r">extra</span></p>
<table><tr><th>Photo</th>{columns}</tr>{''.join(rows)}</table>"""
    (output / "gallery.html").write_text(html, encoding="utf-8")


def write_summary(output: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Five-photo real-sherd segmentation benchmark", "",
        "The five manual masks were used only for evaluation. SAM prompts were derived from the U-Net prediction, not from gold.", "",
        "| Method | Mean Dice | Mean IoU | Mean boundary F1 | Mean HD95 (px) |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, metrics in result["aggregate"].items():
        lines.append(
            f"| {method} | {metrics['dice']['mean']:.3f} | {metrics['iou']['mean']:.3f} | "
            f"{metrics['boundary_f1_at_3px']['mean']:.3f} | {metrics['hausdorff95_px']['mean']:.1f} |"
        )
    lines += [
        "", "The enclosed-hole operation fills only background regions completely surrounded by the predicted sherd. "
        "It does not dilate, erode, or otherwise move the outside contour.", "",
        "These five photographs are an external pilot, not a sufficient test set for a publication-level model comparison.",
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_benchmark(workspace: Path, output: Path, *, sam_checkpoint: Path | None = None,
                  deeplab_checkpoint: Path | None = None) -> dict[str, Any]:
    workspace, output = Path(workspace).resolve(), Path(output).resolve()
    manifest = load_manifest(workspace)
    entries = [manifest["entries"][key] for key in sorted(manifest["entries"])]
    incomplete = [entry["id"] for entry in entries if not entry.get("gold_smooth")]
    if incomplete:
        raise RuntimeError(f"Manual gold masks are missing: {', '.join(incomplete)}")
    images = {entry["id"]: np.asarray(Image.open(workspace / entry["crop"]).convert("RGB")) for entry in entries}
    gold = {entry["id"]: read_mask(workspace / entry["gold_smooth"]) for entry in entries}
    baseline = {entry["id"]: read_mask(workspace / entry["unet_smooth"]) for entry in entries}
    predictions: dict[str, dict[str, np.ndarray]] = {
        "unet_original": baseline,
        "unet_holes_filled": {name: fill_enclosed_holes(mask) for name, mask in baseline.items()},
    }
    timings: dict[str, dict[str, float]] = {
        "unet_original": {name: 0.0 for name in images},
        "unet_holes_filled": {name: 0.0 for name in images},
    }
    prompts = {name: unet_prompt(mask) for name, mask in baseline.items()}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if sam_checkpoint:
        predictions["sam2_unet_prompt"], timings["sam2_unet_prompt"] = sam2_masks(
            images, prompts, Path(sam_checkpoint), device)
    if deeplab_checkpoint:
        raw, timings["deeplab"] = deeplab_masks(images, Path(deeplab_checkpoint), device)
        predictions["deeplab"] = raw
        predictions["deeplab_holes_filled"] = {
            name: fill_enclosed_holes(mask) for name, mask in raw.items()
        }
        timings["deeplab_holes_filled"] = {name: 0.0 for name in images}

    rows: list[dict[str, Any]] = []
    for method, method_predictions in predictions.items():
        for name, prediction in method_predictions.items():
            mask_path = output / "masks" / method / f"{name}.png"
            overlay_path = output / "overlays" / method / f"{name}.png"
            atomic_image(mask_path, Image.fromarray(prediction.astype(np.uint8) * 255, "L"))
            atomic_image(overlay_path, comparison_overlay(images[name], prediction, gold[name]))
            metrics = standardized_metrics(prediction, gold[name])
            rows.append({"image": name, "method": method, "seconds": timings[method][name],
                         **{metric: metrics[metric] for metric in METRIC_NAMES}})
    output.mkdir(parents=True, exist_ok=True)
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    aggregate = {}
    for method in predictions:
        selected = [row for row in rows if row["method"] == method]
        aggregate[method] = {
            metric: {"mean": float(np.mean([row[metric] for row in selected])),
                     "median": float(np.median([row[metric] for row in selected])),
                     "std": float(np.std([row[metric] for row in selected]))}
            for metric in ("dice", "iou", "boundary_f1_at_3px", "hausdorff95_px",
                           "mean_surface_distance_px", "area_error_percent_of_gold")
        }
        aggregate[method]["mean_seconds"] = float(np.mean([row["seconds"] for row in selected]))
    result = {
        "protocol": {
            "gold_used_for_prediction": False,
            "metric_size": [640, 640],
            "boundary_tolerance_px": 3,
            "sam_prompt": "U-Net mask bounding box plus deepest U-Net foreground point",
            "hole_rule": "fill enclosed background only; outer contour unchanged",
        },
        "prompts": {name: {"box": box, "positive_point": point}
                    for name, (box, point) in prompts.items()},
        "rows": rows, "aggregate": aggregate,
    }
    atomic_json(output / "metrics.json", result)
    write_gallery(output, list(predictions), list(images))
    write_summary(output, result)
    return result
