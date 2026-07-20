"""Persistence helpers for manual vessels and graphic-scale calibrations."""

import json
import math
from pathlib import Path


VESSELS_SIDECAR_SUFFIX = "_vessels.json"
SCALE_SIDECAR_SUFFIX = "_scale.json"


def write_vessels_sidecar(masks_dir, base_filename: str, polygons: list) -> Path:
    """Persist manually drawn vessel polygons beside their source mask."""
    masks_dir = Path(masks_dir)
    sidecar = masks_dir / f"{base_filename}{VESSELS_SIDECAR_SUFFIX}"
    if polygons:
        with open(sidecar, "w") as handle:
            json.dump({"image": base_filename, "polygons": polygons}, handle)
    elif sidecar.exists():
        sidecar.unlink()
    return sidecar


def read_vessels_sidecar(masks_dir, base_filename: str) -> list:
    """Load manual vessel polygons, returning an empty list when unavailable."""
    sidecar = Path(masks_dir) / f"{base_filename}{VESSELS_SIDECAR_SUFFIX}"
    if not sidecar.exists():
        return []
    try:
        with open(sidecar) as handle:
            return json.load(handle).get("polygons", [])
    except Exception as exc:
        print(f"Error reading vessels sidecar for {base_filename}: {exc}")
        return []


def write_scale_sidecar(masks_dir, base_filename: str, scales: list) -> Path:
    """Persist scale calibration entries beside their source mask."""
    masks_dir = Path(masks_dir)
    sidecar = masks_dir / f"{base_filename}{SCALE_SIDECAR_SUFFIX}"
    if scales:
        with open(sidecar, "w") as handle:
            json.dump({"image": base_filename, "scales": scales}, handle)
    elif sidecar.exists():
        sidecar.unlink()
    return sidecar


def read_scale_sidecar(masks_dir, base_filename: str) -> list:
    """Load scale entries, returning an empty list when unavailable."""
    sidecar = Path(masks_dir) / f"{base_filename}{SCALE_SIDECAR_SUFFIX}"
    if not sidecar.exists():
        return []
    try:
        with open(sidecar) as handle:
            return json.load(handle).get("scales", [])
    except Exception as exc:
        print(f"Error reading scale sidecar for {base_filename}: {exc}")
        return []


def assign_px_per_cm(scales: list, centroid: tuple):
    """Choose the applicable valid calibration for a card centroid."""

    def px_ratio(scale):
        if scale.get("status") == "unresolved":
            return None
        dx = scale["p2"][0] - scale["p1"][0]
        dy = scale["p2"][1] - scale["p1"][1]
        distance_px = math.hypot(dx, dy)
        real_cm = scale.get("real_cm", 0)
        return (
            round(distance_px / real_cm, 4)
            if real_cm > 0 and distance_px > 0
            else None
        )

    cx, cy = centroid
    for scale in scales:
        if scale.get("status") == "unresolved":
            continue
        zone = scale.get("zone")
        if zone:
            x1, y1, x2, y2 = zone
            if min(x1, x2) <= cx <= max(x1, x2) and min(y1, y2) <= cy <= max(
                y1, y2
            ):
                ratio = px_ratio(scale)
                if ratio is not None:
                    return ratio
    for scale in scales:
        if not scale.get("zone") and scale.get("status") != "unresolved":
            return px_ratio(scale)
    return None


__all__ = [
    "SCALE_SIDECAR_SUFFIX",
    "VESSELS_SIDECAR_SUFFIX",
    "assign_px_per_cm",
    "read_scale_sidecar",
    "read_vessels_sidecar",
    "write_scale_sidecar",
    "write_vessels_sidecar",
]
