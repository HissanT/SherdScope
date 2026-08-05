"""Local-only browser curator for a prepared profile-training workspace."""

from __future__ import annotations

import base64
import io
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from flask import Flask, jsonify, render_template, request, send_file
from PIL import Image
from scipy.ndimage import (
    binary_dilation,
    binary_fill_holes,
    distance_transform_edt,
    label as component_labels,
)

from catalog.profile_segmentation import _ink_mask, propose_profile_mask
from .workspace import CuratorWorkspaceError, _atomic_json, _sha256, utc_now


def _load_manifest(root: Path) -> dict[str, Any]:
    path = root / "manifest.json"
    try:
        with open(path, encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise CuratorWorkspaceError(f"Could not read curator manifest: {exc}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("entries"), dict):
        raise CuratorWorkspaceError("Curator manifest is invalid")
    return value


def _safe_child(root: Path, relative: str) -> Path:
    root = root.resolve()
    target = (root / str(relative)).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise CuratorWorkspaceError("Invalid workspace asset path") from exc
    return target


def _atomic_mask(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp.png")
    try:
        image.save(temporary, format="PNG", optimize=True)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass


def _atomic_snapshot(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copy2(source, temporary)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass


def _decode_mask(mask_data: str, expected_size: tuple[int, int]) -> Image.Image:
    if not isinstance(mask_data, str) or not mask_data:
        raise CuratorWorkspaceError("Edited mask data is required")
    encoded = mask_data.split(",", 1)[1] if "," in mask_data else mask_data
    try:
        payload = base64.b64decode(encoded, validate=True)
        with Image.open(io.BytesIO(payload)) as source:
            mask = source.convert("L")
    except Exception as exc:
        raise CuratorWorkspaceError("Edited mask is not a valid PNG") from exc
    if mask.size != expected_size:
        raise CuratorWorkspaceError(
            "Edited mask dimensions do not match the candidate image"
        )
    return mask.point(lambda value: 255 if value > 32 else 0, mode="L")


def _remove_isolated_specks(mask: Image.Image) -> tuple[Image.Image, int, int]:
    """Remove microscopic disconnected components without eroding real edges."""
    foreground = np.asarray(mask.convert("L")) > 0
    if not foreground.any():
        return mask.convert("L"), 0, 0
    labels, count = component_labels(
        foreground, structure=np.ones((3, 3), dtype=np.uint8)
    )
    sizes = np.bincount(labels.ravel())
    threshold = max(2, int(round(foreground.size * 0.000003)))
    largest = int(np.argmax(sizes[1:]) + 1) if count else 0
    keep = sizes > threshold
    keep[0] = False
    if largest:
        keep[largest] = True
    cleaned = keep[labels]
    removed = int(foreground.sum() - cleaned.sum())
    return Image.fromarray(cleaned.astype(np.uint8) * 255, mode="L"), removed, threshold


def _encode_mask(mask: Image.Image) -> str:
    output = io.BytesIO()
    mask.save(output, format="PNG", optimize=True)
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _guided_profile_recovery(image: Image.Image, current: np.ndarray) -> np.ndarray:
    """Recover thick ink connected to the reviewed mask, excluding hairlines."""
    loose_ink, _threshold = _ink_mask(image, loose=True)
    if not loose_ink.any():
        return np.zeros_like(current, dtype=bool)
    thickness = distance_transform_edt(loose_ink)
    # At the curator's 600-DPI working resolution, construction strokes are
    # typically below this radius while fracture/profile masses are not.
    thick_core = loose_ink & (thickness >= 6.0)
    labels, count = component_labels(
        thick_core, structure=np.ones((3, 3), dtype=np.uint8)
    )
    if not count:
        return np.zeros_like(current, dtype=bool)
    contact = binary_dilation(current, iterations=3)
    touching_labels = np.unique(labels[contact & (labels > 0)])
    if touching_labels.size == 0:
        return np.zeros_like(current, dtype=bool)
    selected_core = np.isin(labels, touching_labels)
    # Restore only the immediate outline around thick ink. Long construction
    # lines have no thick core and therefore cannot grow into the result.
    restored = loose_ink & binary_dilation(selected_core, iterations=5)
    return np.asarray(binary_fill_holes(restored), dtype=bool)


def create_curator_app(workspace: Path) -> Flask:
    workspace = Path(workspace).resolve()
    template_folder = Path(__file__).parent / "templates"
    static_folder = Path(__file__).parent / "static"
    app = Flask(
        __name__,
        template_folder=str(template_folder),
        static_folder=str(static_folder),
        static_url_path="/static",
    )
    app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024

    @app.after_request
    def no_store(response):
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    def index():
        return render_template("index.html", dataset_name=workspace.name)

    @app.get("/api/entries")
    def entries():
        manifest = _load_manifest(workspace)
        output = []
        for training_id, entry in manifest["entries"].items():
            output.append(
                {
                    "training_id": training_id,
                    "crop_file": entry.get("crop_file"),
                    "decision": entry.get("decision", "pending"),
                    "review_note": entry.get("review_note", ""),
                    "original_review_status": entry.get("original_review_status"),
                    "source_pdf": entry.get("source_pdf"),
                    "pdf_page_index": entry.get("pdf_page_index"),
                    "source_render_dpi": entry.get("source_render_dpi"),
                    "training_render_dpi": entry.get("training_render_dpi"),
                    "image_width": entry.get("image_width"),
                    "image_height": entry.get("image_height"),
                    "image_url": f"/api/asset/image/{training_id}",
                    "mask_url": f"/api/asset/mask/{training_id}",
                    "prediction_url": (
                        f"/api/asset/prediction/{training_id}"
                        if entry.get("model_prediction") else None
                    ),
                }
            )
        output.sort(key=lambda item: item["training_id"].lower())
        counts = {
            decision: sum(1 for item in output if item["decision"] == decision)
            for decision in ("pending", "approved", "rejected")
        }
        return jsonify(
            {
                "success": True,
                "entries": output,
                "counts": counts,
                "total": len(output),
                "dataset_name": manifest.get("dataset_name", workspace.name),
            }
        )

    @app.get("/api/asset/<kind>/<training_id>")
    def asset(kind: str, training_id: str):
        manifest = _load_manifest(workspace)
        entry = manifest["entries"].get(training_id)
        if not isinstance(entry, dict):
            return jsonify({"success": False, "error": "Candidate not found"}), 404
        if kind == "image":
            relative = entry.get("candidate_image")
        elif kind == "mask":
            relative = entry.get("approved_mask") or entry.get("draft_mask")
        elif kind == "prediction":
            relative = entry.get("model_prediction")
        else:
            return jsonify({"success": False, "error": "Unknown asset type"}), 400
        try:
            path = _safe_child(workspace, relative)
        except CuratorWorkspaceError as exc:
            return jsonify({"success": False, "error": str(exc)}), 400
        if not path.is_file():
            return jsonify({"success": False, "error": "Asset not found"}), 404
        return send_file(path, mimetype="image/png", max_age=0)

    def persist_mask(training_id: str, *, approve: bool):
        manifest = _load_manifest(workspace)
        entry = manifest["entries"].get(training_id)
        if not isinstance(entry, dict):
            raise CuratorWorkspaceError("Candidate not found")
        image_path = _safe_child(workspace, entry.get("candidate_image"))
        if not image_path.is_file():
            raise CuratorWorkspaceError("Candidate image is missing")
        with Image.open(image_path) as image:
            expected_size = image.size
        data = request.get_json(silent=True) or {}
        mask = _decode_mask(str(data.get("mask_data") or ""), expected_size)
        removed_pixels = 0
        cleanup_threshold = 0
        if approve:
            mask, removed_pixels, cleanup_threshold = _remove_isolated_specks(mask)
        folder = "approved/masks" if approve else "candidates/masks"
        target = workspace / folder / f"{training_id}.png"
        _atomic_mask(target, mask)
        if approve:
            approved_image = workspace / "approved" / "images" / f"{training_id}.png"
            _atomic_snapshot(image_path, approved_image)
            entry["approved_image"] = approved_image.relative_to(workspace).as_posix()
            entry["approved_mask"] = target.relative_to(workspace).as_posix()
            entry["approved_image_sha256"] = _sha256(approved_image)
            entry["decision"] = "approved"
            entry["decision_at"] = utc_now()
        else:
            entry["draft_mask"] = target.relative_to(workspace).as_posix()
        entry["review_note"] = str(data.get("review_note") or "")
        entry["approved_mask_sha256" if approve else "draft_mask_sha256"] = _sha256(
            target
        )
        if approve:
            entry["approval_cleanup"] = {
                "method": "isolated_components_only",
                "removed_pixels": removed_pixels,
                "component_pixel_threshold": cleanup_threshold,
            }
        manifest["updated_at"] = utc_now()
        _atomic_json(workspace / "manifest.json", manifest)
        return entry

    @app.post("/api/entries/<training_id>/draft")
    def save_draft(training_id: str):
        try:
            entry = persist_mask(training_id, approve=False)
            return jsonify({"success": True, "entry": entry})
        except CuratorWorkspaceError as exc:
            return jsonify({"success": False, "error": str(exc)}), 400

    @app.post("/api/entries/<training_id>/approve")
    def approve(training_id: str):
        try:
            entry = persist_mask(training_id, approve=True)
            return jsonify({"success": True, "entry": entry})
        except CuratorWorkspaceError as exc:
            return jsonify({"success": False, "error": str(exc)}), 400

    @app.post("/api/entries/<training_id>/recover")
    def recover(training_id: str):
        """Return a high-DPI recovery proposal without persisting it."""
        try:
            manifest = _load_manifest(workspace)
            entry = manifest["entries"].get(training_id)
            if not isinstance(entry, dict):
                raise CuratorWorkspaceError("Candidate not found")
            image_path = _safe_child(workspace, entry.get("candidate_image"))
            if not image_path.is_file():
                raise CuratorWorkspaceError("Candidate image is missing")
            data = request.get_json(silent=True) or {}
            with Image.open(image_path) as source:
                image = source.convert("RGB")
            current_image = _decode_mask(
                str(data.get("mask_data") or ""), image.size
            )
            current = np.asarray(current_image) > 0
            proposed_raw, evidence = propose_profile_mask(image)
            proposed = np.asarray(proposed_raw) > 0
            guided = _guided_profile_recovery(image, current)
            # A current reviewed mask is the safest seed: use only thick ink
            # connected to it. A global proposal is reserved for an empty mask
            # because it may select the opposite outline of a full vessel.
            recovery_proposal = guided if current.any() else proposed
            if not recovery_proposal.any():
                raise CuratorWorkspaceError("No high-resolution profile ink was found")
            if current.any():
                overlap = int((current & recovery_proposal).sum())
                agreement = overlap / max(
                    1, min(int(current.sum()), int(recovery_proposal.sum()))
                )
                if agreement < 0.45:
                    raise CuratorWorkspaceError(
                        "The high-resolution proposal does not agree with this mask enough to recover it safely"
                    )
                recovered = current | recovery_proposal
            else:
                agreement = 1.0
                recovered = recovery_proposal
            added_pixels = int((recovered & ~current).sum())
            result = Image.fromarray(recovered.astype(np.uint8) * 255, mode="L")
            return jsonify(
                {
                    "success": True,
                    "mask_data": _encode_mask(result),
                    "added_pixels": added_pixels,
                    "agreement": round(float(agreement), 4),
                    "proposal_confidence": round(float(evidence.confidence), 4),
                    "proposal_reasons": evidence.reasons,
                }
            )
        except CuratorWorkspaceError as exc:
            return jsonify({"success": False, "error": str(exc)}), 400

    @app.post("/api/entries/<training_id>/reject")
    def reject(training_id: str):
        manifest = _load_manifest(workspace)
        entry = manifest["entries"].get(training_id)
        if not isinstance(entry, dict):
            return jsonify({"success": False, "error": "Candidate not found"}), 404
        data = request.get_json(silent=True) or {}
        entry["decision"] = "rejected"
        entry["decision_at"] = utc_now()
        entry["review_note"] = str(data.get("review_note") or "")
        manifest["updated_at"] = utc_now()
        _atomic_json(workspace / "manifest.json", manifest)
        return jsonify({"success": True, "entry": entry})

    return app
