"""Local-only review UI for real-sherd gold masks and U-Net comparison."""

from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file

from .pipeline import PilotError, load_manifest, rerun_roi, save_gold


def _safe_asset(root: Path, relative: str) -> Path:
    target = (root / str(relative)).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise PilotError("Invalid pilot asset path") from exc
    return target


def create_app(workspace: Path) -> Flask:
    workspace = Path(workspace).resolve()
    package = Path(__file__).parent
    app = Flask(__name__, template_folder=str(package / "templates"),
                static_folder=str(package / "static"), static_url_path="/pilot-static")
    app.config["MAX_CONTENT_LENGTH"] = 96 * 1024 * 1024

    @app.after_request
    def no_cache(response):
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    def index():
        return render_template("index.html", workspace_name=workspace.name)

    @app.get("/api/entries")
    def entries():
        manifest = load_manifest(workspace)
        output = []
        for entry in sorted(manifest["entries"].values(), key=lambda value: value["id"]):
            item = {
                "id": entry["id"], "filename": entry["source_filename"],
                "status": entry.get("status", "pending"), "roi": entry["roi"],
                "roi_source": entry.get("roi_source"), "original_size": entry["original_size"],
                "crop_size": entry.get("crop_size"), "outline": entry.get("outline", []),
                "smoothing_sigma_px": entry.get("smoothing_sigma_px", 1.0),
                "metrics": entry.get("metrics"),
                "urls": {
                    "original": f"/api/asset/{entry['id']}/original",
                    "crop": f"/api/asset/{entry['id']}/crop",
                    "unet_raw": f"/api/asset/{entry['id']}/unet_raw",
                    "unet_smooth": f"/api/asset/{entry['id']}/unet_smooth",
                    "gold": f"/api/asset/{entry['id']}/gold_smooth" if entry.get("gold_smooth") else None,
                    "overlay": f"/api/asset/{entry['id']}/overlay" if entry.get("overlay") else None,
                },
            }
            output.append(item)
        return jsonify(success=True, entries=output, device=manifest.get("device"),
                       threshold=manifest.get("threshold"),
                       completed=sum(item["status"] == "completed" for item in output))

    @app.get("/api/asset/<entry_id>/<kind>")
    def asset(entry_id: str, kind: str):
        manifest = load_manifest(workspace); entry = manifest["entries"].get(entry_id)
        if not entry or kind not in {"original", "crop", "unet_raw", "unet_smooth", "gold_raw", "gold_smooth", "overlay"}:
            return jsonify(success=False, error="Asset not found"), 404
        relative = entry.get(kind)
        if not relative:
            return jsonify(success=False, error="Asset not found"), 404
        try:
            path = _safe_asset(workspace, relative)
        except PilotError as exc:
            return jsonify(success=False, error=str(exc)), 400
        if not path.is_file():
            return jsonify(success=False, error="Asset not found"), 404
        return send_file(path, mimetype="image/png" if path.suffix.lower() == ".png" else "image/jpeg", max_age=0)

    @app.post("/api/entries/<entry_id>/roi")
    def roi(entry_id: str):
        try:
            data = request.get_json(silent=True) or {}
            entry = rerun_roi(workspace, entry_id, data.get("roi"))
            return jsonify(success=True, roi=entry["roi"], crop_size=entry["crop_size"])
        except PilotError as exc:
            return jsonify(success=False, error=str(exc)), 400

    @app.post("/api/entries/<entry_id>/gold")
    def gold(entry_id: str):
        try:
            data = request.get_json(silent=True) or {}
            entry = save_gold(workspace, entry_id, str(data.get("mask_data") or ""),
                              smoothing_sigma=float(data.get("smoothing_sigma", 1.0)),
                              outline=data.get("outline") or [])
            return jsonify(success=True, metrics=entry["metrics"], status=entry["status"])
        except (PilotError, TypeError, ValueError) as exc:
            return jsonify(success=False, error=str(exc)), 400

    return app

