"""Contour-library and partial-matcher HTTP routes."""

from __future__ import annotations

import json
from pathlib import Path

from flask import jsonify, request, send_file, send_from_directory
from PIL import Image

from catalog.contours import (
    ContourError,
    auto_query_wall_curves_from_fracture,
    approve_all_flags,
    contour_root,
    find_contours_by_citation,
    flagged_contours,
    library_status,
    matcher_root,
    read_manifest,
    resolve_flag,
)
from catalog.matcher import (
    MatcherError,
    load_run,
    pot_runtime_status,
    preprocess_query,
    query_root,
    run_root,
    score_reference_by_citation,
)
from catalog.matcher_evaluation import (
    MatcherEvaluationError,
    evaluation_workbook_path,
    export_matcher_run,
)
from catalog.metadata_fusion import compare_metadata, fuse_shape_results


def register_matcher_routes(app, get_project_manager, coordinator):
    def project_path(project_id):
        manager = get_project_manager()
        project = manager.get_project(project_id)
        return manager.get_project_path(project_id) if project else None

    def status_response(project_id):
        path = project_path(project_id)
        if path is None:
            return jsonify({"success": False, "error": "Project not found"}), 404
        status = library_status(path)
        status["solver"] = pot_runtime_status()
        status["ready_to_match"] = bool(
            status.get("ready_to_match") and status["solver"]["available"]
        )
        return jsonify({"success": True, "status": status})

    def start_library_build(project_id):
        path = project_path(project_id)
        if path is None:
            return jsonify({"success": False, "error": "Project not found"}), 404
        status = library_status(path)
        if not status["ready_to_build"]:
            return jsonify(
                {
                    "success": False,
                    "error": f"{status['pending']} profile mask(s) still need review",
                    "status": status,
                }
            ), 409
        data = request.get_json(silent=True) or {}
        source_pdf = str(data.get("source_pdf") or "").strip() or None
        job = coordinator.start_library_build(project_id, path, source_pdf=source_pdf)
        return jsonify({"success": True, "job": job}), 202

    def matcher_job(project_id, job_id):
        job = coordinator.get(job_id, project_id=project_id)
        if job is None:
            return jsonify({"success": False, "error": "Matcher job not found"}), 404
        return jsonify({"success": True, "job": job})

    def list_flags(project_id):
        path = project_path(project_id)
        if path is None:
            return jsonify({"success": False, "error": "Project not found"}), 404
        flags = flagged_contours(path)
        for item in flags:
            asset_version = (
                f"{item.get('algorithm_version', '')}-"
                f"{str(item.get('source_fingerprint', ''))[:12]}"
            )
            if item.get("preview"):
                item["preview_url"] = (
                    f"/api/projects/{project_id}/matcher/assets/contours/{item['preview']}?v={asset_version}"
                )
            if item.get("overlay"):
                item["overlay_url"] = (
                    f"/api/projects/{project_id}/matcher/assets/contours/{item['overlay']}?v={asset_version}"
                )
            item["mask_url"] = (
                f"/api/projects/{project_id}/profile-mask/accepted/{item['filename']}"
            )
        return jsonify(
            {"success": True, "flags": flags, "status": library_status(path)}
        )

    def update_flag(project_id, reference_id):
        path = project_path(project_id)
        if path is None:
            return jsonify({"success": False, "error": "Project not found"}), 404
        data = request.get_json(silent=True) or {}
        result = resolve_flag(path, reference_id, str(data.get("action") or ""))
        return jsonify({"success": True, **result})

    def approve_flags(project_id):
        path = project_path(project_id)
        if path is None:
            return jsonify({"success": False, "error": "Project not found"}), 404
        result = approve_all_flags(path)
        return jsonify({"success": True, **result})

    def citation_lookup(project_id):
        path = project_path(project_id)
        if path is None:
            return jsonify({"success": False, "error": "Project not found"}), 404
        figure = str(request.args.get("figure") or "")
        item = str(request.args.get("item") or "")
        try:
            matches = find_contours_by_citation(path, figure, item)
        except ContourError as exc:
            return jsonify({"success": False, "error": str(exc)}), 400
        for match in matches:
            for field in ("preview", "overlay", "clean_mask"):
                if match.get(field):
                    match[f"{field}_url"] = (
                        f"/api/projects/{project_id}/matcher/assets/contours/"
                        f"{match[field]}"
                    )
            match["accepted_mask_url"] = (
                f"/api/projects/{project_id}/profile-mask/accepted/"
                f"{match['source_filename']}"
            )
        return jsonify({"success": True, "matches": matches})

    def upload_query(project_id):
        path = project_path(project_id)
        if path is None:
            return jsonify({"success": False, "error": "Project not found"}), 404
        upload = request.files.get("file")
        if upload is None or not upload.filename:
            return jsonify({"success": False, "error": "Choose a PNG query file"}), 400
        if Path(upload.filename).suffix.lower() != ".png":
            return jsonify({"success": False, "error": "Query input must be a PNG file"}), 400
        try:
            metadata_value = request.form.get("metadata", "{}")
            metadata = json.loads(metadata_value)
            if not isinstance(metadata, dict):
                raise ValueError
            metadata.pop("form", None)
        except (json.JSONDecodeError, ValueError):
            return jsonify({"success": False, "error": "Query metadata must be a JSON object"}), 400
        try:
            manual_curves = json.loads(request.form.get("manual_curves", "{}"))
            if not isinstance(manual_curves, dict):
                raise ValueError
        except (json.JSONDecodeError, ValueError):
            return jsonify({"success": False, "error": "Draw all three query curves"}), 400
        try:
            with Image.open(upload.stream) as source:
                if source.format != "PNG":
                    raise MatcherError("Query input must be a valid PNG image")
                image = source.convert("RGBA")
            result = preprocess_query(
                path,
                image,
                original_filename=Path(upload.filename).name,
                metadata=metadata,
                manual_curves=manual_curves,
            )
        except (ContourError, MatcherError) as exc:
            return jsonify({"success": False, "error": str(exc)}), 400
        result["preview_url"] = (
            f"/api/projects/{project_id}/matcher/assets/{result['preview']}"
        )
        result["overlay_url"] = (
            f"/api/projects/{project_id}/matcher/assets/{result['overlay']}"
        )
        result.pop("artifact", None)
        return jsonify({"success": True, "query": result})

    def auto_query_curves(project_id):
        path = project_path(project_id)
        if path is None:
            return jsonify({"success": False, "error": "Project not found"}), 404
        upload = request.files.get("file")
        if upload is None or not upload.filename:
            return jsonify({"success": False, "error": "Choose a PNG query file"}), 400
        try:
            manual_curves = json.loads(request.form.get("manual_curves", "{}"))
            if not isinstance(manual_curves, dict):
                raise ValueError
        except (json.JSONDecodeError, ValueError):
            return jsonify({"success": False, "error": "Draw the fracture curve and place the rim split point first"}), 400
        try:
            with Image.open(upload.stream) as source:
                if source.format != "PNG":
                    raise MatcherError("Query input must be a valid PNG image")
                image = source.convert("RGBA")
            curves = auto_query_wall_curves_from_fracture(image, manual_curves)
        except (ContourError, MatcherError) as exc:
            return jsonify({"success": False, "error": str(exc)}), 400
        return jsonify({"success": True, "curves": curves})

    def start_match(project_id):
        path = project_path(project_id)
        if path is None:
            return jsonify({"success": False, "error": "Project not found"}), 404
        status = library_status(path)
        solver = pot_runtime_status()
        status["solver"] = solver
        status["ready_to_match"] = bool(
            status.get("ready_to_match") and solver["available"]
        )
        if not solver["available"]:
            return jsonify(
                {
                    "success": False,
                    "error": (
                        "The POT FGW solver is unavailable in the running "
                        "SherdScope process. Install POT>=0.9.6,<0.10 and "
                        "restart SherdScope before matching."
                    ),
                    "status": status,
                }
            ), 503
        if not status["ready_to_match"]:
            return jsonify(
                {
                    "success": False,
                    "error": "Finish contour cleaning and flagged review before matching",
                    "status": status,
                }
            ), 409
        data = request.get_json(silent=True) or {}
        query_id = str(data.get("query_id") or "")
        if not (query_root(path) / query_id / "artifact.json").exists():
            return jsonify({"success": False, "error": "Upload and preprocess a query first"}), 400
        job, run_id = coordinator.start_match(project_id, path, query_id)
        return jsonify({"success": True, "job": job, "run_id": run_id}), 202

    def match_result(project_id, run_id):
        path = project_path(project_id)
        if path is None:
            return jsonify({"success": False, "error": "Project not found"}), 404
        try:
            result = load_run(path, run_id)
        except MatcherError as exc:
            job = coordinator.get(run_id, project_id=project_id)
            if job:
                return jsonify({"success": True, "job": job, "result": None}), 202
            return jsonify({"success": False, "error": str(exc)}), 404
        references = read_manifest(path).get("references", {})
        for item in result.get("results", []):
            item["diagnostic_url"] = (
                f"/api/projects/{project_id}/matcher/assets/runs/{run_id}/"
                f"{item['diagnostic']}"
            )
            decorate_result_reference(project_id, item, references)
        return jsonify({"success": True, "result": result})

    def decorate_result_reference(project_id, item, references):
        entry = references.get(item["source_filename"], {})
        clean_asset = entry.get("clean_mask") or entry.get("preview")
        if clean_asset:
            item["reference_mask_url"] = (
                f"/api/projects/{project_id}/matcher/assets/contours/"
                f"{clean_asset}"
            )
            item["reference_image_kind"] = (
                "cleaned_mask" if entry.get("clean_mask") else "cleaned_contour"
            )
        else:
            item["reference_mask_url"] = (
                f"/api/projects/{project_id}/profile-mask/accepted/"
                f"{item['source_filename']}"
            )
            item["reference_image_kind"] = "accepted_mask"

    def forced_reference_score(project_id, run_id):
        path = project_path(project_id)
        if path is None:
            return jsonify({"success": False, "error": "Project not found"}), 404
        data = request.get_json(silent=True) or {}
        try:
            result = score_reference_by_citation(
                path,
                run_id,
                figure=str(data.get("figure") or ""),
                item=str(data.get("item") or ""),
            )
        except (ContourError, MatcherError) as exc:
            return jsonify({"success": False, "error": str(exc)}), 400
        references = read_manifest(path).get("references", {})
        for item in result.get("results", []):
            item["diagnostic_url"] = (
                f"/api/projects/{project_id}/matcher/assets/runs/{run_id}/"
                f"{item['diagnostic']}"
            )
            decorate_result_reference(project_id, item, references)
        return jsonify({"success": True, "result": result})

    def export_evaluation(project_id, run_id):
        path = project_path(project_id)
        if path is None:
            return jsonify({"success": False, "error": "Project not found"}), 404
        try:
            result = export_matcher_run(path, run_id)
        except (MatcherError, MatcherEvaluationError) as exc:
            return jsonify({"success": False, "error": str(exc)}), 400
        return jsonify(
            {
                "success": True,
                "filename": result["filename"],
                "already_present": result["already_present"],
                "query_count": result["query_count"],
                "result_count": result["result_count"],
                "forced_included": result["forced_included"],
                "download_url": (
                    f"/api/projects/{project_id}/matcher/evaluation/workbook"
                ),
            }
        )

    def download_evaluation(project_id):
        path = project_path(project_id)
        if path is None:
            return jsonify({"success": False, "error": "Project not found"}), 404
        workbook = evaluation_workbook_path(path)
        if not workbook.is_file():
            return jsonify(
                {"success": False, "error": "No matcher evaluation workbook exists yet"}
            ), 404
        return send_file(
            workbook,
            mimetype=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
            as_attachment=True,
            download_name=workbook.name,
            max_age=0,
        )

    def serve_matcher_asset(project_id, filename):
        path = project_path(project_id)
        if path is None:
            return jsonify({"success": False, "error": "Project not found"}), 404
        root = matcher_root(path).resolve()
        target = (root / filename).resolve()
        try:
            target.relative_to(root)
        except ValueError:
            return jsonify({"success": False, "error": "Invalid matcher asset path"}), 400
        if not target.is_file():
            return jsonify({"success": False, "error": "Matcher asset not found"}), 404
        return send_from_directory(target.parent, target.name)

    def metadata_diagnostic(project_id):
        """Preview metadata evidence without changing a saved matcher run."""
        if project_path(project_id) is None:
            return jsonify({"success": False, "error": "Project not found"}), 404
        payload = request.get_json(silent=True) or {}
        query_metadata = payload.get("query_metadata") or {}
        reference_metadata = payload.get("reference_metadata") or {}
        if not isinstance(query_metadata, dict) or not isinstance(reference_metadata, dict):
            return jsonify({"success": False, "error": "Metadata must be JSON objects"}), 400
        results = payload.get("results")
        if results is None:
            return jsonify({
                "success": True,
                "diagnostic": compare_metadata(query_metadata, reference_metadata),
                "ranking_changed": False,
            })
        reference_index = payload.get("reference_index") or {}
        if not isinstance(results, list) or not isinstance(reference_index, dict):
            return jsonify({"success": False, "error": "Results must be a list and reference_index an object"}), 400
        return jsonify({
            "success": True,
            "results": fuse_shape_results(results, query_metadata, reference_index),
            "diagnostic_only": True,
        })

    app.add_url_rule(
        "/api/projects/<project_id>/matcher/library",
        endpoint="matcher_library_status",
        view_func=lambda project_id: status_response(project_id),
        methods=["GET"],
    )
    app.add_url_rule(
        "/api/projects/<project_id>/matcher/library/build",
        endpoint="matcher_library_build",
        view_func=lambda project_id: start_library_build(project_id),
        methods=["POST"],
    )
    app.add_url_rule(
        "/api/projects/<project_id>/matcher/jobs/<job_id>",
        endpoint="matcher_job",
        view_func=lambda project_id, job_id: matcher_job(project_id, job_id),
        methods=["GET"],
    )
    app.add_url_rule(
        "/api/projects/<project_id>/matcher/flags",
        endpoint="matcher_flags",
        view_func=lambda project_id: list_flags(project_id),
        methods=["GET"],
    )
    app.add_url_rule(
        "/api/projects/<project_id>/matcher/flags/<reference_id>",
        endpoint="matcher_flag_resolution",
        view_func=lambda project_id, reference_id: update_flag(project_id, reference_id),
        methods=["POST"],
    )
    app.add_url_rule(
        "/api/projects/<project_id>/matcher/flags/approve-all",
        endpoint="matcher_flags_approve_all",
        view_func=lambda project_id: approve_flags(project_id),
        methods=["POST"],
    )
    app.add_url_rule(
        "/api/projects/<project_id>/matcher/contours/lookup",
        endpoint="matcher_contour_lookup",
        view_func=lambda project_id: citation_lookup(project_id),
        methods=["GET"],
    )
    app.add_url_rule(
        "/api/projects/<project_id>/matcher/query",
        endpoint="matcher_query_upload",
        view_func=lambda project_id: upload_query(project_id),
        methods=["POST"],
    )
    app.add_url_rule(
        "/api/projects/<project_id>/matcher/query/auto-curves",
        endpoint="matcher_query_auto_curves",
        view_func=lambda project_id: auto_query_curves(project_id),
        methods=["POST"],
    )
    app.add_url_rule(
        "/api/projects/<project_id>/matcher/runs",
        endpoint="matcher_run_start",
        view_func=lambda project_id: start_match(project_id),
        methods=["POST"],
    )
    app.add_url_rule(
        "/api/projects/<project_id>/matcher/runs/<run_id>",
        endpoint="matcher_run_result",
        view_func=lambda project_id, run_id: match_result(project_id, run_id),
        methods=["GET"],
    )
    app.add_url_rule(
        "/api/projects/<project_id>/matcher/runs/<run_id>/reference-score",
        endpoint="matcher_forced_reference_score",
        view_func=lambda project_id, run_id: forced_reference_score(project_id, run_id),
        methods=["POST"],
    )
    app.add_url_rule(
        "/api/projects/<project_id>/matcher/runs/<run_id>/evaluation",
        endpoint="matcher_evaluation_export",
        view_func=lambda project_id, run_id: export_evaluation(project_id, run_id),
        methods=["POST"],
    )
    app.add_url_rule(
        "/api/projects/<project_id>/matcher/evaluation/workbook",
        endpoint="matcher_evaluation_workbook",
        view_func=lambda project_id: download_evaluation(project_id),
        methods=["GET"],
    )
    app.add_url_rule(
        "/api/projects/<project_id>/matcher/metadata-diagnostic",
        endpoint="matcher_metadata_diagnostic",
        view_func=lambda project_id: metadata_diagnostic(project_id),
        methods=["POST"],
    )
    app.add_url_rule(
        "/api/projects/<project_id>/matcher/assets/<path:filename>",
        endpoint="matcher_asset",
        view_func=lambda project_id, filename: serve_matcher_asset(project_id, filename),
        methods=["GET"],
    )


__all__ = ["register_matcher_routes"]
