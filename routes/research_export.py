"""Researcher-facing CSV and dataset export routes."""

import re
from io import BytesIO
from urllib.parse import quote

from flask import jsonify, request, send_file

from research_export import (
    build_export,
    csv_bytes,
    dataset_zip_bytes,
    save_export_settings,
)


def register_research_export_routes(app, get_project_manager):
    """Register export routes while preserving their original endpoint names."""

    def export_payload(project_id, acronym):
        project_manager = get_project_manager()
        project_metadata = project_manager.get_project(project_id)
        if not project_metadata:
            raise FileNotFoundError("Project not found")
        if not acronym or not re.fullmatch(r"[A-Za-z0-9_]+", str(acronym)):
            raise ValueError(
                "Dataset acronym can only contain letters, numbers, and underscores"
            )
        project_path = project_manager.get_project_path(project_id)
        return project_metadata, build_export(project_path, str(acronym))

    def preview_project_research_export(project_id):
        """Preview approved rows and masks available for final export."""
        try:
            acronym = request.args.get("acronym", "DATA")
            _, result = export_payload(project_id, acronym)
            candidates = []
            for candidate in result["candidates"]:
                item = dict(candidate)
                item["thumbnail_url"] = (
                    f"/api/projects/{project_id}/card/"
                    + quote(candidate["mask_file"])
                )
                candidates.append(item)
            return jsonify(
                {
                    "success": True,
                    "summary": result["summary"],
                    "masks": candidates,
                    "rows": result["frame"].to_dict(orient="records"),
                    "columns": list(result["frame"].columns),
                    "unresolved": result["unresolved"],
                }
            )
        except FileNotFoundError as exc:
            return jsonify({"error": str(exc), "success": False}), 404
        except ValueError as exc:
            return jsonify({"error": str(exc), "success": False}), 400
        except Exception as exc:
            return jsonify({"error": str(exc), "success": False}), 500

    def update_project_research_export_settings(project_id):
        """Persist masks a researcher has chosen to exclude."""
        project_manager = get_project_manager()
        project_path = project_manager.get_project_path(project_id)
        if not project_path:
            return jsonify({"error": "Project not found", "success": False}), 404
        data = request.get_json(silent=True) or {}
        excluded = data.get("excluded_masks")
        if not isinstance(excluded, list) or not all(
            isinstance(item, str) for item in excluded
        ):
            return (
                jsonify(
                    {
                        "error": "excluded_masks must be a list of mask filenames",
                        "success": False,
                    }
                ),
                400,
            )
        known = data.get("known_masks")
        if known is not None and (
            not isinstance(known, list)
            or not all(isinstance(item, str) for item in known)
        ):
            return (
                jsonify(
                    {
                        "error": "known_masks must be a list of mask filenames",
                        "success": False,
                    }
                ),
                400,
            )
        settings = save_export_settings(
            project_path, excluded, known_masks=known
        )
        return jsonify({"success": True, "settings": settings})

    def download_project_research_csv(project_id):
        try:
            data = request.get_json(silent=True) or {}
            acronym = str(data.get("acronym", "")).strip()
            _, result = export_payload(project_id, acronym)
            payload = BytesIO(csv_bytes(result["frame"]))
            payload.seek(0)
            return send_file(
                payload,
                as_attachment=True,
                download_name=f"{acronym}_metadata.csv",
                mimetype="text/csv; charset=utf-8",
            )
        except FileNotFoundError as exc:
            return jsonify({"error": str(exc), "success": False}), 404
        except ValueError as exc:
            return jsonify({"error": str(exc), "success": False}), 400
        except Exception as exc:
            return jsonify({"error": str(exc), "success": False}), 500

    def export_project_results(project_id):
        """Download clean metadata, masks, dictionary, and summary."""
        try:
            data = request.get_json(silent=True) or {}
            acronym = str(data.get("acronym", "")).strip()
            project_metadata, result = export_payload(project_id, acronym)
            payload = BytesIO(
                dataset_zip_bytes(
                    result, project_metadata.get("project_name", project_id)
                )
            )
            payload.seek(0)
            return send_file(
                payload,
                as_attachment=True,
                download_name=f"{acronym}.zip",
                mimetype="application/zip",
            )
        except FileNotFoundError as exc:
            return jsonify({"error": str(exc), "success": False}), 404
        except ValueError as exc:
            return jsonify({"error": str(exc), "success": False}), 400
        except Exception as exc:
            return jsonify({"error": str(exc), "success": False}), 500

    app.add_url_rule(
        "/api/projects/<project_id>/export/preview",
        endpoint="preview_project_research_export",
        view_func=preview_project_research_export,
        methods=["GET"],
    )
    app.add_url_rule(
        "/api/projects/<project_id>/export/settings",
        endpoint="update_project_research_export_settings",
        view_func=update_project_research_export_settings,
        methods=["PATCH"],
    )
    app.add_url_rule(
        "/api/projects/<project_id>/export/csv",
        endpoint="download_project_research_csv",
        view_func=download_project_research_csv,
        methods=["POST"],
    )
    app.add_url_rule(
        "/api/projects/<project_id>/export/dataset",
        endpoint="export_project_results",
        view_func=export_project_results,
        methods=["POST"],
    )
    app.add_url_rule(
        "/api/projects/<project_id>/export",
        endpoint="export_project_results",
        view_func=export_project_results,
        methods=["POST"],
    )
