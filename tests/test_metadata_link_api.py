import os
import io
import json
import threading
import time
import zipfile

import pandas as pd
import pytest
from PIL import Image

pytest.importorskip("ultralytics")
pytest.importorskip("fitz")
os.environ["PYPOTTERYLENS_SKIP_INIT"] = "1"

import app as app_module
from catalog.linkage import (
    Hesban11Profile, apply_reviewer_row_overrides, load_linkage_state,
    save_linkage_state, validate_figure,
)
from services.projects import ProjectManager
from catalog.export import EXPORT_COLUMNS


def test_vessel_box_review_api_adds_boxes_without_allowing_id_rebinding(
        tmp_path, monkeypatch):
    manager = ProjectManager(tmp_path / "projects")
    project = manager.create_project("Box API Test")
    project_id = project["project_id"]
    image_path = manager.get_project_path(project_id, "images") / "page.png"
    Image.new("RGB", (320, 240), "white").save(image_path)
    monkeypatch.setattr(app_module, "project_manager", manager)
    client = app_module.app.test_client()

    initial = client.get(f"/api/projects/{project_id}/vessel-boxes/page")
    assert initial.status_code == 200
    assert initial.get_json()["detections"] == []
    created = client.put(f"/api/projects/{project_id}/vessel-boxes/page", json={
        "detections": [{"reviewed_bbox": [20, 30, 120, 180], "approved": True}],
    })
    assert created.status_code == 200
    item = created.get_json()["detections"][0]
    assert item["vessel_id"] == "page_mask_layer_0"
    assert item["approved"] is True

    mismatch = client.put(f"/api/projects/{project_id}/vessel-boxes/page", json={
        "detections": [{
            **item, "vessel_id": "page_mask_layer_999",
            "reviewed_bbox": [25, 35, 125, 185],
        }],
    })
    assert mismatch.status_code == 200
    saved = mismatch.get_json()["detections"][0]
    assert saved["detection_id"] == item["detection_id"]
    assert saved["vessel_id"] == "page_mask_layer_0"
    assert saved["reviewed_bbox"] == [25, 35, 125, 185]


def test_state_edit_and_apply_endpoints(tmp_path, monkeypatch):
    manager = ProjectManager(tmp_path / "projects")
    project = manager.create_project("API Test")
    project_id = project["project_id"]
    project_path = manager.get_project_path(project_id)
    pd.DataFrame([{
        "file": "page_0", "mask_file": "page_0_mask_layer_0", "Notes": "keep me"
    }]).to_csv(project_path / "cards" / "mask_info.csv", index=False)

    figure = {
        "figure_id": "2.1", "figure_caption": "Figure 2.1", "review_status": "pending",
        "drawing_pages": [{"printed_page": "19", "source_pdf": "hesban.pdf"}],
        "table_pages": [{"printed_page": "20", "source_pdf": "hesban.pdf"}],
        "drawings": [{
            "mask_file": "page_0_mask_layer_0", "fingerprint": "abc", "vessel_number": "1"
        }],
        "table_rows": [{"table_no": "1", "table_type": "Pithos"}],
    }
    validate_figure(figure, Hesban11Profile())
    save_linkage_state(project_path, {
        "schema_version": 1, "profile": "hesban11", "status": "complete",
        "progress": {}, "figures": [figure], "warnings": [],
    })
    monkeypatch.setattr(app_module, "project_manager", manager)
    monkeypatch.setattr(app_module, "get_local_ocr_health", lambda **_: {
        "available": False,
        "message": "Local OCR is installed but could not start.",
        "error": "broken optional dependency",
        "models_initialized": False,
    })
    client = app_module.app.test_client()

    state_response = client.get(f"/api/projects/{project_id}/metadata-link/state")
    assert state_response.status_code == 200
    state_payload = state_response.get_json()
    assert state_payload["state"]["figures"][0]["status"] == "ready"
    assert "table_rows" not in state_payload["state"]["figures"][0]
    assert state_payload["profile"]["columns"][2]["csv_label"] == "Sq/Area"
    assert state_payload["ocr_available"] is False
    assert state_payload["ocr_health"]["error"] == "broken optional dependency"
    assert state_payload["backend_reload"]["required"] is False
    assert state_payload["stale_figure_count"] == 0
    detail = client.get(
        f"/api/projects/{project_id}/metadata-link/figures/2.1").get_json()["figure"]
    assert detail["table_rows"][0]["table_type"] == "Pithos"

    edit_response = client.patch(
        f"/api/projects/{project_id}/metadata-link/figures/2.1",
        json={"drawing_numbers": {"page_0_mask_layer_0": "01"}},
    )
    assert edit_response.status_code == 200
    assert edit_response.get_json()["figure"]["status"] == "ready"
    assert edit_response.get_json()["figure"]["review_history"][-1]["action"] == "edited"

    apply_response = client.post(
        f"/api/projects/{project_id}/metadata-link/apply", json={"figure_ids": ["2.1"]})
    assert apply_response.status_code == 200
    assert apply_response.get_json()["applied_rows"] == 1
    result = pd.read_csv(project_path / "cards" / "mask_info.csv", dtype=str, keep_default_na=False)
    assert result.loc[0, "Notes"] == "keep me"
    assert result.loc[0, "Type"] == "Pithos"
    assert result.loc[0, "Link Status"] == "approved"


def test_state_summary_deduplicates_repeated_stable_figure_keys(
        tmp_path, monkeypatch):
    manager = ProjectManager(tmp_path / "projects")
    project = manager.create_project("Duplicate Summary Test")
    project_id = project["project_id"]
    project_path = manager.get_project_path(project_id)
    figure = {
        "figure_id": "3.86",
        "figure_key": "stable-386",
        "figure_caption": "Figure 3.86",
        "drawing_pages": [],
        "table_pages": [],
        "drawings": [],
        "table_rows": [],
        "warnings": [],
        "review_status": "pending",
        "processing_status": "reviewable",
    }
    save_linkage_state(project_path, {
        "schema_version": 2,
        "profile": "hesban11",
        "status": "complete",
        "progress": {},
        "figures": [dict(figure), dict(figure)],
        "warnings": [],
    })
    monkeypatch.setattr(app_module, "project_manager", manager)
    monkeypatch.setattr(app_module, "get_local_ocr_health", lambda **_: {
        "available": True, "message": "Ready", "error": "",
        "models_initialized": True,
    })

    response = app_module.app.test_client().get(
        f"/api/projects/{project_id}/metadata-link/state")

    assert response.status_code == 200
    figures = response.get_json()["state"]["figures"]
    assert [(item["figure_id"], item["figure_key"]) for item in figures] == [
        ("3.86", "stable-386")
    ]


def _make_api_project(tmp_path, name="API Extra"):
    manager = ProjectManager(tmp_path / "projects")
    project = manager.create_project(name)
    return manager, project["project_id"], manager.get_project_path(project["project_id"])


def test_backend_reload_status_detects_source_changed_after_start(tmp_path, monkeypatch):
    source = tmp_path / "ocr.py"
    source.write_text("before", encoding="utf-8")
    started = source.stat().st_mtime_ns
    monkeypatch.setattr(app_module, "_BACKEND_SOURCE_SNAPSHOT", {source: started})
    os.utime(source, ns=(started + 1_000_000, started + 1_000_000))

    status = app_module.get_backend_reload_status()

    assert status["required"] is True
    assert status["changed_files"] == ["ocr.py"]
    assert "Restart SherdScope" in status["message"]


def _wait_linkage_job(project_id, project_path, job_id, timeout=5):
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = app_module.linkage_job_coordinator.snapshot(
            project_id, project_path, ensure_worker=False)
        completed = next((item for item in state.get("recent", [])
                          if item.get("id") == job_id), None)
        if completed:
            assert completed["status"] == "succeeded", completed.get("error")
            return completed
        time.sleep(.01)
    raise AssertionError(f"job {job_id} did not finish")


def test_edit_rejects_cross_pdf_table_page_and_allows_figure_correction(tmp_path, monkeypatch):
    manager, project_id, project_path = _make_api_project(tmp_path)
    state = {
        "schema_version": 1, "profile": "hesban11", "status": "complete",
        "progress": {}, "warnings": [], "figures": [{
            "figure_id": "2.1", "figure_caption": "Figure 2.1", "review_status": "pending",
            "drawing_pages": [{"image_name": "a_page_0.jpg", "source_pdf": "a.pdf"}],
            "table_pages": [],
            "drawings": [{"mask_file": "a", "fingerprint": "x", "vessel_number": "1"}],
            "table_rows": [{"table_no": "1", "table_type": "Bowl"}],
        }],
    }
    validate_figure(state["figures"][0], Hesban11Profile())
    save_linkage_state(project_path, state)
    (project_path / "page_manifest.json").write_text(json.dumps({
        "schema_version": 1, "profile": "hesban11", "pages": [
            {"image_name": "a_page_0.jpg", "source_pdf": "a.pdf", "logical_index": 0},
            {"image_name": "b_page_0.jpg", "source_pdf": "b.pdf", "logical_index": 1},
        ]}), encoding="utf-8")
    monkeypatch.setattr(app_module, "project_manager", manager)
    client = app_module.app.test_client()
    rejected = client.patch(
        f"/api/projects/{project_id}/metadata-link/figures/2.1",
        json={"table_pages": ["b_page_0.jpg"]})
    assert rejected.status_code == 400
    corrected = client.patch(
        f"/api/projects/{project_id}/metadata-link/figures/2.1",
        json={"figure_id": "Figure 2.01"})
    assert corrected.status_code == 200
    assert corrected.get_json()["figure"]["figure_id"] == "2.01"


def test_run_endpoint_prevents_concurrent_project_job(tmp_path, monkeypatch):
    manager, project_id, project_path = _make_api_project(tmp_path)
    release = threading.Event()

    class BlockingLinker:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, **kwargs):
            release.wait(5)
            return {"figures": [], "status": "complete"}

    monkeypatch.setattr(app_module, "project_manager", manager)
    monkeypatch.setattr(app_module, "MetadataLinker", BlockingLinker)
    client = app_module.app.test_client()
    first = client.post(f"/api/projects/{project_id}/metadata-link/run",
                        json={"backend": "local"})
    assert first.status_code == 202
    second = client.post(f"/api/projects/{project_id}/metadata-link/run",
                         json={"backend": "local"})
    assert second.status_code == 202
    assert second.get_json()["job"]["id"] == first.get_json()["job"]["id"]
    release.set()
    _wait_linkage_job(project_id, project_path, first.get_json()["job"]["id"])


def test_failed_renderer_does_not_overwrite_existing_pdf_or_images(tmp_path, monkeypatch):
    manager, project_id, project_path = _make_api_project(tmp_path)
    source = project_path / "pdf_source" / "book.pdf"
    source.write_bytes(b"original pdf")
    image = project_path / "images" / "API_Extra_page_0.jpg"
    image.write_bytes(b"original image")

    class FailedProcessor:
        def process_pdf_to_folder(self, *args, **kwargs):
            return "Error processing PDF: corrupt input"

    monkeypatch.setattr(app_module, "project_manager", manager)
    monkeypatch.setattr(app_module, "pdf_processor", FailedProcessor())
    client = app_module.app.test_client()
    response = client.post("/api/pdf/upload", data={
        "project_id": project_id, "render_dpi": "400", "split_pages": "false",
        "file": (io.BytesIO(b"not a pdf"), "book.pdf"),
    }, content_type="multipart/form-data")
    assert response.status_code == 500
    assert source.read_bytes() == b"original pdf"
    assert image.read_bytes() == b"original image"


def test_pdf_upload_always_uses_400_dpi_even_if_form_requests_another_value(
        tmp_path, monkeypatch):
    manager, project_id, _ = _make_api_project(tmp_path, "Fixed DPI")
    observed = {}

    class RecordingProcessor:
        def process_pdf_to_folder(
                self, _pdf, output, _split, project_name=None, render_dpi=None):
            from PIL import Image
            observed["processor_dpi"] = render_dpi
            Image.new("RGB", (20, 20), "white").save(
                os.path.join(output, f"{project_name}_page_0.jpg"))
            return "PDF processed successfully"

    def fake_record(_project, _pdf, base, _split, profile_slug=None, render_dpi=None):
        observed["manifest_dpi"] = render_dpi
        return {"pages": [{"image_name": f"{base}_page_0.jpg", "render_dpi": render_dpi}]}

    monkeypatch.setattr(app_module, "project_manager", manager)
    monkeypatch.setattr(app_module, "pdf_processor", RecordingProcessor())
    monkeypatch.setattr(app_module, "record_pdf_pages", fake_record)
    response = app_module.app.test_client().post("/api/pdf/upload", data={
        "project_id": project_id, "render_dpi": "600", "split_pages": "false",
        "file": (io.BytesIO(b"pdf content"), "book.pdf"),
    }, content_type="multipart/form-data")

    assert response.status_code == 200
    assert response.get_json()["render_dpi"] == 400
    assert observed == {"processor_dpi": 400, "manifest_dpi": 400}


def test_same_dpi_reupload_with_changed_content_is_blocked_when_cards_exist(tmp_path, monkeypatch):
    manager, project_id, project_path = _make_api_project(tmp_path, "Guard")
    (project_path / "pdf_source" / "book.pdf").write_bytes(b"old content")
    (project_path / "cards" / "card.png").write_bytes(b"downstream")
    (project_path / "page_manifest.json").write_text(json.dumps({
        "schema_version": 1, "default_render_dpi": 400, "pages": [{
            "image_name": "Guard_page_0.jpg", "source_pdf": "book.pdf",
            "render_dpi": 400, "split_part": None, "logical_index": 0,
        }]}), encoding="utf-8")
    monkeypatch.setattr(app_module, "project_manager", manager)
    client = app_module.app.test_client()
    response = client.post("/api/pdf/upload", data={
        "project_id": project_id, "render_dpi": "400", "split_pages": "false",
        "file": (io.BytesIO(b"different content"), "book.pdf"),
    }, content_type="multipart/form-data")
    assert response.status_code == 409
    assert (project_path / "pdf_source" / "book.pdf").read_bytes() == b"old content"


def test_approved_linkage_columns_reach_project_zip_export(tmp_path, monkeypatch):
    manager, project_id, project_path = _make_api_project(tmp_path, "Export")
    card_name = "page_0_mask_layer_0.png"
    from PIL import Image
    Image.new("RGB", (20, 20), "white").save(project_path / "cards" / card_name)
    pd.DataFrame([{
        "file": "page_0", "mask_file": "page_0_mask_layer_0",
        "figure_id": "2.1", "vessel_number": "1", "table_type": "Bowl",
        "nonplastics_size": "7A\n6A", "link_status": "approved",
    }]).to_csv(project_path / "cards" / "mask_info.csv", index=False)
    monkeypatch.setattr(app_module, "project_manager", manager)
    client = app_module.app.test_client()
    response = client.post(f"/api/projects/{project_id}/export", json={"acronym": "HSB"})
    assert response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(response.data)) as archive:
        exported = archive.read("metadata.csv").decode("utf-8-sig")
        assert "data_dictionary.csv" in archive.namelist()
        assert "export_summary.txt" in archive.namelist()
        assert "images/HSB_Fig2-1_No1.png" in archive.namelist()
    assert "Figure" in exported
    assert "Non-Plastics - Size" in exported
    assert "2.1" in exported
    assert '"7A\n6A"' in exported


def test_clean_export_preview_selection_csv_and_zip(tmp_path, monkeypatch):
    manager, project_id, project_path = _make_api_project(tmp_path, "Clean Export")
    from PIL import Image
    for name in ("page_mask_layer_0.png", "page_mask_layer_1.png"):
        Image.new("RGB", (24, 24), "white").save(project_path / "cards" / name)
    pd.DataFrame([
        {"mask_file": "page_mask_layer_0", "Figure": "2.1", "No.": "1",
         "Type": "Cooking pot", "Fabric Color - Exterior": "5YR 7/3\nPink",
         "Decor": "**", "Link Status": "approved", "bbox_x1": "99"},
        {"mask_file": "page_mask_layer_1", "Figure": "2.1", "No.": "2",
         "Type": "Bowl", "Link Status": "approved_with_overrides", "bbox_x1": "100"},
    ]).to_csv(project_path / "cards" / "mask_info.csv", index=False)
    save_linkage_state(project_path, {"profile": "hesban11", "status": "complete",
        "figures": [
            {"figure_id": "2.1", "review_status": "approved", "warnings": []},
            {"figure_id": "2.2", "review_status": "pending", "warnings": [{
                "message": "A table row is missing", "blocking": True}]},
        ]})
    monkeypatch.setattr(app_module, "project_manager", manager)
    client = app_module.app.test_client()

    preview = client.get(f"/api/projects/{project_id}/export/preview?acronym=HES")
    assert preview.status_code == 200
    payload = preview.get_json()
    assert payload["columns"] == EXPORT_COLUMNS
    assert payload["summary"]["approved_vessels"] == 2
    assert payload["summary"]["unresolved_figures"] == 1
    assert len(payload["masks"]) == 2

    settings = client.patch(f"/api/projects/{project_id}/export/settings",
                            json={"excluded_masks": ["page_mask_layer_1"],
                                  "known_masks": ["page_mask_layer_0", "page_mask_layer_1"]})
    assert settings.status_code == 200
    csv_response = client.post(f"/api/projects/{project_id}/export/csv",
                               json={"acronym": "HES"})
    assert csv_response.status_code == 200
    assert csv_response.data.startswith(b"\xef\xbb\xbf")
    exported = pd.read_csv(io.BytesIO(csv_response.data), dtype=str, keep_default_na=False)
    assert list(exported.columns) == EXPORT_COLUMNS
    assert not ({"Figure Caption", "Diameter Source", "Drawing Page", "Table Pages",
                 "Source PDF", "Link Status"} & set(exported.columns))
    assert len(exported) == 1
    assert exported.loc[0, "Image Filename"] == "HES_Fig2-1_No1.png"
    assert exported.loc[0, "Fabric Color - Exterior"] == "5YR 7/3\nPink"
    assert "bbox_x1" not in exported.columns
    assert "mask_file" not in exported.columns

    dataset = client.post(f"/api/projects/{project_id}/export/dataset",
                          json={"acronym": "HES"})
    assert dataset.status_code == 200
    with zipfile.ZipFile(io.BytesIO(dataset.data)) as archive:
        assert sorted(archive.namelist()) == [
            "data_dictionary.csv", "export_summary.txt",
            "images/HES_Fig2-1_No1.png", "metadata.csv"]
        zipped_csv = archive.read("metadata.csv")
    assert zipped_csv == csv_response.data

    streamed = client.get(
        f"/api/projects/{project_id}/export/dataset?acronym=HES")
    assert streamed.status_code == 200
    assert streamed.content_type == "application/zip"
    assert streamed.data.startswith(b"PK")
    with zipfile.ZipFile(io.BytesIO(streamed.data)) as archive:
        assert archive.read("metadata.csv") == csv_response.data

    prepared = client.post(
        f"/api/projects/{project_id}/export/dataset/prepare",
        json={"acronym": "HES"},
    )
    assert prepared.status_code == 200
    prepared_payload = prepared.get_json()
    assert prepared_payload["size"] > 0
    prepared_download = client.get(prepared_payload["download_url"])
    assert prepared_download.status_code == 200
    assert prepared_download.content_type == "application/zip"
    assert int(prepared_download.headers["Content-Length"]) == len(
        prepared_download.data)
    assert prepared_download.headers["Cache-Control"] == "no-store"
    with zipfile.ZipFile(io.BytesIO(prepared_download.data)) as archive:
        assert archive.testzip() is None
        assert archive.read("metadata.csv") == csv_response.data
    transfer = client.get(prepared_payload["transfer_url"])
    assert transfer.status_code == 200
    assert transfer.data == prepared_download.data
    assert "Content-Disposition" not in transfer.headers


def test_export_setting_edit_preserves_hidden_legacy_exclusions(tmp_path, monkeypatch):
    manager, project_id, project_path = _make_api_project(tmp_path, "Legacy Export")
    (project_path / "export_settings.json").write_text(json.dumps({
        "excluded_masks": ["approved_mask", "not_approved_yet"],
        "legacy_exclusions_imported": True,
    }), encoding="utf-8")
    monkeypatch.setattr(app_module, "project_manager", manager)
    client = app_module.app.test_client()

    response = client.patch(f"/api/projects/{project_id}/export/settings", json={
        "excluded_masks": [], "known_masks": ["approved_mask"],
    })

    assert response.status_code == 200
    assert response.get_json()["settings"]["excluded_masks"] == ["not_approved_yet"]


def test_per_figure_rerun_updates_rows_boundaries_and_evidence(tmp_path, monkeypatch):
    manager, project_id, project_path = _make_api_project(tmp_path, "Rerun")
    from PIL import Image
    Image.new("RGB", (1000, 900), "white").save(project_path / "images" / "page_1.jpg")
    (project_path / "page_manifest.json").write_text(json.dumps({
        "schema_version": 1, "profile": "hesban11", "pages": [{
            "image_name": "page_1.jpg", "source_pdf": "hesban.pdf",
            "logical_index": 1, "printed_page": "20", "page_text": "Figure 2.1",
        }],
    }), encoding="utf-8")
    figure = {
        "figure_id": "2.1", "figure_caption": "Figure 2.1", "review_status": "pending",
        "drawing_pages": [{"image_name": "page_0.jpg", "source_pdf": "hesban.pdf"}],
        "table_pages": [{"image_name": "page_1.jpg", "source_pdf": "hesban.pdf",
                         "logical_index": 1, "printed_page": "20", "crop": None}],
        "drawings": [{"mask_file": "page_0_mask_layer_0", "fingerprint": "x",
                      "vessel_number": "1"}],
        "table_rows": [],
    }
    validate_figure(figure, Hesban11Profile())
    save_linkage_state(project_path, {
        "schema_version": 1, "profile": "hesban11", "status": "complete",
        "progress": {}, "figures": [figure], "warnings": [],
    })

    observed_processing_states = []

    class FakeRerunOCR:
        def extract_table(self, *args, **kwargs):
            observed_processing_states.append(
                load_linkage_state(project_path)["figures"][0]["processing_status"])
            return {
                "is_table": True, "printed_page": "20",
                "rows": [{"table_no": "1", "table_type": "Pithos"}],
                "boundary": {
                    "table_bounds": [100, 100, 900, 760],
                    "upper_header_rule": 100, "lower_header_rule": 145,
                    "data_start_y": 150, "data_end_y": 760,
                    "closing_rule_y": 760, "header_confirmed": True,
                    "has_closing_rule": True, "continues": False,
                    "column_bounds": [[100, 200], [200, 900]],
                    "row_bounds": [{"row": "1", "top": 150, "bottom": 760}],
                    "cell_diagnostics": [{
                        "row": "1", "field": "table_type",
                        "crop": [200, 150, 900, 760],
                        "initial_text": "Pithos", "initial_confidence": .98,
                        "initial_tokens": [{"text": "Pithos", "confidence": .98,
                                            "bbox": [240, 180, 330, 220]}],
                        "focused_crop": [200, 150, 900, 760],
                        "focused_preparation": "standard",
                        "focused_text": "Pithos", "focused_confidence": .99,
                        "focused_tokens": [], "accepted_text": "Pithos",
                        "accepted_source": "focused_cell",
                    }],
                },
                "ocr_diagnostics": [{
                    "row": "1", "field": "nonplastics_type",
                    "crop": [400, 300, 460, 350],
                    "retry_tokens": [{"text": "L", "confidence": .99,
                                      "bbox": [10, 10, 30, 40]}],
                    "page_overlap_tokens": [],
                    "accepted_value": "L", "status": "accepted",
                }],
                "warnings": [],
            }

    monkeypatch.setattr(app_module, "project_manager", manager)
    monkeypatch.setattr(app_module, "PaddleOCRStructuredExtractor", FakeRerunOCR)
    monkeypatch.setattr(app_module, "get_local_ocr_health", lambda **_: {
        "available": True, "message": "Local OCR is ready.", "error": "",
        "models_initialized": True,
    })
    import importlib.util
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *args, **kwargs: object() if name in {"paddle", "paddleocr"}
                        else real_find_spec(name, *args, **kwargs))
    client = app_module.app.test_client()
    response = client.post(
        f"/api/projects/{project_id}/metadata-link/figures/2.1/rerun",
        json={"backend": "ocr"})
    assert response.status_code == 202
    _wait_linkage_job(project_id, project_path, response.get_json()["job"]["id"])
    updated = client.get(
        f"/api/projects/{project_id}/metadata-link/figures/2.1").get_json()["figure"]
    assert updated["status"] == "ready"
    assert updated["table_rows"][0]["table_type"] == "Pithos"
    assert updated["table_pages"][0]["boundary"]["closing_rule_y"] == 760
    diagnostics = client.get(
        f"/api/projects/{project_id}/metadata-link/figures/2.1/diagnostics/page_1.jpg")
    assert diagnostics.status_code == 200
    assert diagnostics.get_json()["diagnostics"]["ocr_diagnostics"][0]["accepted_value"] == "L"
    assert observed_processing_states == ["processing"]

    evidence = client.get(
        f"/api/projects/{project_id}/metadata-link/evidence/page_1.jpg?figure=2.1&kind=table&overlay=1")
    assert evidence.status_code == 200
    assert evidence.mimetype == "image/png"
    mandatory_grid = client.get(
        f"/api/projects/{project_id}/metadata-link/evidence/page_1.jpg"
        "?figure=2.1&kind=table&overlay=0")
    assert mandatory_grid.status_code == 200
    assert mandatory_grid.data == evidence.data
    diagnostic = client.get(
        f"/api/projects/{project_id}/metadata-link/evidence/page_1.jpg"
        "?figure=2.1&kind=table&ocr_row=1&ocr_field=nonplastics_type")
    assert diagnostic.status_code == 200
    assert diagnostic.mimetype == "image/png"
    cell = client.get(
        f"/api/projects/{project_id}/metadata-link/evidence/page_1.jpg"
        "?figure=2.1&kind=table&cell_row=1&cell_field=table_type")
    assert cell.status_code == 200
    assert cell.mimetype == "image/png"
    focused_cell = client.get(
        f"/api/projects/{project_id}/metadata-link/evidence/page_1.jpg"
        "?figure=2.1&kind=table&cell_row=1&cell_field=table_type&cell_view=focused")
    assert focused_cell.status_code == 200
    assert focused_cell.mimetype == "image/png"


def test_manual_column_edges_persist_reset_and_queue_priority_zero(tmp_path, monkeypatch):
    manager, project_id, project_path = _make_api_project(tmp_path, "Manual Columns")
    figure = {
        "figure_id": "2.1", "processing_status": "reviewable", "reviewer_revision": 0,
        "drawings": [{"mask_file": "card_0", "fingerprint": "x", "vessel_number": "1"}],
        "table_rows": [{"table_no": "1", "table_type": "Jar"}],
        "table_pages": [{
            "image_name": "page_1.jpg", "source_pdf": "hesban.pdf",
            "boundary": {
                "column_source": "header_detected",
                "normalized_column_edges": [index / 22 for index in range(23)],
            },
        }],
    }
    validate_figure(figure, Hesban11Profile())
    save_linkage_state(project_path, {"profile": "hesban11", "status": "complete",
                                      "figures": [figure]})
    queued = []

    class FakeCoordinator:
        def enqueue(self, project_id, project_path, kind, priority, payload=None,
                    dedupe_key=None, **kwargs):
            queued.append((kind, priority, payload, dedupe_key))
            return {"id": f"job-{len(queued)}", "kind": kind, "status": "queued"}

    monkeypatch.setattr(app_module, "project_manager", manager)
    monkeypatch.setattr(app_module, "linkage_job_coordinator", FakeCoordinator())
    client = app_module.app.test_client()
    key = figure["figure_key"]
    edges = [.1 + index * .035 for index in range(22)] + [.9]
    saved = client.put(
        f"/api/projects/{project_id}/metadata-link/figures/{key}/pages/page_1.jpg/columns",
        json={"reviewer_revision": 0, "normalized_column_edges": edges})
    assert saved.status_code == 202
    persisted = load_linkage_state(project_path)["figures"][0]
    assert persisted["table_pages"][0]["manual_column_edges"] == [
        round(value, 8) for value in edges]
    assert persisted["review_status"] == "pending"
    assert queued[0][0:2] == ("boundary_reread", 0)

    reset = client.delete(
        f"/api/projects/{project_id}/metadata-link/figures/{key}/pages/page_1.jpg/columns",
        json={"reviewer_revision": 1})
    assert reset.status_code == 202
    assert "manual_column_edges" not in load_linkage_state(
        project_path)["figures"][0]["table_pages"][0]


def test_row_add_delete_and_cell_edit_are_retained_across_reread(tmp_path, monkeypatch):
    manager, project_id, project_path = _make_api_project(tmp_path, "Row Overrides")
    figure = {
        "figure_id": "2.1", "processing_status": "reviewable", "reviewer_revision": 0,
        "drawings": [{"mask_file": "card_0", "fingerprint": "x", "vessel_number": "1"}],
        "table_rows": [
            {"table_no": "1", "table_type": "OCR jar", "source_image": "page.jpg",
             "normalized_table_no": "1"},
            {"table_no": "2", "table_type": "OCR bowl", "source_image": "page.jpg",
             "normalized_table_no": "2"},
        ],
    }
    validate_figure(figure, Hesban11Profile())
    save_linkage_state(project_path, {"profile": "hesban11", "status": "complete",
                                      "figures": [figure]})
    monkeypatch.setattr(app_module, "project_manager", manager)
    client = app_module.app.test_client()
    edited = client.patch(
        f"/api/projects/{project_id}/metadata-link/figures/{figure['figure_key']}",
        json={"reviewer_revision": 0, "table_rows": [
            {"_review_key": "page.jpg|1", "table_no": "1",
             "table_type": "Researcher jar"},
            {"table_no": "3", "table_type": "Added bowl"},
        ]})
    assert edited.status_code == 200
    saved = load_linkage_state(project_path)["figures"][0]
    assert saved["review_overrides"]["cells"]["page.jpg|1"]["table_type"] == "Researcher jar"
    assert saved["review_overrides"]["deleted"] == ["page.jpg|2"]
    assert saved["review_overrides"]["added"][0]["table_no"] == "3"

    saved["table_rows"] = [
        {"table_no": "1", "table_type": "New OCR jar", "source_image": "page.jpg"},
        {"table_no": "2", "table_type": "New OCR bowl", "source_image": "page.jpg"},
    ]
    assert apply_reviewer_row_overrides(saved, Hesban11Profile()) == []
    assert [(row["table_no"], row["table_type"]) for row in saved["table_rows"]] == [
        ("1", "Researcher jar"), ("3", "Added bowl")]


def test_tabular_boxes_use_staged_vessel_number_not_mask_suffix(tmp_path, monkeypatch):
    manager, project_id, project_path = _make_api_project(tmp_path, "Labels")
    from PIL import Image
    Image.new("RGB", (500, 500), "white").save(project_path / "images" / "page_0.jpg")
    pd.DataFrame([{
        "file": "page_0", "mask_file": "page_0_mask_layer_0", "Notes": "keep",
    }]).to_csv(project_path / "cards" / "mask_info.csv", index=False)
    pd.DataFrame([{
        "bbox": "(10, 20, 200, 250)", "mask_file": "page_0_mask_layer_0.png",
    }]).to_csv(project_path / "cards" / "mask_info_annots.csv", index=False)
    save_linkage_state(project_path, {
        "schema_version": 1, "profile": "hesban11", "status": "complete",
        "progress": {}, "warnings": [], "figures": [{
            "figure_id": "2.1", "drawings": [{
                "mask_file": "page_0_mask_layer_0", "image_name": "page_0.jpg",
                "fingerprint": "x", "vessel_number": "1",
            }], "table_rows": [], "matches": [],
        }],
    })
    monkeypatch.setattr(app_module, "project_manager", manager)
    client = app_module.app.test_client()
    response = client.post(f"/api/projects/{project_id}/tabular/load", json={"img_num": 0})
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["annotations"][0]["label"] == "1"
    assert payload["annotations"][0]["row_key"] == "page_0_mask_layer_0"
    assert payload["table"][0]["No."] == "1"
    assert payload["table"][0]["mask_file"] == "page_0_mask_layer_0"
    assert "ID" not in payload["columns"]
    edited = payload["table"]
    edited[0]["No."] = "2"
    saved = client.post(f"/api/projects/{project_id}/tabular/save",
                        json={"image_name": "page_0", "table": edited})
    assert saved.status_code == 200
    assert load_linkage_state(project_path)["figures"][0]["drawings"][0]["vessel_number"] == "2"


def test_reviewable_figure_autosaves_during_later_ocr_and_rejects_stale_revision(
        tmp_path, monkeypatch):
    manager, project_id, project_path = _make_api_project(tmp_path, "Live Review")
    figure = {
        "figure_id": "2.1", "processing_status": "reviewable", "reviewer_revision": 0,
        "drawings": [{"mask_file": "card_0", "fingerprint": "x", "vessel_number": "1"}],
        "table_rows": [{"table_no": "1", "table_type": "OCR value"}],
    }
    validate_figure(figure, Hesban11Profile())
    save_linkage_state(project_path, {
        "schema_version": 1, "profile": "hesban11", "status": "running",
        "figures": [figure], "warnings": [],
    })
    monkeypatch.setattr(app_module, "project_manager", manager)
    client = app_module.app.test_client()
    saved = client.patch(
        f"/api/projects/{project_id}/metadata-link/figures/2.1",
        json={"reviewer_revision": 0,
              "table_rows": [{"table_no": "1", "table_type": "Manual value"}]},
    )
    assert saved.status_code == 200
    assert saved.get_json()["reviewer_revision"] == 1
    stale = client.patch(
        f"/api/projects/{project_id}/metadata-link/figures/2.1",
        json={"reviewer_revision": 0,
              "table_rows": [{"table_no": "1", "table_type": "Lost value"}]},
    )
    assert stale.status_code == 409
    assert stale.get_json()["conflict"] is True
    assert stale.get_json()["figure"]["table_rows"][0]["table_type"] == "Manual value"


def test_processing_figure_cannot_be_edited_during_ocr(tmp_path, monkeypatch):
    manager, project_id, project_path = _make_api_project(tmp_path, "Processing")
    save_linkage_state(project_path, {
        "schema_version": 1, "profile": "hesban11", "status": "running",
        "figures": [{
            "figure_id": "2.1", "processing_status": "processing",
            "drawings": [], "table_rows": [],
        }],
    })
    monkeypatch.setattr(app_module, "project_manager", manager)
    response = app_module.app.test_client().patch(
        f"/api/projects/{project_id}/metadata-link/figures/2.1",
        json={"reviewer_revision": 0, "figure_caption": "Changed"},
    )
    assert response.status_code == 409
    assert response.get_json()["processing"] is True


def test_row_operations_and_safe_warning_override_are_audited_in_csv(
        tmp_path, monkeypatch):
    manager, project_id, project_path = _make_api_project(tmp_path, "Review Tools")
    pd.DataFrame([{"file": "page_0", "mask_file": "card_0"}]).to_csv(
        project_path / "cards" / "mask_info.csv", index=False)
    figure = {
        "figure_id": "2.1", "processing_status": "reviewable", "reviewer_revision": 0,
        "drawing_pages": [{"printed_page": "19", "source_pdf": "hesban.pdf"}],
        "table_pages": [{"printed_page": "20", "source_pdf": "hesban.pdf"}],
        "drawings": [{"mask_file": "card_0", "fingerprint": "x", "vessel_number": "1"}],
        "table_rows": [{"table_no": "1", "table_type": "Pithos"}],
        "extraction_warnings": [{"code": "column_header_fallback",
                                 "message": "Fallback column alignment", "page": "20"}],
    }
    validate_figure(figure, Hesban11Profile())
    warning_id = next(warning["id"] for warning in figure["warnings"]
                      if warning["code"] == "column_header_fallback")
    save_linkage_state(project_path, {
        "schema_version": 1, "profile": "hesban11", "status": "complete",
        "figures": [figure], "warnings": [],
    })
    monkeypatch.setattr(app_module, "project_manager", manager)
    client = app_module.app.test_client()
    edited = client.patch(
        f"/api/projects/{project_id}/metadata-link/figures/2.1",
        json={
            "reviewer_revision": 0,
            "warning_overrides": {warning_id: {
                "reason": "column_alignment_verified", "note": "Checked the scan",
            }},
            "row_operations": [{"action": "sort", "index": -1}],
        },
    )
    assert edited.status_code == 200
    payload = edited.get_json()["figure"]
    assert payload["status"] == "ready"
    assert payload["warning_overrides"][warning_id]["note"] == "Checked the scan"
    applied = client.post(
        f"/api/projects/{project_id}/metadata-link/apply", json={"figure_ids": ["2.1"]})
    assert applied.status_code == 200
    csv = pd.read_csv(project_path / "cards" / "mask_info.csv",
                      dtype=str, keep_default_na=False)
    assert csv.loc[0, "Link Status"] == "approved_with_overrides"


def test_warning_autosave_preserves_table_grid_and_manual_columns(
        tmp_path, monkeypatch):
    manager, project_id, project_path = _make_api_project(
        tmp_path, "Warning Grid Preservation")
    (project_path / "page_manifest.json").write_text(json.dumps({
        "schema_version": 1, "profile": "hesban11", "pages": [{
            "image_name": "page_1.jpg", "source_pdf": "hesban.pdf",
            "logical_index": 1, "printed_page": "20",
        }],
    }), encoding="utf-8")
    boundary = {
        "table_bounds": [100, 120, 900, 760],
        "column_bounds": [[100, 140], [140, 220]],
        "row_bounds": [{"row": "1", "top": 180, "bottom": 260}],
        "normalized_column_edges": [index / 22 for index in range(23)],
        "column_source": "manual",
    }
    manual_edges = [index / 22 for index in range(23)]
    figure = {
        "figure_id": "2.1", "processing_status": "reviewable",
        "drawing_pages": [{"printed_page": "19", "source_pdf": "hesban.pdf"}],
        "table_pages": [{
            "image_name": "page_1.jpg", "source_pdf": "hesban.pdf",
            "logical_index": 1, "printed_page": "20", "crop": None,
            "boundary": boundary, "manual_column_edges": manual_edges,
            "diagnostics_ref": "metadata_link_diagnostics/example.json",
        }],
        "drawings": [{"mask_file": "card_0", "fingerprint": "x",
                      "vessel_number": "1"}],
        "table_rows": [{"table_no": "1", "table_type": "Pithos"}],
        "extraction_warnings": [{
            "code": "column_header_fallback", "message": "Fallback columns",
        }],
    }
    validate_figure(figure, Hesban11Profile())
    warning_id = next(warning["id"] for warning in figure["warnings"]
                      if warning["code"] == "column_header_fallback")
    save_linkage_state(project_path, {
        "profile": "hesban11", "status": "complete", "figures": [figure],
    })
    monkeypatch.setattr(app_module, "project_manager", manager)

    response = app_module.app.test_client().patch(
        f"/api/projects/{project_id}/metadata-link/figures/{figure['figure_key']}",
        json={
            "reviewer_revision": 0,
            # This mirrors the browser autosave payload that exposed the bug.
            "table_pages": [{"image_name": "page_1.jpg"}],
            "warning_overrides": {warning_id: {
                "reason": "column_alignment_verified",
            }},
        },
    )

    assert response.status_code == 200
    saved_page = response.get_json()["figure"]["table_pages"][0]
    assert saved_page["boundary"] == boundary
    assert saved_page["manual_column_edges"] == manual_edges
    assert saved_page["diagnostics_ref"] == "metadata_link_diagnostics/example.json"
    assert app_module.linkage_job_coordinator.snapshot(
        project_id, project_path, ensure_worker=False)["queued"] == []


def test_override_timestamp_is_server_owned_and_stable_across_autosaves(
        tmp_path, monkeypatch):
    manager, project_id, project_path = _make_api_project(tmp_path, "Override Audit")
    figure = {
        "figure_id": "2.1", "processing_status": "reviewable",
        "drawings": [{"mask_file": "card_0", "fingerprint": "x", "vessel_number": "1"}],
        "table_rows": [{"table_no": "1", "table_type": "Pithos"}],
        "extraction_warnings": [{"code": "column_header_fallback", "message": "Fallback columns"}],
    }
    validate_figure(figure, Hesban11Profile())
    warning_id = next(w["id"] for w in figure["warnings"]
                   if w["code"] == "column_header_fallback")
    save_linkage_state(project_path, {"profile": "hesban11", "status": "complete",
                                      "figures": [figure]})
    monkeypatch.setattr(app_module, "project_manager", manager)
    client = app_module.app.test_client()
    bad = client.patch(
        f"/api/projects/{project_id}/metadata-link/figures/{figure['figure_key']}",
        json={"reviewer_revision": 0, "warning_overrides": {warning_id: {
            "reason": "arbitrary_reason", "at": "1900-01-01T00:00:00Z",
        }}},
    )
    assert bad.status_code == 400
    first = client.patch(
        f"/api/projects/{project_id}/metadata-link/figures/{figure['figure_key']}",
        json={"reviewer_revision": 0, "warning_overrides": {warning_id: {
            "reason": "column_alignment_verified", "at": "1900-01-01T00:00:00Z",
        }}},
    )
    assert first.status_code == 200
    first_payload = first.get_json()["figure"]
    first_at = first_payload["warning_overrides"][warning_id]["at"]
    assert not first_at.startswith("1900-")
    second = client.patch(
        f"/api/projects/{project_id}/metadata-link/figures/{figure['figure_key']}",
        json={"reviewer_revision": 1,
              "figure_caption": "An unrelated autosave",
              "warning_overrides": {warning_id: {
                  "reason": "column_alignment_verified", "note": "Still checked",
              }}},
    )
    assert second.status_code == 200
    assert second.get_json()["figure"]["warning_overrides"][warning_id]["at"] == first_at


def test_stable_figure_key_remains_editable_after_figure_id_correction(
        tmp_path, monkeypatch):
    manager, project_id, project_path = _make_api_project(tmp_path, "Stable Key")
    from PIL import Image
    Image.new("RGB", (100, 100), "white").save(project_path / "images" / "page_0.jpg")
    figure = {
        "figure_id": "2.1", "processing_status": "reviewable",
        "drawing_pages": [{"image_name": "page_0.jpg", "source_pdf": "hesban.pdf"}],
        "drawings": [{"mask_file": "card_0", "fingerprint": "x", "vessel_number": "1",
                      "image_name": "page_0.jpg", "bbox": [10, 10, 80, 80]}],
        "table_rows": [{"table_no": "1", "table_type": "Pithos"}],
    }
    validate_figure(figure, Hesban11Profile())
    save_linkage_state(project_path, {"profile": "hesban11", "status": "complete",
                                      "figures": [figure]})
    key = figure["figure_key"]
    monkeypatch.setattr(app_module, "project_manager", manager)
    client = app_module.app.test_client()
    renamed = client.patch(
        f"/api/projects/{project_id}/metadata-link/figures/{key}",
        json={"reviewer_revision": 0, "figure_id": "2.2"},
    )
    assert renamed.status_code == 200
    assert renamed.get_json()["figure"]["figure_key"] == key
    edited = client.patch(
        f"/api/projects/{project_id}/metadata-link/figures/{key}",
        json={"reviewer_revision": 1, "figure_caption": "Figure 2.2 corrected"},
    )
    assert edited.status_code == 200
    assert edited.get_json()["figure"]["figure_id"] == "2.2"
    evidence = client.get(
        f"/api/projects/{project_id}/metadata-link/evidence/page_0.jpg?figure={key}")
    assert evidence.status_code == 200
    assert evidence.mimetype == "image/png"


def test_completed_figure_can_be_approved_while_another_figure_is_processing(
        tmp_path, monkeypatch):
    manager, project_id, project_path = _make_api_project(tmp_path, "Live Approval")
    pd.DataFrame([{"file": "page_0", "mask_file": "card_0"}]).to_csv(
        project_path / "cards" / "mask_info.csv", index=False)
    complete = {
        "figure_id": "2.1", "processing_status": "ready", "reviewer_revision": 0,
        "drawings": [{"mask_file": "card_0", "fingerprint": "x", "vessel_number": "1"}],
        "table_rows": [{"table_no": "1", "table_type": "Pithos"}],
    }
    processing = {
        "figure_id": "2.2", "processing_status": "processing", "reviewer_revision": 0,
        "drawings": [{"mask_file": "card_1", "fingerprint": "y", "vessel_number": "1"}],
        "table_rows": [],
    }
    validate_figure(complete, Hesban11Profile())
    validate_figure(processing, Hesban11Profile())
    save_linkage_state(project_path, {
        "profile": "hesban11", "status": "running", "figures": [complete, processing],
    })
    stale_background = load_linkage_state(project_path)
    monkeypatch.setattr(app_module, "project_manager", manager)
    client = app_module.app.test_client()
    blocked = client.post(
        f"/api/projects/{project_id}/metadata-link/apply",
        json={"figure_ids": ["2.2"]},
    )
    assert blocked.status_code == 409
    assert "still being extracted" in blocked.get_json()["error"]
    response = client.post(
        f"/api/projects/{project_id}/metadata-link/apply",
        json={"figure_ids": ["2.1"], "replace_imported": True},
    )
    assert response.status_code == 200
    approved = load_linkage_state(project_path)["figures"][0]
    assert approved["review_status"] == "approved"
    assert approved["reviewer_revision"] == 1
    assert approved["matches"][0]["applied_values"]["Type"] == "Pithos"

    # Simulate the long-running linker saving its pre-approval in-memory
    # state. The newer reviewer revision must survive this stale progress
    # write, including approval and feature-owned CSV evidence.
    save_linkage_state(project_path, stale_background)
    after_background = load_linkage_state(project_path)["figures"][0]
    assert after_background["review_status"] == "approved"
    assert after_background["processing_status"] == "approved"
    assert after_background["reviewer_revision"] == 1
    assert after_background["matches"][0]["applied_values"]["Type"] == "Pithos"
    csv = pd.read_csv(project_path / "cards" / "mask_info.csv",
                      dtype=str, keep_default_na=False)
    assert csv.loc[0, "Type"] == "Pithos"


def test_measurement_review_exports_only_verified_diameter(tmp_path, monkeypatch):
    from PIL import Image, ImageDraw

    manager, project_id, project_path = _make_api_project(tmp_path, "Diameter API")
    page_name = "page_0.png"
    image = Image.new("L", (1200, 1600), 255)
    draw = ImageDraw.Draw(image)
    draw.line((200, 300, 800, 300), fill=0, width=3)
    draw.line((500, 300, 500, 600), fill=0, width=3)
    for index in range(5):
        left, right = 120 + index * 50, 120 + (index + 1) * 50
        draw.rectangle((left, 1450, right, 1464), outline=0, width=2)
        if index % 2 == 0:
            draw.rectangle((left + 2, 1452, right - 2, 1462), fill=0)
    draw.line((120, 1445, 120, 1465), fill=0, width=2)
    draw.line((370, 1445, 370, 1465), fill=0, width=2)
    image.save(project_path / "images" / page_name)
    pd.DataFrame([{"file": "page_0", "mask_file": "card_0"}]).to_csv(
        project_path / "cards" / "mask_info.csv", index=False)
    (project_path / "page_manifest.json").write_text(json.dumps({"pages": [{
        "image_name": page_name, "source_pdf": "hesban.pdf", "pdf_page_index": 0,
        "logical_index": 0, "render_dpi": 400,
    }]}), encoding="utf-8")
    figure = {
        "figure_id": "2.1", "processing_status": "reviewable",
        "drawing_pages": [{"image_name": page_name, "source_pdf": "hesban.pdf"}],
        "drawings": [{"mask_file": "card_0", "fingerprint": "bbox-a",
                      "image_name": page_name, "bbox": [150, 260, 900, 700],
                      "vessel_number": "1"}],
        "table_rows": [{"table_no": "1", "table_type": "Bowl"}],
    }
    validate_figure(figure, Hesban11Profile())
    save_linkage_state(project_path, {"profile": "hesban11", "status": "complete",
                                      "figures": [figure]})
    monkeypatch.setattr(app_module, "project_manager", manager)
    client = app_module.app.test_client()

    measured = client.post(
        f"/api/projects/{project_id}/metadata-link/figures/{figure['figure_key']}/measure",
        json={"reviewer_revision": 0})
    assert measured.status_code == 202
    _wait_linkage_job(project_id, project_path, measured.get_json()["job"]["id"])
    payload = client.get(
        f"/api/projects/{project_id}/metadata-link/figures/{figure['figure_key']}").get_json()["figure"]
    assert payload["drawings"][0]["measurement"]["status"] == "verified_automatic"

    # A structurally valid automatic measurement is accepted without an extra click.
    applied = client.post(f"/api/projects/{project_id}/metadata-link/apply",
                          json={"figure_ids": ["2.1"]})
    assert applied.status_code == 200
    csv = pd.read_csv(project_path / "cards" / "mask_info.csv",
                      dtype=str, keep_default_na=False)
    assert float(csv.loc[0, "Rim Diameter (cm)"]) > 0
    assert csv.loc[0, "Diameter Status"] == "verified_automatic"

    saved = client.get(
        f"/api/projects/{project_id}/metadata-link/figures/{figure['figure_key']}").get_json()["figure"]
    corrected = client.patch(
        f"/api/projects/{project_id}/metadata-link/figures/{saved['figure_key']}",
        json={"reviewer_revision": saved["reviewer_revision"],
              "scale_calibrations": {page_name: {
                  "p1": [120, 1450], "p2": [370, 1450], "real_cm": 10}},
              "measurements": {"card_0": {
                  "rim_endpoints": [[200, 300], [800, 300]]}}})
    assert corrected.status_code == 200
    corrected_figure = corrected.get_json()["figure"]
    assert corrected_figure["scale_calibrations"][page_name]["status"] == "verified_manual"
    assert corrected_figure["drawings"][0]["measurement"]["verified_cm"] == 24

    reapplied = client.post(
        f"/api/projects/{project_id}/metadata-link/apply",
        json={"figure_ids": ["2.1"], "replace_imported": True})
    assert reapplied.status_code == 200
    csv = pd.read_csv(project_path / "cards" / "mask_info.csv",
                      dtype=str, keep_default_na=False)
    assert csv.loc[0, "Rim Diameter (cm)"] == "24.0"
    assert csv.loc[0, "Diameter Status"] == "verified_manual"
    evidence = client.get(
        f"/api/projects/{project_id}/metadata-link/evidence/{page_name}"
        f"?figure={corrected_figure['figure_key']}&measurement=1")
    assert evidence.status_code == 200
