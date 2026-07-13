import os
import io
import json
import threading
import time
import zipfile

import pandas as pd
import pytest

pytest.importorskip("ultralytics")
pytest.importorskip("fitz")
os.environ["PYPOTTERYLENS_SKIP_INIT"] = "1"

import app as app_module
from metadata_linker import Hesban11Profile, load_linkage_state, save_linkage_state, validate_figure
from project_manager import ProjectManager


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
    client = app_module.app.test_client()

    state_response = client.get(f"/api/projects/{project_id}/metadata-link/state")
    assert state_response.status_code == 200
    assert state_response.get_json()["state"]["figures"][0]["status"] == "ready"

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


def _make_api_project(tmp_path, name="API Extra"):
    manager = ProjectManager(tmp_path / "projects")
    project = manager.create_project(name)
    return manager, project["project_id"], manager.get_project_path(project["project_id"])


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

        def run(self):
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
    assert second.status_code == 409
    release.set()
    for _ in range(100):
        if not app_module._metadata_link_jobs.get(project_id):
            break
        time.sleep(.01)
    assert not app_module._metadata_link_jobs.get(project_id)


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
        exported = archive.read("HSB_metadata.csv").decode("utf-8")
    assert "Figure" in exported
    assert "Non-Plastics - Siz" in exported
    assert "2.1" in exported
    assert '"7A\n6A"' in exported


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
                },
                "warnings": [],
            }

    monkeypatch.setattr(app_module, "project_manager", manager)
    monkeypatch.setattr(app_module, "PaddleOCRStructuredExtractor", FakeRerunOCR)
    import importlib.util
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *args, **kwargs: object() if name in {"paddle", "paddleocr"}
                        else real_find_spec(name, *args, **kwargs))
    client = app_module.app.test_client()
    response = client.post(
        f"/api/projects/{project_id}/metadata-link/figures/2.1/rerun",
        json={"backend": "ocr"})
    assert response.status_code == 200
    updated = response.get_json()["figure"]
    assert updated["status"] == "ready"
    assert updated["table_rows"][0]["table_type"] == "Pithos"
    assert updated["table_pages"][0]["boundary"]["closing_rule_y"] == 760
    assert observed_processing_states == ["processing"]

    evidence = client.get(
        f"/api/projects/{project_id}/metadata-link/evidence/page_1.jpg?figure=2.1&kind=table&overlay=1")
    assert evidence.status_code == 200
    assert evidence.mimetype == "image/png"


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
    app_module._metadata_link_jobs[project_id] = True
    try:
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
    finally:
        app_module._metadata_link_jobs.pop(project_id, None)


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
    app_module._metadata_link_jobs[project_id] = True
    try:
        response = app_module.app.test_client().patch(
            f"/api/projects/{project_id}/metadata-link/figures/2.1",
            json={"reviewer_revision": 0, "figure_caption": "Changed"},
        )
        assert response.status_code == 409
        assert response.get_json()["processing"] is True
    finally:
        app_module._metadata_link_jobs.pop(project_id, None)


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
        "extraction_warnings": [{"code": "missing_table_end",
                                 "message": "No closing line was found", "page": "20"}],
    }
    validate_figure(figure, Hesban11Profile())
    warning_id = next(warning["id"] for warning in figure["warnings"]
                      if warning["code"] == "missing_table_end")
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
                "reason": "visually_confirmed_complete", "note": "Checked the scan",
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


def test_override_timestamp_is_server_owned_and_stable_across_autosaves(
        tmp_path, monkeypatch):
    manager, project_id, project_path = _make_api_project(tmp_path, "Override Audit")
    figure = {
        "figure_id": "2.1", "processing_status": "reviewable",
        "drawings": [{"mask_file": "card_0", "fingerprint": "x", "vessel_number": "1"}],
        "table_rows": [{"table_no": "1", "table_type": "Pithos"}],
        "extraction_warnings": [{"code": "missing_table_end", "message": "No end"}],
    }
    validate_figure(figure, Hesban11Profile())
    warning_id = next(w["id"] for w in figure["warnings"]
                      if w["code"] == "missing_table_end")
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
            "reason": "visually_confirmed_complete", "at": "1900-01-01T00:00:00Z",
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
                  "reason": "visually_confirmed_complete", "note": "Still checked",
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
    app_module._metadata_link_jobs[project_id] = True
    try:
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
    finally:
        app_module._metadata_link_jobs.pop(project_id, None)
