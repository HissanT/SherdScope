from pathlib import Path
import json

import pandas as pd
import pytest
from PIL import Image
import metadata_linker as linker_module

from metadata_linker import (
    AmbiguousSourceError,
    HESBAN_TABLE_COLUMNS,
    Hesban11Profile,
    MetadataLinker,
    StructuredExtractor,
    apply_approved_figures,
    bbox_fingerprint,
    invalidate_linkage_for_card_changes,
    normalize_figure_id,
    normalize_vessel_number,
    MetadataLinkError,
    ReviewerRevisionConflict,
    save_linkage_state,
    validate_figure,
    load_linkage_state,
    migrate_linkage_columns,
)


class FakeExtractor(StructuredExtractor):
    def extract_drawing_identifiers(self, image_path, cards, page_context):
        numbers = {card["mask_file"]: {"number": str(index + 1)}
                   for index, card in enumerate(cards)}
        return {
            "figure_id": page_context.get("figure_id"),
            "figure_caption": page_context.get("figure_caption"),
            "printed_page": page_context.get("printed_page"),
            "drawings": numbers,
        }

    def extract_table(self, image_path, crop, figure_id, expected_numbers, page_context):
        page = Path(image_path).stem
        rows = []
        if page.endswith("page_1"):
            rows = [{"table_no": "1", "table_type": "Pithos", "fabric_core": "Gray"}]
        elif page.endswith("page_2"):
            rows = [{"table_no": "2", "table_type": "Bowl", "nonplastics_size": "7A\n6A"}]
        return {
            "is_table": bool(rows), "figure_id": figure_id if rows else None,
            "printed_page": page_context.get("printed_page"), "rows": rows,
            "boundary": ({"header_confirmed": True,
                          "has_closing_rule": page.endswith("page_2"),
                          "continues": page.endswith("page_1")} if rows else {}),
        }


class SamePageExtractor(FakeExtractor):
    def __init__(self):
        self.crops = []

    def extract_table(self, image_path, crop, figure_id, expected_numbers, page_context):
        self.crops.append((Path(image_path).stem, crop))
        if Path(image_path).stem.endswith("page_0"):
            return {"is_table": True, "figure_id": figure_id, "figure_caption": "Figure 2.1",
                    "rows": [{"table_no": "1", "table_type": "Pithos"},
                             {"table_no": "2", "table_type": "Bowl"}],
                    "boundary": {"header_confirmed": True, "has_closing_rule": True}}
        return {"is_table": False, "rows": []}


def make_project(tmp_path):
    project = tmp_path / "project"
    for folder in ["cards", "images", "pdf_source"]:
        (project / folder).mkdir(parents=True)
    pages = []
    for index, printed in enumerate(["19", "20", "21"]):
        name = f"Hesban_page_{index}.jpg"
        Image.new("RGB", (1000, 1400), "white").save(project / "images" / name)
        pages.append({
            "image_name": name, "source_pdf": "hesban.pdf", "pdf_page_index": 49 + index,
            "printed_page": printed, "split_part": None, "logical_index": index,
            "page_text": f"Figure 2.1{', continued.' if index else ''}\n",
            "figure_id": "2.1", "figure_caption": "Figure 2.1",
        })
    (project / "page_manifest.json").write_text(
        __import__("json").dumps({"schema_version": 1, "profile": "hesban11", "pages": pages}),
        encoding="utf-8")
    pd.DataFrame([
        {"bbox": "(100, 100, 300, 300)", "mask_file": "Hesban_page_0_mask_layer_0.png"},
        {"bbox": "(500, 100, 700, 300)", "mask_file": "Hesban_page_0_mask_layer_1.png"},
    ]).to_csv(project / "cards" / "mask_info_annots.csv", index=False)
    pd.DataFrame([
        {"file": "Hesban_page_0", "mask_file": "Hesban_page_0_mask_layer_0", "Notes": "manual A"},
        {"file": "Hesban_page_0", "mask_file": "Hesban_page_0_mask_layer_1", "Notes": "manual B"},
    ]).to_csv(project / "cards" / "mask_info.csv", index=False)
    return project


def test_normalization_and_caption_detection():
    assert normalize_figure_id("Figure 2.01, continued.") == "2.01"
    assert normalize_figure_id("fig. 2:3") == "2.3"
    assert normalize_vessel_number("No. 002A") == "2a"
    context = Hesban11Profile().detect_figure_context("Figure 2.1, continued. LB/Iron I Pottery")
    assert context == {
        "figure_id": "2.1", "caption": "Figure 2.1, continued. LB/Iron I Pottery",
        "continued": True,
    }
    assert Hesban11Profile().detect_printed_page(
        "Parallels: 13th: example\nFigure Reference 2.1:9\nIRON AGE   21\nFigure 2.1"
    ) == "21"


def test_unique_join_and_duplicate_detection():
    figure = {
        "drawings": [
            {"mask_file": "a", "fingerprint": "x", "vessel_number": "01"},
            {"mask_file": "b", "fingerprint": "y", "vessel_number": "2"},
        ],
        "table_rows": [
            {"table_no": "1", "table_type": "Pithos"},
            {"table_no": "2", "table_type": "Bowl"},
        ],
    }
    validate_figure(figure, Hesban11Profile())
    assert figure["status"] == "ready"
    assert [match["status"] for match in figure["matches"]] == ["ready", "ready"]
    figure["table_rows"].append({"table_no": "2", "table_type": "Duplicate"})
    validate_figure(figure, Hesban11Profile())
    assert figure["status"] == "needs_review"
    assert any(warning["code"] == "duplicate_table_number" for warning in figure["warnings"])


def test_zero_vessel_number_is_unresolved():
    figure = {
        "drawings": [{"mask_file": "page_mask_layer_0", "fingerprint": "x", "vessel_number": "0"}],
        "table_rows": [{"table_no": "0", "table_type": "Pithos"}],
    }
    validate_figure(figure, Hesban11Profile())
    assert figure["status"] == "needs_review"
    assert figure["drawings"][0]["vessel_number"] == ""


def test_safe_warning_override_allows_ready_but_identity_warning_does_not():
    profile = Hesban11Profile()
    figure = {
        "figure_id": "2.1", "processing_status": "reviewable",
        "drawings": [{"mask_file": "a", "fingerprint": "x", "vessel_number": "1"}],
        "table_rows": [{"table_no": "1", "table_type": "Pithos"}],
        "extraction_warnings": [{"code": "column_header_fallback", "message": "Fallback columns"}],
        "warning_overrides": {},
    }
    validate_figure(figure, profile)
    warning = next(item for item in figure["warnings"] if item["code"] == "column_header_fallback")
    assert warning["overrideable"] is True
    assert figure["status"] == "needs_review"
    figure["warning_overrides"][warning["id"]] = {"reason": "tampered_reason"}
    validate_figure(figure, profile)
    assert figure["status"] == "needs_review"
    figure["warning_overrides"][warning["id"]] = {
        "reason": "column_alignment_verified", "note": "Checked page"
    }
    validate_figure(figure, profile)
    assert figure["status"] == "ready"
    figure["drawings"].append({"mask_file": "b", "fingerprint": "y", "vessel_number": "1"})
    validate_figure(figure, profile)
    duplicate = next(item for item in figure["warnings"]
                     if item["code"] == "duplicate_drawing_number")
    assert duplicate["overrideable"] is False
    assert figure["status"] == "needs_review"


def test_missing_table_end_is_kept_as_evidence_but_not_shown_or_blocking():
    figure = {
        "figure_id": "2.1", "processing_status": "reviewable",
        "drawings": [{"mask_file": "a", "fingerprint": "x", "vessel_number": "1"}],
        "table_rows": [{"table_no": "1", "table_type": "Pithos"}],
        "extraction_warnings": [{"code": "missing_table_end", "message": "No closing rule"}],
    }
    validate_figure(figure, Hesban11Profile())
    assert figure["status"] == "ready"
    assert all(warning["code"] != "missing_table_end" for warning in figure["warnings"])
    assert figure["extraction_warnings"][0]["code"] == "missing_table_end"


def test_cross_pdf_assignment_in_loaded_sidecar_is_always_blocking():
    figure = {
        "figure_id": "2.1",
        "drawing_pages": [{"source_pdf": "drawings.pdf"}],
        "table_pages": [{"source_pdf": "different.pdf"}],
        "drawings": [{"mask_file": "a", "fingerprint": "x", "vessel_number": "1"}],
        "table_rows": [{"table_no": "1", "table_type": "Pithos"}],
        "warning_overrides": {},
    }
    validate_figure(figure, Hesban11Profile())
    warning = next(item for item in figure["warnings"]
                   if item["code"] == "cross_pdf_assignment")
    assert warning["overrideable"] is False
    assert warning["blocking"] is True
    assert figure["status"] == "needs_review"


def test_newer_reviewer_revision_survives_stale_background_save(tmp_path):
    project = tmp_path / "project"
    (project / "cards").mkdir(parents=True)
    base = {
        "schema_version": 1, "profile": "hesban11", "status": "running",
        "figures": [{
            "figure_id": "2.1", "processing_status": "reviewable",
            "reviewer_revision": 0,
            "drawings": [{"mask_file": "a", "fingerprint": "x", "vessel_number": "1"}],
            "table_rows": [{"table_no": "1", "table_type": "OCR value"}],
        }],
    }
    save_linkage_state(project, base)
    reviewed = load_linkage_state(project)
    reviewed["figures"][0]["table_rows"][0]["table_type"] = "Manual correction"
    reviewed["figures"][0]["reviewer_revision"] = 1
    save_linkage_state(project, reviewed)
    stale = {**base, "figures": [{**base["figures"][0],
             "table_rows": [{"table_no": "1", "table_type": "Stale OCR"}],
             "reviewer_revision": 0}]}
    save_linkage_state(project, stale)
    final = load_linkage_state(project)
    assert final["figures"][0]["table_rows"][0]["table_type"] == "Manual correction"
    assert final["figures"][0]["reviewer_revision"] == 1


def test_legacy_sidecar_receives_stable_review_defaults(tmp_path):
    project = tmp_path / "project"
    cards = project / "cards"
    cards.mkdir(parents=True)
    legacy = {
        "profile": "hesban11", "status": "complete",
        "figures": [{
            "figure_id": "2.1",
            "drawings": [{"mask_file": "a", "fingerprint": "x", "vessel_number": "1"}],
            "table_rows": [{"table_no": "1", "table_type": "Pithos"}],
        }],
    }
    (cards / "metadata_linkage.json").write_text(json.dumps(legacy), encoding="utf-8")
    first = load_linkage_state(project)["figures"][0]
    second = load_linkage_state(project)["figures"][0]
    assert first["figure_key"] == second["figure_key"]
    assert first["reviewer_revision"] == 0
    assert first["warning_overrides"] == {}
    assert first["processing_status"] == "ready"


def test_atomic_revision_check_rejects_a_simultaneous_stale_save(tmp_path):
    project = tmp_path / "project"
    (project / "cards").mkdir(parents=True)
    state = {
        "schema_version": 1, "profile": "hesban11", "figures": [{
            "figure_id": "2.1", "reviewer_revision": 0,
            "drawings": [{"mask_file": "a", "fingerprint": "x", "vessel_number": "1"}],
            "table_rows": [{"table_no": "1", "table_type": "OCR"}],
        }],
    }
    save_linkage_state(project, state)
    first = load_linkage_state(project)
    stale = load_linkage_state(project)
    first_figure = first["figures"][0]
    first_figure["reviewer_revision"] = 1
    first_figure["table_rows"][0]["table_type"] = "First reviewer"
    save_linkage_state(project, first, expected_revisions={first_figure["figure_key"]: 0})
    stale_figure = stale["figures"][0]
    stale_figure["reviewer_revision"] = 1
    stale_figure["table_rows"][0]["table_type"] = "Stale reviewer"
    with pytest.raises(ReviewerRevisionConflict):
        save_linkage_state(project, stale,
                           expected_revisions={stale_figure["figure_key"]: 0})
    assert load_linkage_state(project)["figures"][0]["table_rows"][0]["table_type"] == "First reviewer"


def test_legacy_columns_migrate_without_overwriting_public_corrections():
    frame = pd.DataFrame([{
        "mask_file": "page_mask_layer_0", "vessel_number": "1", "table_no": "1",
        "table_type": "OCR value", "Type": "Researcher correction", "Notes": "keep",
    }])
    migrated = migrate_linkage_columns(frame)
    assert migrated.loc[0, "No."] == "1"
    assert migrated.loc[0, "Type"] == "Researcher correction"
    assert migrated.loc[0, "Notes"] == "keep"
    assert "table_type" not in migrated.columns


def test_unexpected_table_row_requires_review():
    figure = {
        "drawings": [{"mask_file": "a", "fingerprint": "x", "vessel_number": "1"}],
        "table_rows": [
            {"table_no": "1", "table_type": "Pithos"},
            {"table_no": "99", "table_type": "Wrong table"},
        ],
    }
    validate_figure(figure, Hesban11Profile())
    assert figure["status"] == "needs_review"
    assert any(warning["code"] == "unexpected_table_row" for warning in figure["warnings"])


def test_model_confidence_does_not_decide_join_readiness():
    figure = {
        "extraction_warnings": [{"code": "low_ocr_confidence", "message": "review text"}],
        "drawings": [{"mask_file": "a", "fingerprint": "x", "vessel_number": "1"}],
        "table_rows": [{"table_no": "1", "table_type": "Bowl"}],
    }
    validate_figure(figure, Hesban11Profile())
    assert figure["status"] == "ready"
    assert figure["warnings"][0]["code"] == "low_ocr_confidence"


def test_table_number_is_matched_normally_but_literal_text_is_preserved():
    figure = {
        "drawings": [{"mask_file": "a", "fingerprint": "x", "vessel_number": "1"}],
        "table_rows": [{"table_no": "No. 001", "table_type": "Bowl"}],
    }
    validate_figure(figure, Hesban11Profile())
    assert figure["status"] == "ready"
    assert figure["table_rows"][0]["table_no"] == "No. 001"
    assert figure["matches"][0]["values"]["table_no"] == "No. 001"


def test_p_plus_one_and_p_plus_two_tables_are_joined(tmp_path):
    project = make_project(tmp_path)
    state = MetadataLinker(project, FakeExtractor(), Hesban11Profile()).run()
    assert state["status"] == "complete"
    assert len(state["figures"]) == 1
    figure = state["figures"][0]
    assert figure["figure_id"] == "2.1"
    assert figure["status"] == "ready"
    assert [page["printed_page"] for page in figure["table_pages"]] == ["20", "21"]
    assert figure["matches"][1]["values"]["nonplastics_size"] == "7A\n6A"


def test_only_the_current_figure_is_marked_processing(tmp_path, monkeypatch):
    project = make_project(tmp_path)
    manifest_path = project / "page_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["pages"][1].update(
        page_text="Figure 2.2", figure_id="2.2", figure_caption="Figure 2.2")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    annotations = pd.read_csv(project / "cards" / "mask_info_annots.csv")
    annotations.loc[len(annotations)] = {
        "bbox": "(100, 400, 300, 600)",
        "mask_file": "Hesban_page_1_mask_layer_0.png",
    }
    annotations.to_csv(project / "cards" / "mask_info_annots.csv", index=False)
    info = pd.read_csv(project / "cards" / "mask_info.csv")
    info.loc[len(info)] = {
        "file": "Hesban_page_1",
        "mask_file": "Hesban_page_1_mask_layer_0",
        "Notes": "manual C",
    }
    info.to_csv(project / "cards" / "mask_info.csv", index=False)

    snapshots = []
    real_save = linker_module.save_linkage_state

    def capture_save(project_path, state, **kwargs):
        result = real_save(project_path, state, **kwargs)
        persisted = json.loads(
            (project / "cards" / "metadata_linkage.json").read_text(encoding="utf-8"))
        snapshots.append([
            figure.get("processing_status") for figure in persisted.get("figures", [])
        ])
        return result

    monkeypatch.setattr(linker_module, "save_linkage_state", capture_save)
    MetadataLinker(project, FakeExtractor(), Hesban11Profile()).run()

    assert any(statuses and all(status == "queued" for status in statuses)
               for statuses in snapshots)
    assert all(statuses.count("processing") <= 1 for statuses in snapshots)


def test_same_page_table_search_crops_below_lowest_drawing(tmp_path):
    project = make_project(tmp_path)
    extractor = SamePageExtractor()
    state = MetadataLinker(project, extractor, Hesban11Profile()).run()
    assert state["figures"][0]["status"] == "ready"
    same_page_crop = next(crop for name, crop in extractor.crops if name.endswith("page_0"))
    assert same_page_crop[1] > 300


def test_omitted_adjacent_caption_requires_and_accepts_number_overlap(tmp_path):
    project = make_project(tmp_path)
    manifest_path = project / "page_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["pages"][1].update(page_text="", figure_id="", figure_caption="")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    state = MetadataLinker(project, FakeExtractor(), Hesban11Profile()).run()
    assert state["figures"][0]["status"] == "ready"
    assert state["figures"][0]["table_pages"][0]["printed_page"] == "20"


def test_search_stops_when_a_different_explicit_figure_begins(tmp_path):
    project = make_project(tmp_path)
    manifest_path = project / "page_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["pages"][1].update(page_text="Figure 2.2", figure_id="2.2",
                                 figure_caption="Figure 2.2")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    state = MetadataLinker(project, FakeExtractor(), Hesban11Profile()).run()
    assert state["figures"][0]["table_pages"] == []
    assert state["figures"][0]["status"] == "needs_review"


def test_approval_is_idempotent_and_preserves_manual_columns(tmp_path):
    project = make_project(tmp_path)
    MetadataLinker(project, FakeExtractor(), Hesban11Profile()).run()
    first = apply_approved_figures(project, ["2.1"])
    second = apply_approved_figures(project, ["2.1"])
    frame = pd.read_csv(project / "cards" / "mask_info.csv", dtype=str, keep_default_na=False)
    assert first["applied_rows"] == second["applied_rows"] == 2
    assert frame["Notes"].tolist() == ["manual A", "manual B"]
    assert frame["No."].tolist() == ["1", "2"]
    assert frame["Type"].tolist() == ["Pithos", "Bowl"]
    assert frame["Non-Plastics - Siz"].iloc[1] == "7A\n6A"
    assert frame["Link Status"].tolist() == ["approved", "approved"]


def test_processing_figure_cannot_be_applied_by_core_approval(tmp_path):
    project = make_project(tmp_path)
    state = MetadataLinker(project, FakeExtractor(), Hesban11Profile()).run()
    state["figures"][0]["processing_status"] = "processing"
    save_linkage_state(project, state)
    with pytest.raises(MetadataLinkError, match="still being extracted"):
        apply_approved_figures(project, ["2.1"])
    state["figures"][0]["processing_status"] = "queued"
    save_linkage_state(project, state)
    with pytest.raises(MetadataLinkError, match="waiting to start"):
        apply_approved_figures(project, ["2.1"])


def test_geometry_change_invalidates_only_affected_figure(tmp_path):
    project = make_project(tmp_path)
    state = MetadataLinker(project, FakeExtractor(), Hesban11Profile()).run()
    state["figures"][0]["review_status"] = "approved"
    save_linkage_state(project, state)
    old = pd.read_csv(project / "cards" / "mask_info_annots.csv", dtype=str)
    new = old.copy()
    new.loc[0, "bbox"] = "(101, 100, 300, 300)"
    invalidate_linkage_for_card_changes(project / "cards", old, new)
    changed = __import__("json").loads((project / "cards" / "metadata_linkage.json").read_text(encoding="utf-8"))
    assert changed["figures"][0]["status"] == "needs_review"
    assert changed["figures"][0]["review_status"] == "pending"
    assert changed["figures"][0]["drawings"][0]["bbox"] == [101, 100, 300, 300]
    assert any(warning["code"] == "card_geometry_changed"
               for warning in changed["figures"][0]["warnings"])
    with pytest.raises(MetadataLinkError, match="requires review"):
        apply_approved_figures(project, ["2.1"])


def test_schema_contains_every_hesban_field():
    assert len(HESBAN_TABLE_COLUMNS) == 22
    assert HESBAN_TABLE_COLUMNS[0] == "table_no"
    assert HESBAN_TABLE_COLUMNS[-1] == "fire"


def test_approval_rejects_missing_card_instead_of_false_approval(tmp_path):
    project = make_project(tmp_path)
    state = MetadataLinker(project, FakeExtractor(), Hesban11Profile()).run()
    state["figures"][0]["drawings"][0]["mask_file"] = "missing_card"
    validate_figure(state["figures"][0], Hesban11Profile())
    save_linkage_state(project, state)
    with pytest.raises(MetadataLinkError, match="was not found"):
        apply_approved_figures(project, ["2.1"])
    saved = pd.read_csv(project / "cards" / "mask_info.csv", dtype=str, keep_default_na=False)
    assert "Link Status" not in saved.columns


def test_multiple_pdf_manifest_requires_source_selection(tmp_path):
    project = make_project(tmp_path)
    manifest_path = project / "page_manifest.json"
    manifest = __import__("json").loads(manifest_path.read_text(encoding="utf-8"))
    extra = dict(manifest["pages"][0], source_pdf="other.pdf", logical_index=99)
    manifest["pages"].append(extra)
    manifest_path.write_text(__import__("json").dumps(manifest), encoding="utf-8")
    with pytest.raises(AmbiguousSourceError, match="Select one source PDF"):
        MetadataLinker(project, FakeExtractor(), Hesban11Profile()).run()
    selected = MetadataLinker(project, FakeExtractor(), Hesban11Profile(), "hesban.pdf").run()
    assert len(selected["figures"]) == 1


def test_missing_required_type_never_becomes_ready():
    figure = {
        "drawings": [{"mask_file": "a", "fingerprint": "x", "vessel_number": "1"}],
        "table_rows": [{"table_no": "1", "table_type": ""}],
    }
    validate_figure(figure, Hesban11Profile())
    assert figure["status"] == "needs_review"
    assert any(item["code"] == "missing_required_value" for item in figure["warnings"])


def test_replace_imported_preserves_a_later_manual_correction(tmp_path):
    project = make_project(tmp_path)
    MetadataLinker(project, FakeExtractor(), Hesban11Profile()).run()
    apply_approved_figures(project, ["2.1"])
    frame = pd.read_csv(project / "cards" / "mask_info.csv", dtype=str, keep_default_na=False)
    frame.loc[0, "Type"] = "Researcher correction"
    frame.to_csv(project / "cards" / "mask_info.csv", index=False)
    state = load_linkage_state(project)
    state["figures"][0]["table_rows"][0]["table_type"] = "OCR rerun value"
    state["figures"][0]["table_rows"][1]["table_type"] = "Updated bowl"
    save_linkage_state(project, state)
    apply_approved_figures(project, ["2.1"], replace_imported=True)
    result = pd.read_csv(project / "cards" / "mask_info.csv", dtype=str, keep_default_na=False)
    assert result.loc[0, "Type"] == "Researcher correction"
    assert result.loc[1, "Type"] == "Updated bowl"
    saved_state = load_linkage_state(project)
    assert saved_state["approval_history"]
    assert saved_state["card_index"]["Hesban_page_0_mask_layer_0"]["fingerprint"]


def test_bbox_formatting_change_does_not_invalidate_geometry(tmp_path):
    project = make_project(tmp_path)
    state = MetadataLinker(project, FakeExtractor(), Hesban11Profile()).run()
    save_linkage_state(project, state)
    old = pd.read_csv(project / "cards" / "mask_info_annots.csv", dtype=str)
    new = old.copy()
    new.loc[0, "bbox"] = "[100,100,300,300]"
    invalidate_linkage_for_card_changes(project / "cards", old, new)
    unchanged = load_linkage_state(project)
    assert unchanged["figures"][0]["status"] == "ready"
