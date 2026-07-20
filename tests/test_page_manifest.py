import json
from pathlib import Path

import fitz
import pytest
from PIL import Image

from catalog.linkage import AmbiguousSourceError, ensure_page_manifest, record_pdf_pages


def _write_pdf(path: Path, pages: int = 2) -> None:
    document = fitz.open()
    for index in range(pages):
        page = document.new_page(width=300, height=400)
        page.insert_text((30, 30), str(19 + index))
        page.insert_text((30, 80), f"Figure 2.{index + 1}")
    document.save(path)
    document.close()


def test_manifest_records_split_side_dpi_and_stable_source_order(tmp_path):
    project = tmp_path / "project"
    (project / "pdf_source").mkdir(parents=True)
    (project / "images").mkdir()
    first = project / "pdf_source" / "first.pdf"
    second = project / "pdf_source" / "second.pdf"
    _write_pdf(first, 1)
    _write_pdf(second, 1)
    record_pdf_pages(project, first, "first", True, render_dpi=400)
    record_pdf_pages(project, second, "second", False, render_dpi=450)
    manifest = record_pdf_pages(project, first, "first", True, render_dpi=425)
    assert manifest["schema_version"] == 1
    assert [page["source_pdf"] for page in manifest["pages"]] == [
        "first.pdf", "first.pdf", "second.pdf"]
    assert [page["logical_index"] for page in manifest["pages"]] == [0, 1, 2]
    assert [page["split_side"] for page in manifest["pages"][:2]] == ["left", "right"]
    assert all(page["render_dpi"] == 425 for page in manifest["pages"][:2])
    assert manifest["pages"][0]["pdf_page_index"] == 0


def test_legacy_manifest_reconstruction_uses_only_existing_names(tmp_path):
    project = tmp_path / "project"
    (project / "pdf_source").mkdir(parents=True)
    (project / "images").mkdir()
    pdf = project / "pdf_source" / "legacy.pdf"
    _write_pdf(pdf, 2)
    Image.new("RGB", (20, 20), "white").save(project / "images" / "legacy_page_0.jpg")
    manifest = ensure_page_manifest(project)
    assert [page["image_name"] for page in manifest["pages"]] == ["legacy_page_0.jpg"]
    assert manifest["pages"][0]["render_dpi"] == 300
    assert json.loads((project / "page_manifest.json").read_text())["schema_version"] == 1


def test_legacy_multiple_pdf_reconstruction_refuses_unsafe_source_mapping(tmp_path):
    project = tmp_path / "project"
    (project / "pdf_source").mkdir(parents=True)
    (project / "images").mkdir()
    _write_pdf(project / "pdf_source" / "first.pdf", 1)
    _write_pdf(project / "pdf_source" / "second.pdf", 1)
    Image.new("RGB", (20, 20), "white").save(project / "images" / "project_page_0.jpg")
    with pytest.raises(AmbiguousSourceError, match="could not be mapped safely"):
        ensure_page_manifest(project, source_pdf="second.pdf")
