"""Figure-to-table metadata linking for SherdScope.

The module deliberately separates publication-layout rules, matching, persistence,
and CSV application from the Flask/vision backend.  This keeps the research-data
join deterministic and makes vision output a reviewable candidate rather than the
source of truth.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import threading
from typing import Any, Callable, Iterable, Optional

import pandas as pd

from catalog.profiles import HESBAN_TABLE_COLUMNS


SCHEMA_VERSION = 1
LINKAGE_SCHEMA_VERSION = 2
MANIFEST_NAME = "page_manifest.json"
STATE_NAME = "metadata_linkage.json"
DIAGNOSTICS_DIR_NAME = "metadata_diagnostics"

IDENTITY_COLUMNS = [
    "figure_id", "figure_caption", "vessel_number", "drawing_printed_page",
    "table_printed_pages", "source_pdf", "link_status",
]

MEASUREMENT_COLUMNS = ["rim_diameter_cm", "diameter_status"]
LINKAGE_COLUMNS = IDENTITY_COLUMNS + HESBAN_TABLE_COLUMNS + MEASUREMENT_COLUMNS
NON_BLOCKING_WARNING_CODES = {
    "low_ocr_confidence", "row_anchor_conflict_resolved",
    "cross_column_ocr_withheld",
}
HIDDEN_REVIEW_WARNING_CODES = {"missing_table_end"}
OVERRIDABLE_WARNING_CODES = {
    "missing_required_value", "column_header_fallback",
}
WARNING_OVERRIDE_REASONS = {
    "missing_required_value": {"publication_field_blank"},
    "column_header_fallback": {"column_alignment_verified"},
}
REVIEWER_OWNED_FIGURE_FIELDS = {
    "figure_id", "figure_caption", "table_rows", "table_pages",
    "warning_overrides", "reviewer_revision", "draft_saved_at",
    "review_history", "review_status", "processing_status", "matches",
    "scale_calibrations", "review_overrides",
}
_LINKAGE_STATE_LOCK = threading.RLock()

# JSON and extractor code retain the stable technical names above. CSV files
# and browser-facing tables use publication-style names through this mapping.
PUBLIC_LINKAGE_MAP = {
    "figure_id": "Figure",
    "figure_caption": "Figure Caption",
    "vessel_number": "No.",
    "drawing_printed_page": "Drawing Page",
    "table_printed_pages": "Table Pages",
    "source_pdf": "Source PDF",
    "link_status": "Link Status",
    "table_type": "Type",
    "rim_diameter_cm": "Rim Diameter (cm)",
    "diameter_status": "Diameter Status",
    "table_square": "Sq/Area",
    "table_locus": "Loc",
    "table_pail": "Pail",
    "table_registration": "Reg",
    "fabric_exterior": "Fabric Color - Exterior",
    "fabric_core": "Fabric Color - Core",
    "fabric_interior": "Fabric Color - Interior",
    "nonplastics_type": "Non-Plastics - Typ",
    "nonplastics_size": "Non-Plastics - Siz",
    "nonplastics_shape": "Non-Plastics - Shap",
    "nonplastics_density": "Non-Plastics - Den",
    "voids_type_size": "Voids - Ty/Sz",
    "voids_density": "Voids - Den",
    "manufacture": "Man",
    "surface_exterior": "Surface Treatment - Ext",
    "surface_exterior_color": "Surface Treatment - Exterior Color",
    "surface_interior": "Surface Treatment - Int",
    "surface_interior_color": "Surface Treatment - Interior Color",
    "decor": "Decor",
    "fire": "Fire",
}
PUBLIC_LINKAGE_COLUMNS = list(dict.fromkeys(PUBLIC_LINKAGE_MAP.values()))
ALL_LINKAGE_STORAGE_COLUMNS = list(dict.fromkeys(LINKAGE_COLUMNS + PUBLIC_LINKAGE_COLUMNS))


class MetadataLinkError(RuntimeError):
    pass


class ReviewerRevisionConflict(MetadataLinkError):
    """A reviewer tried to save over a newer draft revision."""

    def __init__(self, figure: dict[str, Any]):
        super().__init__("This figure has a newer saved draft")
        self.figure = figure


class AmbiguousSourceError(MetadataLinkError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def natural_key(value: str) -> list[Any]:
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", str(value))]


def order_figure_table_rows(figure: dict[str, Any]) -> list[dict[str, Any]]:
    """Keep extracted and reviewed rows in natural printed-number order.

    Targeted page rereads retain rows from the other table pages and replace
    only the selected page.  Without a shared ordering step, the replacement
    rows are appended after the retained rows.  Sorting here makes ordering a
    state invariant for fresh extraction, retries, rereads, and legacy state.
    Page order is a stable tie-breaker for duplicate printed row labels.
    """
    rows = figure.get("table_rows", [])
    if not isinstance(rows, list) or len(rows) < 2:
        return rows if isinstance(rows, list) else []

    table_pages = figure.get("table_pages", [])
    page_order = {
        str(page.get("image_name", "")): index
        for index, page in enumerate(sorted(
            (page for page in table_pages if isinstance(page, dict)),
            key=lambda page: (
                int(page.get("logical_index", 10**9))
                if str(page.get("logical_index", "")).lstrip("-").isdigit()
                else 10**9,
                natural_key(str(page.get("image_name", ""))),
            ),
        ))
    }
    unknown_page = len(page_order) + 1

    def row_key(item: tuple[int, dict[str, Any]]) -> tuple[Any, ...]:
        original_index, row = item
        number = str(row.get("normalized_table_no") or
                     row.get("table_no") or "").strip()
        source_rank = page_order.get(str(row.get("source_image", "")), unknown_page)
        if number:
            return (0, natural_key(number), source_rank, original_index)
        return (1, [], source_rank, original_index)

    ordered = [row for _, row in sorted(enumerate(rows), key=row_key)]
    figure["table_rows"] = ordered
    return ordered


def normalize_figure_id(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    match = re.search(r"(?i)\b(?:figure|fig\.?|abb\.?|tav\.?|plate|pl\.?)\s*([A-Za-z]?\d+(?:[.:-]\d+)*(?:[A-Za-z])?)", text)
    if match:
        text = match.group(1)
    text = text.strip().strip(".,;:()[]")
    text = re.sub(r"\s+", "", text)
    text = text.replace(":", ".").replace("-", ".")
    return text.lower()


def normalize_vessel_number(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"(?i)^no\.?\s*", "", text)
    text = re.sub(r"\s+", "", text).strip(".,;:()[]")
    match = re.fullmatch(r"0*(\d+)([a-z]?)", text)
    return f"{int(match.group(1))}{match.group(2)}" if match else text


def is_positive_vessel_number(value: Any) -> bool:
    normalized = normalize_vessel_number(value)
    match = re.fullmatch(r"(\d+)([a-z]?)", normalized)
    return bool(match and int(match.group(1)) > 0)


def migrate_linkage_columns(frame: pd.DataFrame, drop_legacy: bool = True) -> pd.DataFrame:
    """Map legacy technical linkage columns to the public CSV schema.

    A non-empty public value always wins, which preserves user corrections.
    ``table_no`` is folded into ``No.`` only when ``vessel_number`` did not
    already supply it.
    """
    frame = frame.copy()
    if "Sq" in frame.columns:
        if "Sq/Area" not in frame.columns:
            frame["Sq/Area"] = ""
        current = frame["Sq/Area"].fillna("").astype(str)
        legacy = frame["Sq"].fillna("").astype(str)
        frame.loc[current.eq(""), "Sq/Area"] = legacy[current.eq("")]
        if drop_legacy:
            frame = frame.drop(columns=["Sq"])
    for internal, public in PUBLIC_LINKAGE_MAP.items():
        if public not in frame.columns and internal not in frame.columns:
            continue
        if public not in frame.columns:
            frame[public] = ""
        if internal in frame.columns:
            public_values = frame[public].fillna("").astype(str)
            legacy_values = frame[internal].fillna("").astype(str)
            frame.loc[public_values.eq(""), public] = legacy_values[public_values.eq("")]
    if "table_no" in frame.columns:
        if "No." not in frame.columns:
            frame["No."] = ""
        public_values = frame["No."].fillna("").astype(str)
        table_values = frame["table_no"].fillna("").astype(str)
        frame.loc[public_values.eq(""), "No."] = table_values[public_values.eq("")]
    if drop_legacy:
        frame = frame.drop(columns=[column for column in LINKAGE_COLUMNS if column in frame.columns])
    return frame


def parse_bbox(value: Any) -> tuple[int, int, int, int]:
    if isinstance(value, (list, tuple)) and len(value) == 4:
        return tuple(int(float(v)) for v in value)  # type: ignore[return-value]
    parts = re.findall(r"-?\d+(?:\.\d+)?", str(value))
    if len(parts) != 4:
        raise ValueError(f"Invalid bbox: {value!r}")
    return tuple(int(float(v)) for v in parts)  # type: ignore[return-value]


def bbox_fingerprint(image_name: str, bbox: Any) -> str:
    coords = parse_bbox(bbox)
    payload = f"{image_name}|{','.join(map(str, coords))}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def mask_stem(value: Any) -> str:
    return Path(str(value)).stem


class PublicationProfile(ABC):
    slug = "base"
    candidate_lookahead = 2
    table_columns: list[str] = []
    required_table_columns: tuple[str, ...] = ("table_no",)

    @abstractmethod
    def detect_figure_context(self, text: str) -> dict[str, str]:
        raise NotImplementedError

    def normalize_figure(self, value: Any) -> str:
        return normalize_figure_id(value)

    def normalize_number(self, value: Any) -> str:
        return normalize_vessel_number(value)

    def detect_printed_page(self, text: str) -> str:
        lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
        edge_lines = lines[:6] + lines[-6:]
        # Publication running heads are substantially safer than arbitrary numbers
        # in figure captions, table rows, or prose references.
        for line in edge_lines:
            if re.fullmatch(r"\d{1,4}", line):
                return line
        for line in edge_lines:
            match = re.fullmatch(r"[A-Z][A-Z ]{2,}\s+(\d{1,4})", line)
            if match:
                return match.group(1)
            match = re.fullmatch(r"(\d{1,4})\s+[A-Z][A-Z ]{2,}", line)
            if match:
                return match.group(1)
        candidates: list[str] = []
        for line in edge_lines:
            candidates.extend(re.findall(r"(?<![.\d])\d{1,4}(?![.\d])", line))
        return candidates[0] if candidates else ""


class Hesban11Profile(PublicationProfile):
    slug = "hesban11"
    candidate_lookahead = 2
    table_columns = HESBAN_TABLE_COLUMNS
    required_table_columns = ("table_no", "table_type")
    _caption = re.compile(
        r"(?i)\bFigure\s+([A-Za-z]?\d+(?:\.\d+)+(?:[A-Za-z])?)\s*(,\s*continued\.)?[^\n]*"
    )

    def normalize_figure(self, value: Any) -> str:
        normalized = normalize_figure_id(value)
        return normalized if re.fullmatch(
            r"[a-z]?\d+(?:\.\d+)*(?:[a-z])?", normalized) else ""

    def detect_figure_context(self, text: str) -> dict[str, str]:
        match = self._caption.search(text or "")
        if not match:
            return {"figure_id": "", "caption": "", "continued": False}
        caption = match.group(0).strip()
        return {
            "figure_id": self.normalize_figure(match.group(1)),
            "caption": caption,
            "continued": bool(match.group(2)),
        }


def get_profile(slug: str = "hesban11") -> PublicationProfile:
    if slug == Hesban11Profile.slug:
        return Hesban11Profile()
    raise MetadataLinkError(f"Unknown publication profile: {slug}")


def _atomic_json_write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp",
                                     dir=path.parent)
    temp = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def _atomic_csv_write(path: Path, frame: pd.DataFrame) -> None:
    """Replace a CSV only after the complete new file has reached disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp",
                                     dir=path.parent)
    os.close(fd)
    temp = Path(temp_name)
    try:
        with open(temp, "w", encoding="utf-8", newline="") as handle:
            frame.to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def read_json(path: Path, default: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    if not path.exists():
        return dict(default or {})
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _diagnostic_path(project_path: Path, figure_key: str, image_name: str) -> Path:
    safe_figure = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(figure_key or "figure"))
    safe_page = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(str(image_name)).stem)
    return (Path(project_path) / "cards" / DIAGNOSTICS_DIR_NAME /
            safe_figure / f"{safe_page}.json")


def externalize_linkage_diagnostics(project_path: Path,
                                    state: dict[str, Any]) -> bool:
    """Move large per-cell evidence out of the frequently rewritten state."""
    changed = False
    cards_path = Path(project_path) / "cards"
    for figure in state.get("figures", []):
        figure_key = _figure_key(figure)
        for page in figure.get("table_pages", []):
            boundary = page.get("boundary") if isinstance(page.get("boundary"), dict) else {}
            cells = boundary.pop("cell_diagnostics", None)
            ocr = page.pop("ocr_diagnostics", None)
            if cells is None and ocr is None:
                continue
            diagnostic_path = _diagnostic_path(
                project_path, figure_key, str(page.get("image_name", "page")))
            existing = read_json(diagnostic_path, {})
            payload = {
                "schema_version": LINKAGE_SCHEMA_VERSION,
                "figure_key": figure_key,
                "image_name": page.get("image_name", ""),
                "status": boundary.get("diagnostic_status", existing.get("status", {})),
                "row_anchor_conflicts": boundary.get(
                    "row_anchor_conflicts", existing.get("row_anchor_conflicts", [])),
                "cell_diagnostics": cells if cells is not None else existing.get(
                    "cell_diagnostics", []),
                "ocr_diagnostics": ocr if ocr is not None else existing.get(
                    "ocr_diagnostics", []),
                "updated_at": utc_now(),
            }
            _atomic_json_write(diagnostic_path, payload)
            page["diagnostics_ref"] = diagnostic_path.relative_to(cards_path).as_posix()
            changed = True
    return changed


def load_page_diagnostics(project_path: Path, figure: dict[str, Any],
                          image_name: str) -> dict[str, Any]:
    page = next((candidate for candidate in figure.get("table_pages", [])
                 if candidate.get("image_name") == Path(image_name).name), None)
    if not page:
        return {}
    reference = str(page.get("diagnostics_ref", ""))
    path = ((Path(project_path) / "cards" / reference) if reference else
            _diagnostic_path(project_path, _figure_key(figure), image_name))
    return read_json(path, {})


def _page_image_name(base_name: str, index: int, split_part: Optional[str]) -> str:
    return f"{base_name}_page_{index}{split_part or ''}.jpg"


def record_pdf_pages(project_path: Path, pdf_path: Path, base_name: str,
                     split_pages: bool, profile_slug: str = "hesban11",
                     render_dpi: int = 400) -> dict[str, Any]:
    """Append/replace page records after a PDF has been rendered."""
    import fitz  # PyMuPDF is already a runtime dependency of PyPotteryLens.

    project_path = Path(project_path)
    manifest_path = project_path / MANIFEST_NAME
    manifest = read_json(manifest_path, {
        "schema_version": SCHEMA_VERSION, "profile": profile_slug, "pages": []
    })
    existing_pages = sorted(manifest.get("pages", []),
                            key=lambda p: int(p.get("logical_index", 0)))
    replaced_positions = [index for index, page in enumerate(existing_pages)
                          if page.get("source_pdf") == pdf_path.name]
    insertion_index = replaced_positions[0] if replaced_positions else len(existing_pages)
    old_pages = [p for p in existing_pages if p.get("source_pdf") != pdf_path.name]
    profile = get_profile(profile_slug)
    new_pages: list[dict[str, Any]] = []

    doc = fitz.open(str(pdf_path))
    try:
        for pdf_index, page in enumerate(doc):
            text = page.get_text("text") or ""
            context = profile.detect_figure_context(text)
            parts = ["a", "b"] if split_pages else [None]
            for split_part in parts:
                new_pages.append({
                    "image_name": _page_image_name(base_name, pdf_index, split_part),
                    "source_pdf": pdf_path.name,
                    "pdf_page_index": pdf_index,
                    "printed_page": profile.detect_printed_page(text),
                    "split_part": split_part,
                    "split_side": ({"a": "left", "b": "right"}.get(split_part)
                                   if split_part else None),
                    "render_dpi": int(render_dpi),
                    "logical_index": 0,
                    "page_text": text,
                    "figure_id": context.get("figure_id", ""),
                    "figure_caption": context.get("caption", ""),
                })
    finally:
        doc.close()

    ordered_pages = old_pages[:insertion_index] + new_pages + old_pages[insertion_index:]
    for logical_index, page in enumerate(ordered_pages):
        page["logical_index"] = logical_index
    manifest.update({"schema_version": SCHEMA_VERSION, "profile": profile_slug,
                     "default_render_dpi": int(render_dpi),
                     "updated_at": utc_now(), "pages": ordered_pages})
    _atomic_json_write(manifest_path, manifest)
    return manifest


def ensure_page_manifest(project_path: Path, profile_slug: str = "hesban11",
                         source_pdf: Optional[str] = None) -> dict[str, Any]:
    project_path = Path(project_path)
    manifest_path = project_path / MANIFEST_NAME
    if manifest_path.exists():
        return read_json(manifest_path)

    all_pdfs = sorted((project_path / "pdf_source").glob("*.pdf"),
                      key=lambda p: natural_key(p.name))
    pdfs = list(all_pdfs)
    if source_pdf:
        pdfs = [p for p in pdfs if p.name == source_pdf]
    if len(pdfs) != 1:
        raise AmbiguousSourceError(
            "A page manifest could not be reconstructed automatically. Select one source PDF."
        )

    image_names = sorted(p.name for p in (project_path / "images").iterdir() if p.is_file())
    match = re.search(r"(.+)_page_\d+[ab]?\.[^.]+$", image_names[0]) if image_names else None
    if not match:
        raise MetadataLinkError("Could not infer rendered page names for this project")
    base_name = match.group(1)
    if len(all_pdfs) > 1:
        selected_stem = re.sub(r"[^a-z0-9]+", "", pdfs[0].stem.lower())
        inferred_base = re.sub(r"[^a-z0-9]+", "", base_name.lower())
        if selected_stem not in inferred_base:
            raise AmbiguousSourceError(
                "The selected PDF could not be mapped safely to the rendered image names. "
                "Create a manifest by re-rendering that source in a new project."
            )
    split_pages = any(re.search(r"_page_\d+[ab]\.", name) for name in image_names)
    manifest = record_pdf_pages(project_path, pdfs[0], base_name, split_pages, profile_slug, 300)
    # Legacy folders can contain thumbnails or an interrupted old render.  A
    # reconstructed manifest represents only images that actually exist and
    # never renames those images.
    existing = set(image_names)
    manifest["pages"] = [page for page in manifest.get("pages", [])
                         if page.get("image_name") in existing]
    if not manifest["pages"]:
        raise MetadataLinkError("No rendered PDF pages matched the inferred legacy filename pattern")
    for index, page in enumerate(manifest["pages"]):
        page["logical_index"] = index
    _atomic_json_write(manifest_path, manifest)
    return manifest


class StructuredExtractor(ABC):
    @abstractmethod
    def extract_drawing_identifiers(self, image_path: Path, cards: list[dict[str, Any]],
                                    page_context: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def extract_table(self, image_path: Path, crop: Optional[tuple[int, int, int, int]],
                      figure_id: str, expected_numbers: list[str],
                      page_context: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


def _empty_state(profile: str) -> dict[str, Any]:
    return {
        "schema_version": LINKAGE_SCHEMA_VERSION, "profile": profile, "status": "idle",
        "progress": {"current": 0, "total": 0, "message": ""},
        "figures": [], "card_index": {}, "warnings": [], "approval_history": [],
        "created_at": utc_now(), "updated_at": utc_now(),
    }


def _figure_key(figure: dict[str, Any]) -> str:
    existing = str(figure.get("figure_key", "")).strip()
    if existing:
        return existing
    masks = sorted(mask_stem(drawing.get("mask_file", ""))
                   for drawing in figure.get("drawings", []) if drawing.get("mask_file"))
    pages = sorted(str(page.get("image_name", ""))
                   for page in figure.get("drawing_pages", []) if page.get("image_name"))
    seed = "|".join(masks or pages or [str(figure.get("figure_id", "unresolved"))])
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:20]


def _warning_id(warning: dict[str, Any]) -> str:
    existing = str(warning.get("id", "")).strip()
    if existing:
        return existing
    seed = "|".join(str(warning.get(key, "")) for key in
                    ("code", "message", "page", "row", "column", "mask_file"))
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]


def _ensure_review_defaults(state: dict[str, Any]) -> dict[str, Any]:
    state["schema_version"] = LINKAGE_SCHEMA_VERSION
    state.setdefault("figures", [])
    for figure in state.get("figures", []):
        figure["figure_key"] = _figure_key(figure)
        try:
            figure["reviewer_revision"] = max(
                0, int(figure.get("reviewer_revision", 0) or 0))
        except (TypeError, ValueError):
            figure["reviewer_revision"] = 0
        figure.setdefault("warning_overrides", {})
        figure.setdefault("scale_calibrations", {})
        override_layer = figure.setdefault("review_overrides", {})
        override_layer.setdefault("cells", {})
        override_layer.setdefault("deleted", [])
        override_layer.setdefault("added", [])
        figure.setdefault("draft_saved_at", "")
        figure.setdefault("processing_status", (
            "reviewable" if figure.get("matches") or state.get("status") == "complete"
            else "processing"))
        order_figure_table_rows(figure)
    return state


def load_linkage_state(project_path: Path) -> dict[str, Any]:
    with _LINKAGE_STATE_LOCK:
        state_path = Path(project_path) / "cards" / STATE_NAME
        state = read_json(state_path, _empty_state("hesban11"))
        needs_migration = int(state.get("schema_version", 1) or 1) < LINKAGE_SCHEMA_VERSION
        inline = any(
            isinstance(page.get("boundary"), dict) and
            "cell_diagnostics" in page.get("boundary", {})
            for figure in state.get("figures", [])
            for page in figure.get("table_pages", [])
        )
        if needs_migration or inline:
            backup = state_path.with_name("metadata_linkage.v1.backup.json")
            if state_path.exists() and not backup.exists():
                shutil.copy2(state_path, backup)
            externalize_linkage_diagnostics(project_path, state)
            state["schema_version"] = LINKAGE_SCHEMA_VERSION
            state["updated_at"] = utc_now()
            _atomic_json_write(state_path, state)
    state = _ensure_review_defaults(state)
    profile = get_profile(str(state.get("profile", "hesban11")))
    for figure in state.get("figures", []):
        if figure.get("drawings") or figure.get("table_rows"):
            validate_figure(figure, profile)
    return state


def save_linkage_state(project_path: Path, state: dict[str, Any],
                       expected_revisions: Optional[dict[str, int]] = None) -> None:
    state = _ensure_review_defaults(state)
    state_path = Path(project_path) / "cards" / STATE_NAME
    # Background OCR holds an in-memory state while reviewers may autosave a
    # completed figure. Merge any newer reviewer revision before every write so
    # later progress updates cannot restore stale OCR values over manual work.
    with _LINKAGE_STATE_LOCK:
        externalize_linkage_diagnostics(project_path, state)
        current = _ensure_review_defaults(read_json(state_path, _empty_state(
            str(state.get("profile", "hesban11")))))
        current_by_key = {figure.get("figure_key"): figure
                          for figure in current.get("figures", [])}
        for figure_key, expected in (expected_revisions or {}).items():
            saved = current_by_key.get(figure_key)
            saved_revision = int(saved.get("reviewer_revision", 0) or 0) if saved else 0
            if saved_revision != int(expected):
                raise ReviewerRevisionConflict(saved or {
                    "figure_key": figure_key, "reviewer_revision": saved_revision,
                })
        profile = get_profile(str(state.get("profile", "hesban11")))
        for figure in state.get("figures", []):
            saved = current_by_key.get(figure.get("figure_key"))
            if not saved or int(saved.get("reviewer_revision", 0) or 0) <= int(
                    figure.get("reviewer_revision", 0) or 0):
                continue
            for field in REVIEWER_OWNED_FIGURE_FIELDS:
                if field in saved:
                    figure[field] = saved[field]
            saved_drawings = {str(item.get("mask_file")): item
                              for item in saved.get("drawings", [])}
            for drawing in figure.get("drawings", []):
                saved_drawing = saved_drawings.get(str(drawing.get("mask_file")))
                if saved_drawing:
                    drawing["vessel_number"] = saved_drawing.get("vessel_number", "")
                    if "measurement" in saved_drawing:
                        drawing["measurement"] = dict(saved_drawing["measurement"])
            validate_figure(figure, profile)
        card_index: dict[str, Any] = {}
        for figure in state.get("figures", []):
            # Keep persisted warnings and matches in step with edits and
            # invalidations, not only with API reads. This makes the sidecar a
            # trustworthy recovery source after a crash or page reload.
            validate_figure(figure, profile)
            matches = {str(match.get("mask_file")): match
                       for match in figure.get("matches", [])}
            for drawing in figure.get("drawings", []):
                mask_file = str(drawing.get("mask_file", ""))
                if not mask_file:
                    continue
                match = matches.get(mask_file, {})
                card_index[mask_file] = {
                    "fingerprint": drawing.get("fingerprint", ""),
                    "figure_id": figure.get("figure_id", ""),
                    "vessel_number": drawing.get("vessel_number", ""),
                    "match_status": match.get("status", "needs_review"),
                }
        state["card_index"] = card_index
        state["updated_at"] = utc_now()
        _atomic_json_write(state_path, state)


def _annotations_by_image(project_path: Path) -> dict[str, list[dict[str, Any]]]:
    annots_path = Path(project_path) / "cards" / "mask_info_annots.csv"
    if not annots_path.exists():
        raise MetadataLinkError("cards/mask_info_annots.csv was not found")
    frame = pd.read_csv(annots_path).fillna("")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for _, row in frame.iterrows():
        stem = mask_stem(row["mask_file"])
        image_name = stem.split("_mask_layer_")[0]
        bbox = parse_bbox(row["bbox"])
        grouped.setdefault(image_name, []).append({
            "mask_file": stem,
            "bbox": list(bbox),
            "fingerprint": bbox_fingerprint(image_name, bbox),
        })
    return grouped


def validate_figure(figure: dict[str, Any], profile: PublicationProfile) -> dict[str, Any]:
    warnings: list[dict[str, Any]] = [
        dict(warning) for warning in figure.get("extraction_warnings", [])
        if warning.get("code") not in HIDDEN_REVIEW_WARNING_CODES
    ]
    drawing_sources = {str(page.get("source_pdf", "")).strip()
                       for page in figure.get("drawing_pages", [])
                       if str(page.get("source_pdf", "")).strip()}
    table_sources = {str(page.get("source_pdf", "")).strip()
                     for page in figure.get("table_pages", [])
                     if str(page.get("source_pdf", "")).strip()}
    if drawing_sources and table_sources and not table_sources.issubset(drawing_sources):
        warnings.append({
            "code": "cross_pdf_assignment",
            "message": "Drawing and table pages come from different source PDFs.",
        })
    previous_matches = {
        (str(match.get("mask_file", "")), str(match.get("fingerprint", ""))): match
        for match in figure.get("matches", [])
    }
    drawings = figure.get("drawings", [])
    rows = figure.get("table_rows", [])
    drawing_map: dict[str, list[dict[str, Any]]] = {}
    row_map: dict[str, list[dict[str, Any]]] = {}
    for drawing in drawings:
        number = profile.normalize_number(drawing.get("vessel_number"))
        number = number if is_positive_vessel_number(number) else ""
        drawing["vessel_number"] = number
        if number:
            drawing_map.setdefault(number, []).append(drawing)
        else:
            warnings.append({"code": "missing_drawing_number",
                             "mask_file": drawing.get("mask_file", ""),
                             "message": f"No valid positive printed number for {drawing.get('mask_file')}"})
    for row in rows:
        number = profile.normalize_number(row.get("table_no"))
        number = number if is_positive_vessel_number(number) else ""
        row["normalized_table_no"] = number
        if number:
            row_map.setdefault(number, []).append(row)
        else:
            warnings.append({"code": "missing_table_number", "row": "",
                             "message": "A table row has no number"})
        for column in profile.required_table_columns:
            if column != "table_no" and not str(row.get(column, "")).strip():
                warnings.append({
                    "code": "missing_required_value",
                    "row": number, "column": column,
                    "message": f"Table row {number or '?'} is missing required field {column}",
                })
    for number, values in drawing_map.items():
        if len(values) > 1:
            warnings.append({"code": "duplicate_drawing_number", "row": number,
                             "message": f"Drawing number {number} occurs {len(values)} times"})
    for number, values in row_map.items():
        if len(values) > 1:
            warnings.append({"code": "duplicate_table_number", "row": number,
                             "message": f"Table number {number} occurs {len(values)} times"})

    matches: list[dict[str, Any]] = []
    for drawing in drawings:
        number = drawing.get("vessel_number", "")
        candidates = row_map.get(number, []) if number else []
        status = "ready" if len(drawing_map.get(number, [])) == 1 and len(candidates) == 1 else "needs_review"
        if number and not candidates:
            warnings.append({"code": "missing_table_row", "row": number,
                             "message": f"No table row for drawing {number}"})
        match = {
            "mask_file": drawing.get("mask_file"), "fingerprint": drawing.get("fingerprint"),
            "vessel_number": number, "status": status,
            "values": dict(candidates[0]) if status == "ready" else {},
        }
        previous = previous_matches.get((str(match["mask_file"]), str(match["fingerprint"])), {})
        if previous.get("applied_values"):
            match["applied_values"] = dict(previous["applied_values"])
        matches.append(match)
    for number in sorted(set(row_map) - set(drawing_map), key=natural_key):
        warnings.append({"code": "unexpected_table_row", "row": number,
                         "message": f"Table row {number} has no drawing"})

    overrides = figure.setdefault("warning_overrides", {})
    unique_warnings: list[dict[str, Any]] = []
    seen_warning_ids: set[str] = set()
    for warning in warnings:
        warning["id"] = _warning_id(warning)
        if warning["id"] in seen_warning_ids:
            continue
        seen_warning_ids.add(warning["id"])
        warning["overrideable"] = warning.get("code") in OVERRIDABLE_WARNING_CODES
        override = overrides.get(warning["id"])
        warning["overridden"] = bool(
            warning["overrideable"] and isinstance(override, dict) and
            override.get("reason") in WARNING_OVERRIDE_REASONS.get(
                str(warning.get("code", "")), set()))
        warning["blocking"] = bool(
            warning.get("code") not in NON_BLOCKING_WARNING_CODES and
            not warning["overridden"])
        unique_warnings.append(warning)

    figure["matches"] = matches
    figure["warnings"] = unique_warnings
    # Any unmatched/extra/unnumbered data makes the proposed join ambiguous.  Do
    # not present it as approvable merely because every drawing happened to find
    # one row; extra rows are a common sign of a wrong or truncated table page.
    blocking_warnings = [warning for warning in unique_warnings if warning["blocking"]]
    figure["status"] = ("ready" if matches and
                        all(m["status"] == "ready" for m in matches) and
                        not blocking_warnings else "needs_review")
    if (figure.get("processing_status") not in {"processing", "queued"} and
            figure.get("review_status") != "approved"):
        figure["processing_status"] = (
            "ready" if figure["status"] == "ready" else "reviewable")
    return figure


def apply_reviewer_row_overrides(figure: dict[str, Any],
                                 profile: PublicationProfile) -> list[str]:
    """Reapply edited cells and row additions/deletions after fresh OCR."""
    layer = figure.get("review_overrides", {})
    cells = layer.get("cells", {}) if isinstance(layer.get("cells"), dict) else {}
    deleted = {str(value) for value in layer.get("deleted", [])}
    added = layer.get("added", []) if isinstance(layer.get("added"), list) else []
    matched: set[str] = set()
    kept_rows = []
    seen_manual: set[str] = set()
    for index, row in enumerate(figure.get("table_rows", [])):
        number = profile.normalize_number(row.get("table_no")) or f"row-{index}"
        key = f"{row.get('source_image', '')}|{number}"
        if key in deleted:
            matched.add(key)
            continue
        values = cells.get(key)
        if isinstance(values, dict):
            for column, value in values.items():
                if column in HESBAN_TABLE_COLUMNS:
                    row[column] = "" if value is None else str(value)
            matched.add(key)
        kept_rows.append(row)
    for saved in added:
        if not isinstance(saved, dict):
            continue
        manual_id = str(saved.get("review_override_id", ""))
        manual_key = f"manual:{manual_id}" if manual_id else ""
        if manual_key and manual_key in seen_manual:
            continue
        candidate = dict(saved)
        number = profile.normalize_number(candidate.get("table_no"))
        source = str(candidate.get("source_image", ""))
        # If a later OCR pass finds the formerly missing row, keep the OCR
        # provenance and layer the researcher's values onto that row.
        existing = next((row for row in kept_rows
                         if profile.normalize_number(row.get("table_no")) == number
                         and (not source or row.get("source_image", "") == source)), None)
        if existing is not None:
            for column in HESBAN_TABLE_COLUMNS:
                existing[column] = "" if candidate.get(column) is None else str(
                    candidate.get(column, ""))
            if manual_id:
                existing["review_override_id"] = manual_id
                existing["review_override_source_key"] = (
                    f"{existing.get('source_image', '')}|{number}")
        else:
            kept_rows.append(candidate)
        if manual_key:
            seen_manual.add(manual_key)
    figure["table_rows"] = kept_rows
    unmatched = (set(cells) | deleted) - matched
    return sorted(unmatched)


class MetadataLinker:
    def __init__(self, project_path: Path, extractor: StructuredExtractor,
                 profile: Optional[PublicationProfile] = None,
                 source_pdf: Optional[str] = None):
        self.project_path = Path(project_path)
        self.extractor = extractor
        self.profile = profile or Hesban11Profile()
        self.source_pdf = source_pdf

    def run(self, progress: Optional[Callable[[int, int, str], None]] = None,
            resume: bool = False,
            should_pause: Optional[Callable[[], bool]] = None) -> dict[str, Any]:
        previous_state = load_linkage_state(self.project_path)
        previous_figures = {
            self.profile.normalize_figure(figure.get("figure_id")): figure
            for figure in previous_state.get("figures", [])
        }
        manifest = ensure_page_manifest(self.project_path, self.profile.slug, self.source_pdf)
        all_pages = manifest.get("pages", [])
        sources = {str(page.get("source_pdf", "")) for page in all_pages}
        if self.source_pdf:
            pages = [page for page in all_pages if page.get("source_pdf") == self.source_pdf]
            if not pages:
                raise MetadataLinkError(f"Source PDF was not found in the page manifest: {self.source_pdf}")
        else:
            if len(sources) > 1:
                raise AmbiguousSourceError("Select one source PDF before running metadata linking.")
            pages = list(all_pages)
        pages = sorted(pages, key=lambda p: int(p.get("logical_index", 0)))
        page_by_stem = {Path(p["image_name"]).stem: p for p in pages}
        page_pos = {Path(p["image_name"]).stem: i for i, p in enumerate(pages)}
        grouped = _annotations_by_image(self.project_path)
        drawing_pages = sorted(grouped, key=lambda name: page_pos.get(name, 10**9))
        can_resume = bool(
            resume and previous_state.get("status") in {"running", "paused", "error"}
            and previous_state.get("run_source_pdf", self.source_pdf or "") ==
            (self.source_pdf or ""))
        state = previous_state if can_resume else _empty_state(self.profile.slug)
        state["approval_history"] = list(previous_state.get("approval_history", []))
        state["status"] = "running"
        state["run_source_pdf"] = self.source_pdf or ""
        state["progress"] = {**state.get("progress", {}),
                             "total": len(drawing_pages), "message": "Starting"}
        save_linkage_state(self.project_path, state)

        figures: dict[str, dict[str, Any]] = {
            self.profile.normalize_figure(figure.get("figure_id")): figure
            for figure in state.get("figures", [])
        } if can_resume else {}
        processed_drawing_pages = {
            page.get("image_name") for figure in figures.values()
            for page in figure.get("drawing_pages", [])
        }
        for index, image_stem in enumerate(drawing_pages, 1):
            page = page_by_stem.get(image_stem)
            if not page:
                state["warnings"].append({"code": "page_not_in_manifest", "message": image_stem})
                continue
            if page.get("image_name") in processed_drawing_pages:
                continue
            image_path = self.project_path / "images" / page["image_name"]
            context = self.profile.detect_figure_context(page.get("page_text", ""))
            context.update(page)
            result = self.extractor.extract_drawing_identifiers(image_path, grouped[image_stem], context) or {}
            figure_id = self.profile.normalize_figure(result.get("figure_id") or context.get("figure_id"))
            if not figure_id:
                figure_id = f"unresolved-{page.get('logical_index', index)}"
            previous_figure = previous_figures.get(figure_id, {})
            figure = figures.setdefault(figure_id, {
                "figure_id": figure_id,
                "figure_caption": result.get("figure_caption") or context.get("caption") or page.get("figure_caption", ""),
                "drawing_pages": [], "table_pages": [], "drawings": [], "table_rows": [],
                "warnings": [], "review_status": "pending",
                "review_history": list(previous_figure.get("review_history", [])),
                # validate_figure carries applied_values forward only when both
                # card identity and geometry fingerprint are unchanged.
                "matches": list(previous_figure.get("matches", [])),
                "figure_key": previous_figure.get("figure_key", ""),
                # Drawing-page OCR discovers figures before their table pass
                # begins.  Keep them queued until the loop below actively
                # extracts that one figure, otherwise every discovered figure
                # appears to be processing at the same time in the reviewer UI.
                "processing_status": "queued",
                "reviewer_revision": int(previous_figure.get("reviewer_revision", 0) or 0),
                "warning_overrides": dict(previous_figure.get("warning_overrides", {})),
                "scale_calibrations": dict(previous_figure.get("scale_calibrations", {})),
                "review_overrides": dict(previous_figure.get(
                    "review_overrides", {"cells": {}})),
                "draft_saved_at": previous_figure.get("draft_saved_at", ""),
            })
            figure["drawing_pages"].append({
                "image_name": page["image_name"], "logical_index": page["logical_index"],
                "printed_page": result.get("printed_page") or page.get("printed_page", ""),
                "source_pdf": page.get("source_pdf", ""),
                "render_dpi": page.get("render_dpi"),
                "pdf_page_index": page.get("pdf_page_index"),
                "split_part": page.get("split_part"),
            })
            returned = result.get("drawings", {})
            previous_drawings = {
                (mask_stem(item.get("mask_file", "")), str(item.get("fingerprint", ""))): item
                for item in previous_figure.get("drawings", [])
            }
            for card in grouped[image_stem]:
                value = returned.get(card["mask_file"], {}) if isinstance(returned, dict) else {}
                number = value.get("number") if isinstance(value, dict) else value
                normalized_number = self.profile.normalize_number(number)
                drawing = {**card, "image_name": page["image_name"],
                           "vessel_number": (normalized_number
                                             if is_positive_vessel_number(normalized_number) else "")}
                previous_drawing = previous_drawings.get(
                    (mask_stem(card.get("mask_file", "")), str(card.get("fingerprint", ""))))
                if previous_drawing and previous_drawing.get("measurement"):
                    drawing["measurement"] = dict(previous_drawing["measurement"])
                figure["drawings"].append(drawing)

            state["progress"] = {"current": index, "total": len(drawing_pages),
                                 "message": f"Reading drawing page {index}/{len(drawing_pages)}"}
            save_linkage_state(self.project_path, {**state, "figures": list(figures.values())})
            if progress:
                progress(index, len(drawing_pages), state["progress"]["message"])
            if should_pause and should_pause():
                state["status"] = "paused"
                state["figures"] = list(figures.values())
                state["progress"]["message"] = "Paused for priority work"
                save_linkage_state(self.project_path, state)
                return state

        # Extract candidate table pages once per figure after all continued drawing pages are grouped.
        total_work = len(drawing_pages) + len(figures)
        state["progress"] = {"current": len(drawing_pages), "total": total_work,
                             "message": "Starting table extraction"}
        for queued_figure in figures.values():
            if not can_resume or queued_figure.get("processing_status") in {"processing", "queued"}:
                queued_figure["processing_status"] = "queued"
        save_linkage_state(self.project_path, {**state, "figures": list(figures.values())})
        for figure_index, figure in enumerate(figures.values(), 1):
            if (can_resume and figure.get("processing_status") not in
                    {"processing", "queued"}):
                continue
            if can_resume and figure.get("processing_status") == "processing":
                figure["table_pages"] = []
                figure["table_rows"] = []
                figure["extraction_warnings"] = []
            figure["processing_status"] = "processing"
            state["progress"] = {
                "current": len(drawing_pages) + figure_index - 1,
                "total": total_work,
                "message": f"Reading figure {figure.get('figure_id')} ({figure_index}/{len(figures)})",
            }
            save_linkage_state(self.project_path, {**state, "figures": list(figures.values())})
            expected = sorted({d["vessel_number"] for d in figure["drawings"] if d.get("vessel_number")}, key=natural_key)
            positions = [page_pos.get(Path(p["image_name"]).stem) for p in figure["drawing_pages"]]
            positions = [p for p in positions if p is not None]
            if not positions:
                continue
            start, end = min(positions), max(positions) + self.profile.candidate_lookahead
            seen_rows: set[tuple[str, str]] = set()
            max_drawing_bottom: dict[str, int] = {}
            for drawing in figure["drawings"]:
                max_drawing_bottom[drawing["image_name"]] = max(
                    max_drawing_bottom.get(drawing["image_name"], 0), int(drawing["bbox"][3]))

            for pos in range(start, min(end + 1, len(pages))):
                candidate = pages[pos]
                candidate_context = self.profile.detect_figure_context(candidate.get("page_text", ""))
                explicit = candidate_context.get("figure_id", "")
                if pos > max(positions) and explicit and explicit != figure["figure_id"]:
                    break
                image_path = self.project_path / "images" / candidate["image_name"]
                crop = None
                bottom = max_drawing_bottom.get(candidate["image_name"])
                if bottom:
                    from PIL import Image
                    with Image.open(image_path) as image:
                        crop = (0, min(image.height, bottom + 20), image.width, image.height)
                saved_page = next((item for item in previous_figures.get(
                    figure["figure_id"], {}).get("table_pages", [])
                    if item.get("image_name") == candidate["image_name"]), {})
                table = self.extractor.extract_table(image_path, crop, figure["figure_id"], expected,
                                                     {**candidate, **candidate_context,
                                                      "manual_column_edges": saved_page.get(
                                                          "manual_column_edges")}) or {}
                rows = table.get("rows", []) if table.get("is_table", bool(table.get("rows"))) else []
                normalized_rows = []
                for row in rows:
                    clean = {column: "" if row.get(column) is None else str(row.get(column))
                             for column in self.profile.table_columns}
                    clean["normalized_table_no"] = self.profile.normalize_number(clean.get("table_no"))
                    clean["source_image"] = candidate["image_name"]
                    clean["source_printed_page"] = table.get("printed_page") or candidate.get("printed_page", "")
                    key = (clean["normalized_table_no"], candidate["image_name"])
                    if clean["normalized_table_no"] and key not in seen_rows:
                        seen_rows.add(key)
                        normalized_rows.append(clean)
                overlap = bool(set(expected) & {r["normalized_table_no"] for r in normalized_rows})
                raw_table_caption = str(table.get("figure_caption") or "")
                visual_context = self.profile.detect_figure_context(raw_table_caption)
                # A model echoing the requested target ID is not evidence of an
                # explicit caption. Searchable text or a parseable raw caption is.
                table_figure = self.profile.normalize_figure(
                    explicit or visual_context.get("figure_id"))
                if table_figure and table_figure != figure["figure_id"]:
                    warning = {
                        "code": "conflicting_figure_caption",
                        "message": (f"Table page {candidate['image_name']} names figure "
                                    f"{table_figure}, not {figure['figure_id']}"),
                    }
                    if warning not in figure.setdefault("extraction_warnings", []):
                        figure["extraction_warnings"].append(warning)
                needs_columns = bool(table.get("needs_column_review"))
                accepted = ((bool(normalized_rows) and
                             (table_figure == figure["figure_id"] or
                              (not table_figure and overlap))) or
                            (needs_columns and
                             (not table_figure or table_figure == figure["figure_id"])))
                if accepted:
                    boundary = table.get("boundary") if isinstance(table.get("boundary"), dict) else {}
                    figure["table_pages"].append({
                        "image_name": candidate["image_name"], "logical_index": candidate["logical_index"],
                        "printed_page": table.get("printed_page") or candidate.get("printed_page", ""),
                        "source_pdf": candidate.get("source_pdf", ""), "crop": list(crop) if crop else None,
                        "figure_caption": raw_table_caption or candidate.get("figure_caption", ""),
                        "boundary": boundary,
                        "manual_column_edges": saved_page.get("manual_column_edges"),
                        "ocr_diagnostics": [
                            item for item in table.get("ocr_diagnostics", [])
                            if isinstance(item, dict)
                        ],
                    })
                    figure["table_rows"].extend(normalized_rows)
                    for warning in table.get("warnings", []):
                        if isinstance(warning, dict) and warning not in figure.setdefault("extraction_warnings", []):
                            figure["extraction_warnings"].append(warning)
                    # A closing rule is authoritative: this table cannot spill
                    # into later candidate pages.
                    if boundary.get("has_closing_rule"):
                        break

            # An unclosed page is valid only when the immediately following
            # accepted page has another verified Hesban header.
            accepted_pages = figure.get("table_pages", [])
            for page_index, table_page in enumerate(accepted_pages):
                boundary = table_page.get("boundary", {})
                # Legacy/optional structured extractors do not emit geometric
                # evidence. Preserve their existing review workflow; the local
                # OCR backend always supplies and enforces these boundaries.
                if not boundary:
                    continue
                if boundary.get("has_closing_rule"):
                    boundary["continues"] = False
                    continue
                next_page = accepted_pages[page_index+1] if page_index+1 < len(accepted_pages) else None
                adjacent = bool(next_page and
                                int(next_page.get("logical_index", -2)) == int(table_page.get("logical_index", -1)) + 1)
                repeated_header = bool(next_page and next_page.get("boundary", {}).get("header_confirmed"))
                if adjacent and repeated_header:
                    boundary["continues"] = True
                else:
                    warning = {
                        "code": "missing_table_end",
                        "message": f"Table page {table_page.get('image_name')} has no closing rule or verified continuation.",
                    }
                    if warning not in figure.setdefault("extraction_warnings", []):
                        figure["extraction_warnings"].append(warning)

            # A dense table response may be truncated. Retry only the still-missing
            # expected numbers against accepted table pages, never against unrelated pages.
            found_numbers = {self.profile.normalize_number(row.get("table_no", ""))
                             for row in figure["table_rows"]}
            missing = [number for number in expected if number not in found_numbers]
            if missing:
                page_lookup = {p["image_name"]: p for p in pages}
                for table_page in list(figure["table_pages"]):
                    candidate = page_lookup.get(table_page["image_name"])
                    if not candidate or not missing:
                        continue
                    retry = self.extractor.extract_table(
                        self.project_path / "images" / candidate["image_name"],
                        tuple(table_page["crop"]) if table_page.get("crop") else None,
                        figure["figure_id"], missing, {**candidate, "retry_missing": True}) or {}
                    for warning in retry.get("warnings", []):
                        if isinstance(warning, dict) and warning not in figure.setdefault("extraction_warnings", []):
                            figure["extraction_warnings"].append(warning)
                    for row in retry.get("rows", []):
                        clean = {column: "" if row.get(column) is None else str(row.get(column))
                                 for column in self.profile.table_columns}
                        normalized_number = self.profile.normalize_number(clean.get("table_no"))
                        clean["normalized_table_no"] = normalized_number
                        if normalized_number not in missing:
                            continue
                        clean["source_image"] = candidate["image_name"]
                        clean["source_printed_page"] = retry.get("printed_page") or candidate.get("printed_page", "")
                        figure["table_rows"].append(clean)
                        missing.remove(normalized_number)
            try:
                from catalog.measurements import measure_figure
                figure_sources = {page.get("source_pdf") for page in figure.get("drawing_pages", [])}
                figure_dpis = {page.get("render_dpi") for page in figure.get("drawing_pages", [])
                               if page.get("render_dpi") is not None}
                previous_ratios = [
                    calibration.get("px_per_cm")
                    for candidate in figures.values()
                    if candidate is not figure
                    if ({page.get("source_pdf") for page in candidate.get("drawing_pages", [])} &
                        figure_sources)
                    if ((not figure_dpis and not {page.get("render_dpi") for page in
                                                  candidate.get("drawing_pages", [])
                                                  if page.get("render_dpi") is not None}) or
                        {page.get("render_dpi") for page in candidate.get("drawing_pages", [])
                         if page.get("render_dpi") is not None} & figure_dpis)
                    for calibration in candidate.get("scale_calibrations", {}).values()
                    if (calibration.get("status") in {"suggested", "verified", "verified_automatic", "verified_manual"} and
                        calibration.get("px_per_cm"))
                ]
                measure_figure(self.project_path, figure, project_ratios=previous_ratios)
            except Exception as exc:
                figure.setdefault("measurement_warnings", []).append({
                    "code": "measurement_failed", "message": str(exc),
                })
            unmatched_overrides = apply_reviewer_row_overrides(figure, self.profile)
            for override_key in unmatched_overrides:
                figure.setdefault("extraction_warnings", []).append({
                    "code": "review_override_unmatched",
                    "message": f"A saved table correction no longer matches an OCR row: {override_key}",
                })
            validate_figure(figure, self.profile)
            figure["processing_status"] = (
                "ready" if figure.get("status") == "ready" else "reviewable")
            current = len(drawing_pages) + figure_index
            state["progress"] = {
                "current": current, "total": total_work,
                "message": f"Finished figure {figure.get('figure_id')} ({figure_index}/{len(figures)})",
            }
            save_linkage_state(self.project_path, {**state, "figures": list(figures.values())})
            if progress:
                progress(current, total_work, state["progress"]["message"])
            if should_pause and should_pause():
                state["figures"] = list(figures.values())
                state["status"] = "paused"
                state["progress"]["message"] = "Paused for priority work"
                save_linkage_state(self.project_path, state)
                return state

        state["figures"] = sorted(figures.values(), key=lambda f: natural_key(f["figure_id"]))
        state["status"] = "complete"
        state["progress"] = {"current": total_work, "total": total_work, "message": "Complete"}
        save_linkage_state(self.project_path, state)
        return state


def linkage_totals(state: dict[str, Any]) -> dict[str, int]:
    figures = state.get("figures", [])
    matches = [m for f in figures for m in f.get("matches", [])]
    return {
        "linkage_figures": len(figures),
        "linkage_ready": sum(m.get("status") == "ready" for m in matches),
        "linkage_reviewed": sum(len(f.get("matches", [])) for f in figures
                                if f.get("review_status") == "approved"),
        "linkage_unresolved": sum(m.get("status") != "ready" for m in matches),
    }


def apply_approved_figures(project_path: Path, figure_ids: Iterable[str],
                           replace_imported: bool = False) -> dict[str, Any]:
    # Approval changes both the CSV and the review sidecar. Hold the same
    # in-process lock used by autosaves so a reviewer edit cannot land between
    # validation, CSV materialization, and the approval revision update.
    with _LINKAGE_STATE_LOCK:
        return _apply_approved_figures_locked(
            project_path, figure_ids, replace_imported)


def _apply_approved_figures_locked(project_path: Path, figure_ids: Iterable[str],
                                   replace_imported: bool = False) -> dict[str, Any]:
    project_path = Path(project_path)
    state = load_linkage_state(project_path)
    requested = {normalize_figure_id(value) for value in figure_ids if normalize_figure_id(value)}
    if not requested:
        raise MetadataLinkError("No figures were selected for approval")
    info_path = project_path / "cards" / "mask_info.csv"
    if not info_path.exists():
        raise MetadataLinkError("cards/mask_info.csv was not found")
    frame = pd.read_csv(info_path, dtype=str, keep_default_na=False)
    if "mask_file" not in frame.columns:
        raise MetadataLinkError("cards/mask_info.csv has no mask_file column")
    frame = migrate_linkage_columns(frame, drop_legacy=True)
    for column in PUBLIC_LINKAGE_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""

    figures_by_id = {normalize_figure_id(figure.get("figure_id")): figure
                     for figure in state.get("figures", [])}
    unknown = sorted(requested - set(figures_by_id), key=natural_key)
    if unknown:
        raise MetadataLinkError(f"Unknown figure(s): {', '.join(unknown)}")
    expected_revisions = {
        str(figures_by_id[figure_id].get("figure_key", "")):
        int(figures_by_id[figure_id].get("reviewer_revision", 0) or 0)
        for figure_id in requested
    }

    # Validate every requested figure and target before changing either the CSV
    # or review state, so a bad later figure cannot create a partial approval.
    targets: dict[tuple[str, str], pd.Series] = {}
    for figure_id in requested:
        figure = figures_by_id[figure_id]
        if figure.get("processing_status") in {"processing", "queued"}:
            raise MetadataLinkError(
                f"Figure {figure.get('figure_id')} is still being extracted or waiting to start")
        if figure.get("status") != "ready":
            raise MetadataLinkError(f"Figure {figure.get('figure_id')} still requires review")
        validate_figure(figure, get_profile(state.get("profile", "hesban11")))
        if figure.get("status") != "ready":
            raise MetadataLinkError(f"Figure {figure.get('figure_id')} still has unresolved matches")
        for match in figure.get("matches", []):
            target = frame["mask_file"].map(mask_stem) == mask_stem(match.get("mask_file"))
            count = int(target.sum())
            if count != 1:
                reason = "was not found" if count == 0 else f"matched {count} CSV rows"
                raise MetadataLinkError(f"Card {match.get('mask_file')} {reason}")
            targets[(figure_id, str(match.get("mask_file")))] = target

    applied = 0
    for figure_id in requested:
        figure = figures_by_id[figure_id]
        drawing_pages = figure.get("drawing_pages", [])
        table_pages = figure.get("table_pages", [])
        has_review_overrides = any(warning.get("overridden")
                                   for warning in figure.get("warnings", []))
        provenance = {
            "figure_id": figure.get("figure_id", ""),
            "figure_caption": figure.get("figure_caption", ""),
            "drawing_printed_page": "; ".join(dict.fromkeys(str(p.get("printed_page", "")) for p in drawing_pages if p.get("printed_page"))),
            "table_printed_pages": "; ".join(dict.fromkeys(str(p.get("printed_page", "")) for p in table_pages if p.get("printed_page"))),
            "source_pdf": "; ".join(dict.fromkeys(str(p.get("source_pdf", "")) for p in drawing_pages + table_pages if p.get("source_pdf"))),
            "link_status": ("approved_with_overrides" if has_review_overrides
                            else "approved"),
        }
        for match in figure.get("matches", []):
            target = targets[(figure_id, str(match.get("mask_file")))]
            drawing = next((item for item in figure.get("drawings", [])
                            if mask_stem(item.get("mask_file")) ==
                            mask_stem(match.get("mask_file"))), {})
            measurement = drawing.get("measurement", {})
            try:
                verified_number = float(measurement.get("verified_cm"))
            except (TypeError, ValueError):
                verified_number = 0.0
            measurement_status = str(measurement.get("status", ""))
            verified = (measurement_status in {"verified", "verified_automatic", "verified_manual"} and
                        math.isfinite(verified_number) and verified_number > 0)
            verified_value = verified_number if verified else ""
            internal_values = {**provenance, **match.get("values", {}),
                               "vessel_number": match.get("vessel_number", ""),
                               "rim_diameter_cm": (
                                   f"{float(verified_value):.1f}" if verified_value != "" else ""),
                               "diameter_status": (
                                   ("verified_automatic" if measurement_status == "verified_automatic"
                                    else "verified_manual") if verified else "unresolved"),
                               }
            values = {public: str(internal_values.get(internal, "") or "")
                      for internal, public in PUBLIC_LINKAGE_MAP.items()}
            previously_imported_raw = match.get("applied_values", {})
            previously_imported = {
                PUBLIC_LINKAGE_MAP.get(column, column): value
                for column, value in previously_imported_raw.items()
                if PUBLIC_LINKAGE_MAP.get(column, column) in PUBLIC_LINKAGE_COLUMNS
            }
            newly_imported: dict[str, str] = {}
            for column in PUBLIC_LINKAGE_COLUMNS:
                new_value = "" if values.get(column) is None else str(values.get(column, ""))
                old_values = frame.loc[target, column]
                prior_value = str(previously_imported.get(column, ""))
                # A value is feature-owned only while the CSV still equals the
                # exact value this feature last wrote. A later manual correction
                # must survive a rerun even when link_status remains approved.
                may_replace = replace_imported and old_values.eq(prior_value) if column in previously_imported else False
                write_mask = old_values.eq("") | may_replace
                frame.loc[target & write_mask.reindex(frame.index, fill_value=False), column] = new_value
                if bool(write_mask.any()):
                    newly_imported[column] = new_value
            match["applied_values"] = {**previously_imported, **newly_imported}
            applied += int(target.sum())
        figure["review_status"] = "approved"
        figure["processing_status"] = "approved"
        figure["reviewer_revision"] = (
            int(figure.get("reviewer_revision", 0) or 0) + 1)
        figure["draft_saved_at"] = utc_now()
        figure["approved_at"] = utc_now()
        figure.setdefault("review_history", []).append({
            "action": "approved", "at": figure["approved_at"],
            "replace_imported": bool(replace_imported),
        })
    state.setdefault("approval_history", []).append({
        "at": utc_now(), "figure_ids": sorted(requested, key=natural_key),
        "applied_rows": applied, "replace_imported": bool(replace_imported),
    })
    _atomic_csv_write(info_path, frame)
    save_linkage_state(project_path, state, expected_revisions=expected_revisions)
    return {"applied_rows": applied, "totals": linkage_totals(state)}


def invalidate_linkage_for_card_changes(cards_path: Path, old_annots: Optional[pd.DataFrame],
                                        new_annots: pd.DataFrame) -> None:
    cards_path = Path(cards_path)
    state_path = cards_path / STATE_NAME
    if not state_path.exists() or old_annots is None:
        return
    old_map = {mask_stem(row["mask_file"]): parse_bbox(row["bbox"])
               for _, row in old_annots.iterrows()}
    new_rows = {mask_stem(row["mask_file"]): row for _, row in new_annots.iterrows()}
    new_map = {key: parse_bbox(row["bbox"]) for key, row in new_rows.items()}
    changed = {key for key in set(old_map) | set(new_map) if old_map.get(key) != new_map.get(key)}
    if not changed:
        return
    state = read_json(state_path)
    for figure in state.get("figures", []):
        affected = False
        for drawing in figure.get("drawings", []):
            key = mask_stem(drawing.get("mask_file"))
            if key not in changed or key not in new_rows:
                continue
            bbox = parse_bbox(new_rows[key]["bbox"])
            drawing["bbox"] = list(bbox)
            drawing["fingerprint"] = bbox_fingerprint(drawing.get("image_name", ""), bbox)
            drawing["measurement"] = {
                "status": "unresolved", "warning": "card_geometry_changed",
                "drawing_fingerprint": drawing["fingerprint"],
            }
        for match in figure.get("matches", []):
            if mask_stem(match.get("mask_file")) in changed:
                match["status"] = "needs_review"
                match["values"] = {}
                affected = True
        if affected:
            figure["status"] = "needs_review"
            figure["review_status"] = "pending"
            figure.pop("approved_at", None)
            persistent_warning = {
                "code": "card_geometry_changed",
                "message": "Card extraction changed; review this figure again.",
            }
            extraction_warnings = figure.setdefault("extraction_warnings", [])
            if not any(warning.get("code") == "card_geometry_changed"
                       for warning in extraction_warnings):
                extraction_warnings.append(persistent_warning)
    save_linkage_state(cards_path.parent, state)
