"""Research-facing CSV and dataset package generation.

The working ``mask_info.csv`` intentionally retains private identifiers needed by
the editor.  This module is the boundary that turns it into a stable, readable
research dataset without exposing those implementation details.
"""

from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO, StringIO
import csv
import json
from pathlib import Path
import re
import zipfile

import pandas as pd

from catalog.linkage import PUBLIC_LINKAGE_MAP, load_linkage_state, migrate_linkage_columns
from catalog.profiles import HESBAN_COLUMN_SPECS


EXPORT_SCHEMA_VERSION = 1
EXPORT_SETTINGS_NAME = "export_settings.json"

EXPORT_COLUMNS = [
    "Image Filename", "Figure", "No.", "Vessel Type", "Rim Diameter (cm)",
] + [spec.csv_label for spec in HESBAN_COLUMN_SPECS[2:]]

SOURCE_COLUMNS = {
    "Figure": "Figure",
    "No.": "No.",
    "Vessel Type": "Type",
    "Rim Diameter (cm)": "Rim Diameter (cm)",
}
SOURCE_COLUMNS.update({
    spec.csv_label: PUBLIC_LINKAGE_MAP[spec.key]
    for spec in HESBAN_COLUMN_SPECS[2:]
})

DATA_DICTIONARY = {
    "Image Filename": "Filename of the matching vessel mask in the images folder.",
    "Figure": "Printed publication figure identifier.",
    "No.": "Vessel number shared by the drawing and table row.",
    "Vessel Type": "Vessel type as printed in the publication table.",
    "Rim Diameter (cm)": "Measured illustrated rim diameter in centimetres.",
}


def _clean_text(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value)
    return "" if text in {"nan", "None"} else text


def _safe_piece(value: str, fallback: str) -> str:
    text = re.sub(r"[^A-Za-z0-9-]+", "-", _clean_text(value).replace(".", "-"))
    return text.strip("-") or fallback


def _mask_stem(value: str) -> str:
    return Path(_clean_text(value)).stem


def _resolve_card(cards_path: Path, mask_file: str) -> Path | None:
    name = Path(_clean_text(mask_file)).name
    direct = cards_path / name
    if direct.is_file() and direct.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        return direct
    stem = Path(name).stem
    for suffix in (".png", ".jpg", ".jpeg"):
        candidate = cards_path / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def load_export_settings(project_path: Path) -> dict:
    """Load settings and migrate the former Post Processing exclusions once."""
    path = project_path / EXPORT_SETTINGS_NAME
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
    else:
        data = {}
    excluded = {_mask_stem(item) for item in data.get("excluded_masks", []) if item}
    legacy = project_path / "cards_modified" / "excluded_cards.json"
    if not data.get("legacy_exclusions_imported") and legacy.exists():
        try:
            legacy_data = json.loads(legacy.read_text(encoding="utf-8"))
            values = legacy_data if isinstance(legacy_data, list) else legacy_data.get("excluded", [])
            excluded.update(_mask_stem(item) for item in values if item)
        except (OSError, json.JSONDecodeError, AttributeError):
            pass
        data["legacy_exclusions_imported"] = True
    data.update({
        "schema_version": EXPORT_SCHEMA_VERSION,
        "excluded_masks": sorted(excluded),
    })
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return data


def save_export_settings(project_path: Path, excluded_masks: list[str],
                         known_masks: list[str] | None = None) -> dict:
    settings = load_export_settings(project_path)
    submitted = {_mask_stem(item) for item in excluded_masks if item}
    if known_masks is None:
        merged = submitted
    else:
        # The Export page only lists currently approved cards.  Preserve older
        # exclusions for unresolved/not-yet-approved cards that were not in
        # that visible set, so a routine checkbox edit cannot erase them.
        known = {_mask_stem(item) for item in known_masks if item}
        existing = {_mask_stem(item) for item in settings.get("excluded_masks", []) if item}
        merged = (existing - known) | submitted
    settings["excluded_masks"] = sorted(merged)
    settings["updated_at"] = datetime.now(timezone.utc).isoformat()
    (project_path / EXPORT_SETTINGS_NAME).write_text(
        json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8")
    return settings


def build_export(project_path: Path, acronym: str) -> dict:
    cards_path = project_path / "cards"
    csv_path = cards_path / "mask_info.csv"
    if not csv_path.exists():
        raise FileNotFoundError("No linked metadata CSV was found for this project")
    source = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    source = migrate_linkage_columns(source, drop_legacy=False)
    settings = load_export_settings(project_path)
    excluded = {_mask_stem(item) for item in settings.get("excluded_masks", [])}

    status = source.get("Link Status", pd.Series([""] * len(source), index=source.index))
    approved = source[status.isin({"approved", "approved_with_overrides"})].copy()
    mask_column = next((name for name in ("mask_file", "filename", "Filename")
                        if name in approved.columns), None)
    if not mask_column:
        approved["mask_file"] = ""
        mask_column = "mask_file"

    rows, images, candidates, seen_names = [], [], [], set()
    prefix = _safe_piece(acronym, "DATA")
    for _, source_row in approved.iterrows():
        mask_file = _clean_text(source_row.get(mask_column, ""))
        card = _resolve_card(cards_path, mask_file)
        if card is None:
            continue
        figure = _clean_text(source_row.get("Figure", ""))
        number = _clean_text(source_row.get("No.", ""))
        base = f"{prefix}_Fig{_safe_piece(figure, 'Unknown')}_No{_safe_piece(number, 'Unknown')}"
        export_name = f"{base}{card.suffix.lower()}"
        duplicate = 2
        while export_name.lower() in seen_names:
            export_name = f"{base}-{duplicate}{card.suffix.lower()}"
            duplicate += 1
        seen_names.add(export_name.lower())
        is_included = _mask_stem(mask_file) not in excluded
        candidates.append({
            "mask_file": card.name,
            "mask_key": _mask_stem(mask_file),
            "figure": figure,
            "vessel_number": number,
            "vessel_type": _clean_text(source_row.get("Type", "")),
            "included": is_included,
            "export_name": export_name,
        })
        if not is_included:
            continue
        row = {column: "" for column in EXPORT_COLUMNS}
        row["Image Filename"] = export_name
        for target, origin in SOURCE_COLUMNS.items():
            row[target] = _clean_text(source_row.get(origin, ""))
        rows.append(row)
        images.append((card, export_name, _mask_stem(mask_file)))

    frame = pd.DataFrame(rows, columns=EXPORT_COLUMNS).fillna("")
    state = load_linkage_state(project_path)
    figures = state.get("figures", [])
    unresolved = [{
        "figure": _clean_text(figure.get("figure_id", "")),
        "status": "Processing" if figure.get("processing_status") == "processing" else "Needs attention",
        "warnings": [_clean_text(item.get("message", "")) for item in figure.get("warnings", [])
                     if item.get("blocking")],
    } for figure in figures if figure.get("review_status") != "approved"]
    return {
        "frame": frame,
        "images": images,
        "candidates": candidates,
        "unresolved": unresolved,
        "summary": {
            "approved_figures": sum(1 for figure in figures if figure.get("review_status") == "approved"),
            "approved_vessels": len(approved),
            "unresolved_figures": len(unresolved),
            "included_masks": len(images),
            "excluded_masks": max(0, len(candidates) - len(images)),
        },
    }


def csv_bytes(frame: pd.DataFrame) -> bytes:
    text = frame.fillna("").astype(str).to_csv(
        index=False, columns=EXPORT_COLUMNS, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
    return text.encode("utf-8-sig")


def dataset_zip_bytes(result: dict, project_name: str) -> bytes:
    payload = BytesIO()
    now = datetime.now(timezone.utc).isoformat()
    with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("metadata.csv", csv_bytes(result["frame"]))
        for source, export_name, _ in result["images"]:
            archive.write(source, f"images/{export_name}")
        dictionary = StringIO()
        writer = csv.writer(dictionary, lineterminator="\n")
        writer.writerow(["Column", "Description"])
        for column in EXPORT_COLUMNS:
            writer.writerow([column, DATA_DICTIONARY.get(
                column, f"Publication value recorded in the {column} column.")])
        archive.writestr("data_dictionary.csv", "\ufeff" + dictionary.getvalue())
        unresolved_names = ", ".join(item["figure"] for item in result["unresolved"]) or "None"
        summary = result["summary"]
        archive.writestr("export_summary.txt", "\n".join([
            f"Project: {project_name}", f"Exported: {now}",
            f"Included vessel masks: {summary['included_masks']}",
            f"Excluded or unresolved masks: {summary['excluded_masks']}",
            f"Unresolved figures: {unresolved_names}", "",
        ]))
    return payload.getvalue()
