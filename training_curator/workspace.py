"""Prepare versioned, non-destructive U-Net profile-training candidates."""

from __future__ import annotations

import hashlib
import html
import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import fitz
from PIL import Image

from catalog.profile_segmentation import ACCEPTED_DIR, profile_mask_path
from catalog.vessels import CROP_MANIFEST


WORKSPACE_SCHEMA_VERSION = 1
DEFAULT_DATASET_NAME = "profile_unet_v1"
DEFAULT_DPI = 600
MIN_DPI = 300
MAX_DPI = 900
REVIEWED_STATUSES = {"approved", "edited"}


class CuratorWorkspaceError(ValueError):
    """Raised when training candidates cannot be prepared safely."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def workspace_path(
    project_path: Path, dataset_name: str = DEFAULT_DATASET_NAME
) -> Path:
    safe_name = "".join(
        char for char in str(dataset_name) if char.isalnum() or char in {"_", "-"}
    ).strip("_-")
    if not safe_name:
        raise CuratorWorkspaceError("Dataset name must contain letters or numbers")
    return Path(project_path) / "training" / safe_name


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as handle:
            value = json.load(handle)
    except FileNotFoundError as exc:
        raise CuratorWorkspaceError(f"Required file is missing: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise CuratorWorkspaceError(f"Could not read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CuratorWorkspaceError(f"Expected a JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_dpi(dpi: int) -> int:
    value = int(dpi)
    if not MIN_DPI <= value <= MAX_DPI:
        raise CuratorWorkspaceError(
            f"Training crop DPI must be between {MIN_DPI} and {MAX_DPI}"
        )
    return value


def _source_documents(project_path: Path) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    cards = project_path / "cards"
    crop_document = _read_json(cards / CROP_MANIFEST)
    review_document = _read_json(cards / "profile_review.json")
    page_document = _read_json(project_path / "page_manifest.json")

    crops = {
        str(item.get("crop_file") or ""): item
        for item in crop_document.get("vessels", [])
        if isinstance(item, dict) and item.get("crop_file")
    }
    reviews = {
        str(filename): record
        for filename, record in (review_document.get("profiles") or {}).items()
        if isinstance(record, dict)
    }
    pages = {
        Path(str(page.get("image_name") or "")).stem: page
        for page in page_document.get("pages", [])
        if isinstance(page, dict) and page.get("image_name")
    }
    if not crops:
        raise CuratorWorkspaceError("The vessel crop manifest contains no crops")
    if not pages:
        raise CuratorWorkspaceError("The page manifest contains no PDF page mappings")
    return crops, reviews, pages


def _eligible_records(project_path: Path) -> list[dict[str, Any]]:
    crops, reviews, pages = _source_documents(project_path)
    eligible: list[dict[str, Any]] = []
    for crop_file, crop in crops.items():
        review = reviews.get(crop_file)
        if not review or review.get("review_status") not in REVIEWED_STATUSES:
            continue
        card_path = project_path / "cards" / crop_file
        accepted_path = profile_mask_path(
            project_path / "cards", crop_file, ACCEPTED_DIR
        )
        page_key = str(crop.get("image") or "")
        page = pages.get(page_key)
        if not card_path.is_file() or not accepted_path.is_file() or page is None:
            continue
        source_pdf = project_path / "pdf_source" / str(page.get("source_pdf") or "")
        if not source_pdf.is_file():
            continue
        bbox = crop.get("crop_bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        eligible.append(
            {
                "training_id": Path(crop_file).stem,
                "crop_file": crop_file,
                "card_path": card_path,
                "accepted_path": accepted_path,
                "crop": crop,
                "page": page,
                "source_pdf": source_pdf,
                "review": review,
            }
        )
    if not eligible:
        raise CuratorWorkspaceError(
            "No approved or edited profiles could be mapped safely back to a source PDF"
        )
    return sorted(eligible, key=lambda item: item["training_id"].lower())


def _source_image_size(record: dict[str, Any], pdf_page: fitz.Page) -> tuple[int, int]:
    provenance = record["crop"].get("mask_provenance") or {}
    size = provenance.get("source_image_size")
    if isinstance(size, list) and len(size) == 2 and min(size) > 0:
        return int(size[0]), int(size[1])

    page = record["page"]
    source_dpi = int(page.get("render_dpi") or 400)
    full_width = max(1, round(pdf_page.rect.width * source_dpi / 72))
    full_height = max(1, round(pdf_page.rect.height * source_dpi / 72))
    if page.get("split_side") in {"left", "right"}:
        return max(1, full_width // 2), full_height
    return full_width, full_height


def _pdf_clip(
    record: dict[str, Any], pdf_page: fitz.Page
) -> fitz.Rect:
    source_width, source_height = _source_image_size(record, pdf_page)
    x1, y1, x2, y2 = [float(value) for value in record["crop"]["crop_bbox"]]
    split_side = record["page"].get("split_side")
    if split_side in {"left", "right"}:
        full_width = source_width * 2
        if split_side == "right":
            x1 += source_width
            x2 += source_width
    else:
        full_width = source_width
    full_height = source_height
    page_rect = pdf_page.rect
    clip = fitz.Rect(
        page_rect.x0 + x1 / full_width * page_rect.width,
        page_rect.y0 + y1 / full_height * page_rect.height,
        page_rect.x0 + x2 / full_width * page_rect.width,
        page_rect.y0 + y2 / full_height * page_rect.height,
    ) & page_rect
    if clip.is_empty or clip.width <= 0 or clip.height <= 0:
        raise CuratorWorkspaceError(
            f"Crop box for {record['crop_file']} does not intersect its PDF page"
        )
    return clip


def _render_record(
    record: dict[str, Any],
    image_path: Path,
    mask_path: Path,
    *,
    dpi: int,
    document: fitz.Document | None = None,
) -> dict[str, Any]:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    owns_document = document is None
    document = document or fitz.open(str(record["source_pdf"]))
    try:
        page_index = int(record["page"].get("pdf_page_index"))
        if not 0 <= page_index < len(document):
            raise CuratorWorkspaceError(
                f"PDF page index is invalid for {record['crop_file']}"
            )
        page = document[page_index]
        clip = _pdf_clip(record, page)
        pixmap = page.get_pixmap(
            dpi=dpi,
            clip=clip,
            alpha=False,
            colorspace=fitz.csRGB,
        )
        pixmap.save(str(image_path))
    finally:
        if owns_document:
            document.close()

    with Image.open(image_path) as rendered:
        target_size = rendered.size
    with Image.open(record["accepted_path"]) as accepted:
        migrated = accepted.convert("L").resize(
            target_size, Image.Resampling.NEAREST
        )
        migrated = migrated.point(lambda value: 255 if value > 0 else 0, mode="L")
        migrated.save(mask_path, format="PNG", optimize=True)

    return {
        "image_width": target_size[0],
        "image_height": target_size[1],
        "pdf_clip_points": [
            round(clip.x0, 5),
            round(clip.y0, 5),
            round(clip.x1, 5),
            round(clip.y1, 5),
        ],
        "image_sha256": _sha256(image_path),
        "draft_mask_sha256": _sha256(mask_path),
    }


def _manifest_entry(
    record: dict[str, Any],
    rendered: dict[str, Any],
    *,
    root: Path,
    image_path: Path,
    mask_path: Path,
    dpi: int,
    prior: dict[str, Any] | None,
) -> dict[str, Any]:
    decision = str((prior or {}).get("decision") or "pending")
    if decision not in {"pending", "approved", "rejected"}:
        decision = "pending"
    approved_mask = (prior or {}).get("approved_mask")
    return {
        "training_id": record["training_id"],
        "crop_file": record["crop_file"],
        "decision": decision,
        "decision_at": (prior or {}).get("decision_at"),
        "review_note": (prior or {}).get("review_note", ""),
        "candidate_image": image_path.relative_to(root).as_posix(),
        "draft_mask": mask_path.relative_to(root).as_posix(),
        "approved_image": (prior or {}).get("approved_image"),
        "approved_mask": approved_mask,
        "approved_image_sha256": (prior or {}).get("approved_image_sha256"),
        "approved_mask_sha256": (prior or {}).get("approved_mask_sha256"),
        "source_card": record["card_path"].relative_to(
            record["card_path"].parents[1]
        ).as_posix(),
        "source_accepted_mask": record["accepted_path"].relative_to(
            record["accepted_path"].parents[3]
        ).as_posix(),
        "source_pdf": record["source_pdf"].name,
        "pdf_page_index": int(record["page"].get("pdf_page_index")),
        "page_image": record["page"].get("image_name"),
        "split_side": record["page"].get("split_side"),
        "source_render_dpi": int(record["page"].get("render_dpi") or 400),
        "training_render_dpi": dpi,
        "page_bbox": record["crop"].get("page_bbox"),
        "crop_bbox": record["crop"].get("crop_bbox"),
        "original_review_status": record["review"].get("review_status"),
        "original_review_note": record["review"].get("review_note", ""),
        **rendered,
    }


def prepare_workspace(
    project_path: Path,
    *,
    dataset_name: str = DEFAULT_DATASET_NAME,
    dpi: int = DEFAULT_DPI,
    limit: int | None = None,
    force: bool = False,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Render candidates and migrate reviewed masks into a versioned workspace."""
    project_path = Path(project_path).resolve()
    if not (project_path / "project.json").is_file():
        raise CuratorWorkspaceError("Project folder must contain project.json")
    dpi = _validate_dpi(dpi)
    root = workspace_path(project_path, dataset_name)
    manifest_path = root / "manifest.json"
    prior_manifest = (
        _read_json(manifest_path) if manifest_path.is_file() else {"entries": {}}
    )
    prior_entries = prior_manifest.get("entries") or {}
    records = _eligible_records(project_path)
    if limit is not None:
        records = records[: max(0, int(limit))]
    entries: dict[str, dict[str, Any]] = dict(prior_entries)
    total = len(records)
    rendered_count = skipped_count = 0
    documents: dict[Path, fitz.Document] = {}
    try:
        for index, record in enumerate(records, start=1):
            training_id = record["training_id"]
            image_path = root / "candidates" / "images" / f"{training_id}.png"
            mask_path = root / "candidates" / "masks" / f"{training_id}.png"
            prior = prior_entries.get(training_id)
            reusable = bool(
                not force
                and prior
                and int(prior.get("training_render_dpi") or 0) == dpi
                and image_path.is_file()
                and mask_path.is_file()
            )
            if progress:
                progress(index, total, training_id)
            if reusable:
                skipped_count += 1
                continue
            source_pdf = record["source_pdf"]
            if source_pdf not in documents:
                documents[source_pdf] = fitz.open(str(source_pdf))
            rendered = _render_record(
                record,
                image_path,
                mask_path,
                dpi=dpi,
                document=documents[source_pdf],
            )
            entries[training_id] = _manifest_entry(
                record,
                rendered,
                root=root,
                image_path=image_path,
                mask_path=mask_path,
                dpi=dpi,
                prior=prior,
            )
            rendered_count += 1
            if rendered_count % 10 == 0:
                document = {
                    "schema_version": WORKSPACE_SCHEMA_VERSION,
                    "dataset_name": dataset_name,
                    "project_path": str(project_path),
                    "created_at": prior_manifest.get("created_at") or utc_now(),
                    "updated_at": utc_now(),
                    "config": {"training_render_dpi": dpi},
                    "entries": entries,
                }
                _atomic_json(manifest_path, document)
    finally:
        for document in documents.values():
            document.close()

    document = {
        "schema_version": WORKSPACE_SCHEMA_VERSION,
        "dataset_name": dataset_name,
        "project_path": str(project_path),
        "created_at": prior_manifest.get("created_at") or utc_now(),
        "updated_at": utc_now(),
        "config": {"training_render_dpi": dpi},
        "entries": entries,
    }
    root.mkdir(parents=True, exist_ok=True)
    _atomic_json(manifest_path, document)
    return {
        "workspace": root,
        "manifest": manifest_path,
        "eligible": total,
        "rendered": rendered_count,
        "reused": skipped_count,
        "total_entries": len(entries),
    }


def _pilot_sample(records: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    count = max(1, min(int(count), len(records)))
    if count == len(records):
        return records
    ordered = sorted(
        records,
        key=lambda item: (
            (
                int(item["crop"]["crop_bbox"][2])
                - int(item["crop"]["crop_bbox"][0])
            )
            * (
                int(item["crop"]["crop_bbox"][3])
                - int(item["crop"]["crop_bbox"][1])
            ),
            item["training_id"],
        ),
    )
    positions = {
        round(index * (len(ordered) - 1) / max(1, count - 1))
        for index in range(count)
    }
    return [ordered[position] for position in sorted(positions)]


def build_dpi_pilot(
    project_path: Path,
    *,
    dataset_name: str = DEFAULT_DATASET_NAME,
    dpi: int = DEFAULT_DPI,
    count: int = 20,
    force: bool = False,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Create a small side-by-side old/new quality report before full rendering."""
    project_path = Path(project_path).resolve()
    dpi = _validate_dpi(dpi)
    root = workspace_path(project_path, dataset_name) / "dpi_pilot"
    records = _pilot_sample(_eligible_records(project_path), count)
    rows = []
    documents: dict[Path, fitz.Document] = {}
    try:
        for index, record in enumerate(records, start=1):
            if progress:
                progress(index, len(records), record["training_id"])
            old_path = root / "old" / f"{record['training_id']}.png"
            new_path = root / f"new_{dpi}dpi" / f"{record['training_id']}.png"
            mask_path = root / "migrated_masks" / f"{record['training_id']}.png"
            old_path.parent.mkdir(parents=True, exist_ok=True)
            if force or not old_path.is_file():
                shutil.copy2(record["card_path"], old_path)
            if force or not new_path.is_file() or not mask_path.is_file():
                source_pdf = record["source_pdf"]
                if source_pdf not in documents:
                    documents[source_pdf] = fitz.open(str(source_pdf))
                rendered = _render_record(
                    record,
                    new_path,
                    mask_path,
                    dpi=dpi,
                    document=documents[source_pdf],
                )
            else:
                with Image.open(new_path) as image:
                    rendered = {
                        "image_width": image.width,
                        "image_height": image.height,
                    }
            with Image.open(old_path) as old_image:
                old_size = old_image.size
            rows.append(
                {
                    "training_id": record["training_id"],
                    "old": old_path.relative_to(root).as_posix(),
                    "new": new_path.relative_to(root).as_posix(),
                    "mask": mask_path.relative_to(root).as_posix(),
                    "old_size": list(old_size),
                    "new_size": [
                        rendered["image_width"],
                        rendered["image_height"],
                    ],
                }
            )
    finally:
        for document in documents.values():
            document.close()

    report = root / "index.html"
    cards = "\n".join(
        f"""<article>
<h2>{html.escape(row['training_id'])}</h2>
<p>Current: {row['old_size'][0]}×{row['old_size'][1]} · Direct {dpi} DPI:
{row['new_size'][0]}×{row['new_size'][1]}</p>
<div><figure><figcaption>Current crop</figcaption><img src="{row['old']}"></figure>
<figure><figcaption>Direct {dpi}-DPI PNG crop</figcaption><img src="{row['new']}"></figure>
<figure><figcaption>Migrated mask</figcaption><img src="{row['mask']}"></figure></div>
</article>"""
        for row in rows
    )
    report.write_text(
        f"""<!doctype html><html><head><meta charset="utf-8">
<title>SherdScope {dpi}-DPI crop pilot</title>
<style>
body{{font:15px system-ui;margin:24px;background:#f4f6f9;color:#172033}}
article{{background:white;padding:18px;margin:0 0 22px;border:1px solid #d8dee8;border-radius:10px}}
article div{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}}
figure{{margin:0}} figcaption{{font-weight:700;margin-bottom:6px}}
img{{width:100%;height:360px;object-fit:contain;border:1px solid #d8dee8;background:white}}
@media(max-width:900px){{article div{{grid-template-columns:1fr}}}}
</style></head><body>
<h1>Lossless direct-from-PDF crop pilot</h1>
<p>Compare edge clarity only. These files do not replace project cards or masks.</p>
{cards}</body></html>""",
        encoding="utf-8",
    )
    _atomic_json(
        root / "pilot.json",
        {
            "schema_version": 1,
            "created_at": utc_now(),
            "project_path": str(project_path),
            "target_dpi": dpi,
            "count": len(rows),
            "rows": rows,
        },
    )
    return {"pilot_root": root, "report": report, "count": len(rows), "dpi": dpi}
