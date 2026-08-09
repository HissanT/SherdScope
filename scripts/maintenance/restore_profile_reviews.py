"""Restore pre-regeneration profile reviews from a curator workspace.

This migration is intentionally project-scoped and creates a timestamped backup
before replacing any accepted mask or review record. New cards that are absent
from the curator manifest remain untouched and unresolved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from catalog.profile_segmentation import (
    ACCEPTED_DIR,
    PROFILE_DIR,
    profile_mask_path,
    read_profile_review,
    write_profile_review,
)


def restore(project: Path, dataset_name: str, *, apply: bool) -> dict[str, object]:
    project = project.resolve()
    cards = project / "cards"
    workspace = project / "training" / dataset_name
    manifest_path = workspace / "manifest.json"
    if not cards.is_dir() or not manifest_path.is_file():
        raise FileNotFoundError("Project cards or curator manifest were not found")
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    entries = manifest.get("entries", {})
    document = read_profile_review(cards)
    records = document.setdefault("profiles", {})
    restorable: list[tuple[str, Path, Path, str, str]] = []
    missing = []
    for entry in entries.values():
        filename = str(entry.get("crop_file") or "")
        status = str(entry.get("original_review_status") or "")
        if status not in {"approved", "edited", "no_profile"} or not filename:
            continue
        card_path = cards / filename
        # The draft is the immutable mask imported from SherdScope before the
        # accidental regeneration. Curator approvals may contain later edits
        # made only for model training, so they must not silently replace the
        # user's original SherdScope review.
        relative_mask = entry.get("draft_mask")
        source_mask = workspace / str(relative_mask or "")
        if not card_path.is_file() or not source_mask.is_file():
            missing.append(filename)
            continue
        restorable.append((
            filename,
            card_path,
            source_mask,
            status,
            str(entry.get("original_review_note") or ""),
        ))

    result: dict[str, object] = {
        "project": str(project),
        "dataset": dataset_name,
        "restorable": len(restorable),
        "missing": len(missing),
        "applied": False,
    }
    if not apply:
        return result

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = cards / "profile_recovery_backups" / stamp
    backup.mkdir(parents=True, exist_ok=False)
    sidecar = cards / "profile_review.json"
    if sidecar.is_file():
        shutil.copy2(sidecar, backup / sidecar.name)
    accepted_root = cards / PROFILE_DIR / ACCEPTED_DIR
    if accepted_root.is_dir():
        shutil.copytree(accepted_root, backup / "accepted")

    contour_manifest_path = project / "matcher" / "contours" / "manifest.json"
    contour_manifest = None
    if contour_manifest_path.is_file():
        shutil.copy2(contour_manifest_path, backup / "contour_manifest.json")
        with open(contour_manifest_path, encoding="utf-8") as handle:
            contour_manifest = json.load(handle)

    for filename, card_path, source_mask, status, note in restorable:
        with Image.open(card_path) as card_image, Image.open(source_mask) as mask_image:
            mask = mask_image.convert("L")
            if mask.size != card_image.size:
                mask = mask.resize(card_image.size, Image.Resampling.LANCZOS)
            mask = mask.point(lambda value: 255 if value >= 128 else 0, mode="L")
            target = profile_mask_path(cards, filename, ACCEPTED_DIR)
            target.parent.mkdir(parents=True, exist_ok=True)
            mask.save(target, format="PNG", optimize=True)
        if contour_manifest is not None:
            contour_entry = (contour_manifest.get("entries") or {}).get(filename)
            if isinstance(contour_entry, dict) and contour_entry.get("state") == "ready":
                # The recovered mask is the same reviewed source represented by
                # the existing contour, resampled from the archival 600-DPI
                # curator copy back to the current card dimensions.
                contour_entry["source_fingerprint"] = hashlib.sha256(target.read_bytes()).hexdigest()
                contour_entry["profile_recovered_from"] = dataset_name
                contour_entry["profile_recovered_at"] = datetime.now(timezone.utc).isoformat()
        record = dict(records.get(filename) or {"filename": filename})
        record.update({
            "filename": filename,
            "accepted_mask": target.relative_to(cards).as_posix(),
            "review_status": status,
            "review_note": note,
            "restored_from_curator": dataset_name,
            "restored_at": datetime.now(timezone.utc).isoformat(),
        })
        records[filename] = record
    write_profile_review(cards, document)
    if contour_manifest is not None:
        with open(contour_manifest_path, "w", encoding="utf-8") as handle:
            json.dump(contour_manifest, handle, indent=2)
    result.update({"applied": True, "backup": str(backup)})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project", type=Path)
    parser.add_argument("--dataset", default="profile_unet_v1")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    print(json.dumps(restore(args.project, args.dataset, apply=args.apply), indent=2))


if __name__ == "__main__":
    main()
