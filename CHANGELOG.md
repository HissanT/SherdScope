# Changelog

## 2026-07-11

### Added

- Added a reusable figure-to-table metadata-linking engine with a Hesban 11 publication profile, normalized figure and vessel identities, same-page through p+2 table discovery, and explicit human approval.
- Added local PP-OCRv5 mobile PaddleOCR as the default reader while retaining Gemma and OpenRouter as optional fallbacks.
- Added versioned PDF page manifests with source PDF, PDF and printed page numbers, split-page identity, logical order, and render DPI, including conservative reconstruction for legacy single-PDF projects.
- Added `cards/metadata_linkage.json` for extraction evidence, card fingerprints, validation status, reviewer edits, approval history, and feature-owned values.
- Added a Tabular review interface and project-scoped run, state, evidence, edit, and apply endpoints.
- Added 22 Hesban table fields and stable provenance columns to approved `mask_info.csv` rows.
- Added 400 DPI PDF rendering by default, a 200-600 DPI upload control, and JPEG quality settings intended to improve small printed text.
- Added a persistent per-figure review workspace with editable drawing numbers, add/duplicate/delete/sort/undo table-row controls, autosaved drafts, structured warning cards, and an explicit readiness checklist.
- Added auditable reviewer overrides for safe layout warnings while keeping missing, duplicate, unmatched, conflicting, and cross-PDF identities non-overridable.

### Changed

- Card re-extraction now preserves existing manual metadata by `mask_file` and invalidates linkage approval when card geometry changes.
- PDF replacement is rendered in a staging area before valid output replaces project files. Existing downstream masks or cards block changed PDF content, DPI, or split-page settings that would make coordinates stale.
- Approved linkage columns continue through the existing postprocessing and ZIP export path, including multiline CSV cells.
- Version increased from 0.3.0 to 0.4.0. PaddleOCR installation is optional for the main application but required for the default local linkage workflow.
- Hesban table extraction now requires a visually verified two-rule heading, reads only between the lower heading rule and closing rule, and calculates columns relative to the detected table width.
- Bounding boxes and editable card rows now show the printed vessel `No.` while stable `mask_layer_N` filenames remain private.
- Linkage fields use publication-style grouped headings in the review UI and clear public column names in working and exported CSV files. Older technical linkage columns migrate without overwriting non-empty researcher corrections.
- The extracted-table editor now uses the full panel width, grouped sticky headers, sticky `No.`/`Type` columns, larger cells, automatic multiline height, and boundary-overlay evidence previews.

### Fixed

- Blocked table-page assignments that cross PDF source boundaries.
- Preserved unexpected OCR table rows so extra-row validation cannot be bypassed.
- Prevented a backend from treating an echoed target figure ID as evidence of a printed table caption.
- Preserved later researcher corrections by replacing only values that still match the exact value previously imported by the linker.
- Corrected stable manifest ordering, split-part and split-side handling, normalized bounding-box comparison, job cleanup after extractor failure, and progress reporting during table extraction.
- Made low OCR confidence informational rather than an approval decision; unique structural validation remains the readiness rule.
- Made JSON and CSV replacement writes use unique flushed temporary files, and made malformed PaddleOCR results fail safely as empty extraction results.
- Fixed ZIP export selection when `cards_modified` exists but contains no card images.
- Prevented page numbers, table headings, and text below the closing rule from being interpreted as table rows.
- Rejected vessel number zero and stopped exposing zero-based mask suffixes as vessel identities.
- Added a dedicated per-figure OCR rerun for manually assigned or corrected table pages.
- Re-read `Type`, `Man`, and `Surface Treatment - Ext` as narrow individual cells so repeated type names are not skipped and merged `W **` tokens do not cross columns.
- Recovered table-row anchors when PaddleOCR joins a printed number to its type, including forms such as `14 Jar/Jug` and `20Jar/Jug`.
- Batched drawing-number OCR and added horizontal proximity scoring to reduce neighboring vessel-number swaps on crowded plates.
- Enlarged drawing evidence labels and made completed background OCR replace stale `?` box labels without requiring page navigation.
- Replaced fixed Hesban column proportions with page-specific boundaries derived from the actual main headings and grouped subheadings on every table page.
- Added visible vertical column overlays and persisted header-anchor evidence so reviewers can inspect exactly which OCR geometry produced each cell.
- Added an explicit `column_header_fallback` warning when a damaged header cannot safely provide enough anchors for dynamic boundaries.
- Preserved open figure panels and extracted-table scroll positions during background OCR progress polling, preventing the review UI from closing every 1.5 seconds.
- Moved header-derived column endings toward the next heading instead of splitting whitespace evenly, preventing long values such as `Cooking pot` and `Base, pedestal` from being cropped.
- Prevented background OCR progress writes from overwriting newer reviewer corrections by adding stable figure keys and revision-aware state merging.
- Prevented the 1.5-second progress refresh from erasing focused or unsaved form values while a reviewer edits an already completed figure.
- Persisted card-geometry invalidation warnings in extraction evidence so a reload cannot accidentally make a changed bounding box approvable.
- Added `approved_with_overrides` CSV status and retained each override reason, note, and timestamp in `metadata_linkage.json`.

### Validation

- `python -m pytest -q -rs`: 41 passed; 0 failed and 0 skipped.
- Major-bug-only follow-up review reran the focused OCR/linker/API/manifest tests: 41 passed; no new major implementation fixes were required.
- OCR/linker regression suite after the Type, row-anchor, drawing-number, and label-readability fixes: 42 passed.
- Dynamic header-boundary regression suite: 44 passed; real Test Project #3 pages 1, 6, and 18 each resolved all 22 header anchors with different page-specific bounds.
- Real Figure 2.3 verification preserved all eight complete Type values, including three `Cooking pot` rows and two `Base, pedestal` rows, with no column warnings.
- Python compilation passed for the application, linker, OCR adapter, project/PDF utilities, and all four test modules.
- Import checks passed with application background initialization disabled for testing.
- JavaScript syntax checks passed for the PDF and Tabular workflows; the Unix launcher passed `bash -n`; `git diff --check` passed with only informational CRLF warnings.
- Accessible review workflow regression suite: 50 Python tests passed; Python compilation and the updated Tabular JavaScript syntax check passed.

### Known limitations

- The project job lock is in-process and does not coordinate multiple independent Flask server processes.
- Ambiguous legacy projects containing multiple PDFs may require rerendering or a new project because unsafe source-to-image guesses are rejected.
- OCR accuracy still depends on scan and segmentation quality. The automated suite uses synthetic and mocked data; the reviewer did not independently rerun real-corpus PaddleOCR quality checks.
- The Windows launcher was inspected but did not receive a native batch-parser check. Optional Gemma/OpenRouter paths were not tested end to end with real credentials.
