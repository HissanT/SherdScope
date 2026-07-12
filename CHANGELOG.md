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

### Validation

- `python -m pytest -q -rs`: 41 passed; 0 failed and 0 skipped.
- Major-bug-only follow-up review reran the focused OCR/linker/API/manifest tests: 41 passed; no new major implementation fixes were required.
- OCR/linker regression suite after the Type, row-anchor, drawing-number, and label-readability fixes: 42 passed.
- Python compilation passed for the application, linker, OCR adapter, project/PDF utilities, and all four test modules.
- Import checks passed with application background initialization disabled for testing.
- JavaScript syntax checks passed for the PDF and Tabular workflows; the Unix launcher passed `bash -n`; `git diff --check` passed with only informational CRLF warnings.

### Known limitations

- The project job lock is in-process and does not coordinate multiple independent Flask server processes.
- Ambiguous legacy projects containing multiple PDFs may require rerendering or a new project because unsafe source-to-image guesses are rejected.
- OCR accuracy still depends on scan and segmentation quality. The automated suite uses synthetic and mocked data; the reviewer did not independently rerun real-corpus PaddleOCR quality checks.
- The Windows launcher was inspected but did not receive a native batch-parser check. Optional Gemma/OpenRouter paths were not tested end to end with real credentials.
