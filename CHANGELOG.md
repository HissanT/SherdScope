# Changelog

## 2026-07-16

### Added

- Added a SherdScope pottery-profile and magnifying-lens logo for the browser tab, splash screen, application header, and information panel.
- Added a compact categorical-cell OCR preparation path that trims, pads, and enlarges isolated printed glyphs before local PaddleOCR recognition.
- Added saved Non-Plastics Type OCR diagnostics showing the exact first-line crop, raw retry token, confidence, overlapping whole-page tokens, and accepted value inside the figure review workspace.

### Changed

- Rebranded the active application interface, browser title, startup message, help link, and information panel from PyPotteryLens to SherdScope while retaining clear PyPotteryLens attribution and GPL licensing information.
- Simplified the researcher CSV to 25 analysis fields by removing Figure Caption, Diameter Source, Drawing Page, Table Pages, Source PDF, and Link Status from final exports. The working linkage sidecar and CSV still retain provenance internally.
- Split PDF rendering, processor configuration, scale/vessel sidecars, and researcher export routes into focused modules while retaining the existing imports, URLs, file formats, and behavior.

### Fixed

- Preserved the document, publication viewer, figure list, and editable-table vertical and horizontal scroll positions across the 1.5-second background linkage refresh.
- Corrected table group jumps to calculate their destination from the first actual data cell in each group rather than a grouped heading, eliminating the rightward overshoot.
- Corrected the Non-Plastics Type retry to inspect only the first printed line of each row. The previous whole-row crop included lower-line fragments from neighboring multiline cells and produced blanks or values such as `I`, `T`, `L+h`, and `L+7`.
- Removed a verified compact Type code from a preceding Fabric Interior token only when the original token geometrically crossed the column boundary.
- Kept Actions, No., and Type headings visible during vertical table scrolling while allowing their body cells to scroll horizontally with the rest of each row.
- Removed a global textarea minimum width that made the No. editor overflow its cell and visually touch the Type editor.
- Removed the unreachable Post Processing JavaScript, hidden page markup, and unused Post Processing CSS after confirming that the script was not loaded and the workflow had already been replaced by Export.
- Removed statically confirmed unused Python imports, local variables, and duplicate late imports without changing application logic.

### Tests

- Full Python suite: 77 tests passed.
- JavaScript syntax checks passed for the Review & Link, Export, and main application scripts.
- Python compilation and diff validation passed.
- Live browser checks confirmed exact group-jump alignment, sticky identity headers after 520 pixels of vertical scrolling, separated No./Type editors, SherdScope branding, and the reduced 25-column Export preview.
- Real PP-OCRv5 mobile-model acceptance on Figure 2.1 read all 19 Non-Plastics Type cells correctly across both table pages; all 19 diagnostics were accepted.

## 2026-07-15

### Changed

- Replaced the active Tabular and Post Processing workflow with two focused steps: **Review & Link** and **Export**.
- Simplified figure review to a narrow figure list, one large publication-page viewer, a selected-figure table, compact controls, and a light publication-style interface.
- Made structurally valid automatic scale and rim-diameter results immediately usable as `verified_automatic`; researcher corrections are retained separately as `verified_manual`.
- Removed model/backend terminology and repeated diameter-verification buttons from the normal review workflow. Local PaddleOCR now runs behind one **Read and Link Tables** action.
- Replaced the Post Processing navigation with a dedicated export page where researchers can include or exclude approved vessel masks and preview the final dataset.
- Made figure-table extraction visibly sequential: figures that have not started are now labelled **Waiting**, only the current figure is labelled **Processing**, and completed figures become available for review one at a time.
- Simplified the extracted table to one horizontal scrolling area. Actions, vessel number, and vessel type now move with the rest of the row, the unnecessary **Sort by No.** control is gone, and group-jump buttons move the same table viewport.

### Added

- Added clean research export endpoints for preview, saved mask selection, CSV download, and dataset ZIP download.
- Added project-level `export_settings.json`, including one-time migration of older Post Processing exclusion choices.
- Added a fixed researcher-facing 31-column schema, UTF-8 BOM output, correct multiline CSV quoting, readable diameter-source labels, and stable figure/number-based image filenames.
- Added dataset packages containing `metadata.csv`, an `images/` folder, `data_dictionary.csv`, and `export_summary.txt`.
- Added persistent publication-viewer state so the selected evidence page, zoom level, and box visibility survive the background OCR refresh cycle.
- Added plain-language explanations for unresolved diameter measurements, including missing scale, missing top rim, missing centreline, invalid drawing crop, and disagreement between the two diameter estimates.

### Fixed

- Stopped final exports from depending on `cards_modified`, image flips, ENT/FRAG classifications, or legacy merged-classification CSVs.
- Prevented bounding boxes, OCR evidence, fingerprints, internal mask keys, classifier fields, and other implementation details from leaking into the final research CSV.
- Prevented `nan`, `None`, mojibake, and alphabetically scrambled headings from appearing in exported metadata.
- Fixed a startup regression found during browser acceptance after replacing the Post Processing tab.
- Fixed publication previous/next, zoom, reset, and box-visibility controls being undone or visually ignored while linkage progress refreshed the page.
- Fixed a legacy CSS cascade that overrode publication zoom and allowed older dark-theme styles to leak into the redesigned light workspace.
- Fixed queued figures being silently changed to ready or reviewable before their OCR work had started. Editing, measurement, and approval endpoints now reject unfinished figures.
- Fixed the CSV group-jump buttons so each button moves directly to its named column group in the single table viewport.
- Hid the missing-table-closing-line warning from normal review and approval while retaining it in the saved OCR evidence for auditing.
- Fixed export settings so changing visible mask choices cannot erase older hidden exclusions imported from Post Processing.
- Fixed a download race where CSV or ZIP export could begin before a pending include/exclude autosave finished; downloads now wait for the save and stop if it fails.
- Corrected the upstream PyPotteryLens attribution from Leonardo Cardarelli to Lorenzo Cardarelli in SherdScope documentation.

### Tests

- Full Python suite: 75 tests passed.
- Python compilation passed for the application, linker, measurement detector, and research-export modules through `compileall`.
- JavaScript syntax validation passed for the Review & Link, Export, and main navigation scripts.
- Diff validation passed with no whitespace errors.

## 2026-07-14

### Added

- Added deterministic automatic detection of the standard Hesban `0-10 CM` graphic ruler, including structural validation, split-page sibling reuse, same-PDF median checks, evidence bounds, fingerprints, and reviewer-correctable manual calibration.
- Added rim-diameter suggestions derived from the detected illustrated rim span and an independent centreline-to-profile radius. Suggestions are withheld when the two estimates differ by more than five percent and are never calculated from the card bounding-box width.
- Added reviewer measurement controls, draggable scale and rim endpoints, evidence overlays, per-drawing verification, and revision-protected persistence in `metadata_linkage.json` and existing scale sidecars.
- Added `Rim Diameter (cm)` and `Diameter Status` to the review table and CSV integration. Only reviewer-verified values are exported; unresolved measurements remain blank without blocking otherwise valid table metadata.
- Added a figure-scoped measurement endpoint, full-screen table mode, grouped column-jump controls, and synchronized upper and lower horizontal scrollbars.

### Fixed

- Removed the lower-45%-of-page assumption from Hesban ruler detection. The detector now searches the complete rendered page and identifies the scale from its segmented alternating-block structure, allowing short diagram plates whose ruler appears higher on the page.
- Replaced whole-span rim selection with the Hesban drawing convention used by the corpus: inspect only the top 10% of the card, identify the highest credible rim stroke, measure its connected left edge to the central reconstruction axis, and mirror that radius across the axis. Small publication gaps before the separate right profile no longer shorten the estimated diameter.
- Grouped the first few scan rows of the same top stroke so slightly uneven printed lines are measured as one rim while longer lower vessel lines remain excluded.
- Changed scale and rim evidence overlays to thin lines with small hollow handles, and made the editor open on a close evidence crop with Zoom in, Zoom out, and Fit evidence controls so handles no longer cover the ruler or lip.
- Prevented ambiguous, median-rejected, or otherwise unresolved ruler candidates from supplying `px_per_cm`, generating diameter suggestions, or becoming usable legacy scale records.
- Preserved existing zoned manual scales when the linker updates the page-wide ruler instead of replacing the complete legacy scale-sidecar list.
- Added locked scale/CSV persistence and cleared stale card ratios when calibration becomes unusable, preventing concurrent reviewer saves or rejected scales from leaving trusted-looking values behind.
- Added true local deskewing for mildly rotated rulers, stronger broken/faint/noisy ruler handling, same-PDF/render-DPI median filtering, and safer split-page sibling reuse.
- Corrected centreline selection to prefer the vertical reconstruction axis nearest the rim midpoint instead of a longer outer vessel wall.
- Rejected non-finite scale coordinates, drawing boxes, ratios, and diameter values at the measurement and API boundaries.
- Preserved full-precision automatic diameter suggestions when a reviewer verifies the displayed rounded value, and added scale/diameter reviewer-history records.
- Made rejected scales appear unresolved in the review screen and made both horizontal table scrollbars keyboard focusable.
- Corrected sticky-column sizing and offsets so borders and padding cannot cover the first character of `Type`, `Sq`, `Loc`, or later cells.
- Invalidated diameter evidence when its page calibration, source page, or card geometry changes while preserving compatible legacy manual calibrations.
- Protected later researcher-entered CSV diameter corrections from automatic reapproval replacement.

### Validation

- Focused measurement and linkage API suite: 24 tests passed during review.
- Full Python suite: 71 tests passed in SherdScope and 71 tests passed after synchronizing the reviewed files to the working PyPotteryLens fork.
- Python compilation passed for `app.py`, `metadata_linker.py`, `hesban_measurements.py`, and `utils.py` in both repositories.
- JavaScript syntax validation passed for `static/js/tabular-tab.js` in both repositories.
- Synthetic automatic-scale tests passed for clean, faint, broken, mildly skewed, noisy, absent, and competing ruler cases; persistence, split-page reuse, median rejection, rim measurement, revision handling, and verified-only CSV export were also covered.
- `git diff --check` passed in both repositories; only informational Windows LF/CRLF conversion notices were reported.

### Known limitations

- Live browser acceptance for endpoint dragging, full-screen tables, sticky columns, synchronized scrolling, keyboard behavior, and narrow layouts remains manual because the installed browser-control package is missing its required runtime script.
- Representative Hesban corpus checks for everted, inverted, thickened, and incomplete rim styles were not run during this review; automatic diameter results remain reviewer suggestions until verified.

## 2026-07-12

### Reviewed

- Completed an independent implementation review of the accessible figure-review and CSV-correction workflow against its agreed specification.
- Confirmed that completed figures remain editable during later OCR work, reviewer drafts use stable figure keys and revisions, and CSV approval remains a separate validation step.

### Fixed

- Serialized browser autosaves and guarded them with edit versions so an older response cannot mark newer typing as saved or allow polling to erase it.
- Added conflict-aware three-way draft merging: disjoint server changes are merged automatically, while overlapping changes keep the local draft visible and report a save conflict for review.
- Switched review controls, save/rerun routes, and evidence previews to stable figure keys so correcting a printed Figure ID does not break later editing or image evidence.
- Persisted per-figure OCR rerun processing state before extraction and restored the figure to a reviewable state after success or failure, blocking same-figure edits and approval during extraction.
- Allowed a completed ready figure to be approved while a different figure is processing; selected processing figures are still rejected, and approval is protected by the linkage lock, reviewer revisions, and stale-background merging.
- Restricted warning overrides to the three approved warning codes and their preset reasons, with server-owned timestamps that remain stable across later autosaves.
- Added non-overridable validation for cross-PDF assignments and a core approval guard for figures that are still processing.
- Added direct warning actions for editing drawing numbers, opening affected rows, and creating missing rows; drawing-number changes now immediately update row highlighting and refresh evidence.
- Corrected the sticky drawing-number workspace and table-toolbar positioning, including focus visibility and narrow-screen fallbacks.
- Reapproval now replaces only values still owned by the linker, allowing corrected extracted data to update CSV while preserving later manual researcher corrections.

### Validation

- Focused linkage and API suite: 39 tests passed.
- Full Python suite: 56 tests passed in 13.58 seconds.
- Python compilation passed for `app.py`, `metadata_linker.py`, and `ocr_extractor.py`.
- JavaScript syntax validation passed for `static/js/tabular-tab.js`.
- `git diff --check` passed; only informational Windows LF/CRLF conversion notices were reported.
- Interactive browser acceptance was not run, so sticky positioning, narrow-screen layout, and full keyboard behavior still require a short manual UI check.

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
