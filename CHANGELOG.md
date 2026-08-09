# Changelog

## 2026-08-08 - DCT shape-retrieval baseline

- Added a paper-inspired, shape-only Discrete Cosine Transform baseline using
  100-point rim outlines, 20 harmonics, coefficient RMSD, deterministic pose
  normalization, and an inspectable reference-coverage search.
- Set the experiment's reference-coverage search to 50% through 100% in 5%
  increments. Smaller candidates are excluded because the real queries already
  preserve substantial profile chunks and should not be matched to tiny local
  sections.
- Added read-only single-query and paired synthetic experiment commands with
  versioned JSON/CSV results, summary metrics, protected output directories,
  and a query reconstruction plot.
- Added an eight-worker-capable known-parent batch command that consumes the
  existing 29-query and 40-query manifests, verifies saved artifact hashes,
  resolves exact Figure/Item parents, and reports combined and per-cohort
  Top-K ranks and MRR.
- Added flushed terminal progress for reference-bank construction and every
  completed batch query, including completed count, true-parent rank, selected
  coverage, per-query time, and total elapsed time.
- Added plain-English method and hyperparameter documentation plus regression
  tests for pose invariance, partial-profile coverage recovery, reconstruction
  behavior, reproducible output, and invalid configuration handling.

## 2026-08-08 - real-sherd expert evaluation workflow

- Added a guarded, resumable launcher for the 68 smoothed real-sherd queries
  that reuses the standard isolated record exporter and three-sheet workbook,
  then generates the non-ground-truth Markdown evaluation and figures.
- Added a localhost-only 0–3 expert scoring interface with atomic autosave,
  exact resume behavior, query/candidate notes, a no-acceptable-match flag,
  flipped-photo support, and keyboard grading for the saved top-N candidates.
- Added reproducible expert-score CSV export, a generated real-sherd scorecard,
  and PNG/SVG plots for expert scores, match cost, winner margin, retrieval
  rank, cost-versus-score correlation, and per-stage runtime.

## 2026-08-06 - complete diagnostic-profile IoU evaluation

- Evaluated all 2,662 preserved automatic profile proposals against their
  accepted reviewer masks without modifying either artifact set. Added macro
  IoU, micro-IoU, Dice, precision, recall, percentiles, threshold counts,
  mask-area change, and the lowest-IoU cases to `PROFILE_IOU_EVAL.md`.
- Added a reproducible per-profile evaluator under `scripts/evaluation` and
  regression coverage for exact binary IoU and dimension-mismatch handling.
- Added the headline segmentation results to `EVAL_SCORES.md`; the complete
  per-profile measurements remain in the ignored evaluation output directory.

## 2026-08-05 - committed evaluation scorecard

- Added `EVAL_SCORES.md` as the version-controlled scorecard for the latest
  completed 29-query v15 shape-only and shape-plus-metadata comparison. It
  records ranks, Top-K flags, costs and gaps, retrieval-channel ranks, cascade
  survival, paired metadata effects, metadata coverage, Top-5 overlap, and
  per-stage runtime.
- Added a reproducible report builder under `scripts/evaluation` so the
  scorecard can be regenerated from the stored batch manifest and raw run
  records without manually copying workbook cells.

## 2026-08-05 - unified metadata score explanations

- Made each displayed metadata summary derive from the effective difference
  between the existing shape score and final clipped fused score. The text can
  therefore no longer call metadata supportive when the numeric score was
  penalized, or conflicting when it received a bonus.
- This is display-only: retrieval, metadata adjustments, fused scores, and
  ranking are unchanged.

## 2026-08-05 - repository cleanup

- Moved standalone research commands out of the repository root and into the
  `scripts/matcher`, `scripts/real_sherd`, `scripts/segmentation`,
  `scripts/training`, and `scripts/maintenance` packages. Commands now run as
  Python modules so imports resolve consistently from the project root.
- Removed the hard-coded, one-off Query 2 workbook repair script. Its result is
  already represented by the general evaluation exporter, so retaining the
  project-specific copy risked writing stale query mappings.
- Quarantined generated Python/test caches and Codex scratch analysis files for
  safe removal, and moved the downloaded SAM checkpoint under `models`. Saved
  projects, query corpora, reviewed artifacts, experiment outputs, and research
  logs were left untouched.

## 2026-08-05 - 400-candidate retrieval pool

- Standardized the shape-only and shape-plus-metadata retrieval pool at 400
  references before the expensive scoring cascade. This gives borderline
  candidates more opportunity to survive coarse retrieval without changing
  the later scoring weights.
- Extended the synthetic benchmark report through Top-400 while retaining
  Top-300 as a useful historical comparison point.

## 2026-08-04 - structural scale validation

- Removed the project-wide `scale_median_disagreement` rejection. Hesban PDFs
  can contain multiple legitimate scan scales within the same source file and
  render DPI, so structurally valid ruler detections are no longer compared to
  one global pixels-per-centimetre median.
- Added compatibility migration for saved median-disagreement records. Legacy
  automatic rulers with retained endpoints are restored as structurally
  verified calibrations and their obsolete warning is removed.

## 2026-08-04 — final balanced metadata calibration

- Versioned the matcher as v15 and metadata model as v6. Retained the stronger
  general v13 metadata bonus and penalty, while halving the aggressive v14
  severe-diameter quadratic tail and its cap. This is the final middle-ground
  calibration used for the next 30-query experiment.

## 2026-08-04 — continuing severe-diameter penalty

- Versioned the matcher as v14 and metadata model as v5. Added a smooth,
  reliability-scaled quadratic penalty beyond a 5 cm rim-diameter gap so
  severe conflicts continue separating instead of saturating near the generic
  metadata penalty ceiling. The extra tail remains capped and does not affect
  close diameter agreement.

## 2026-08-04 — stronger metadata calibration

- Versioned the matcher as v13 and the metadata model as v4. Increased the
  nominal metadata ceiling from 15% to 22%, the maximum compatibility bonus
  from 0.018 to 0.030 normalized cost, and the maximum incompatibility penalty
  from 0.045 to 0.075. This keeps shape primary while giving reliable diameter
  and fabric evidence more power at retrieval and every cascade cutoff.

## 2026-08-04

### Full-pipeline metadata retrieval

- Versioned the matcher as v12 and changed the 30-query batch so its
  shape-plus-metadata arm starts from the full reference catalogue. Metadata
  now influences the 300-candidate retrieval cutoff and every later survivor
  stage instead of only reranking the fine-scored pool.
- Kept shape-only and metadata-aware runs independent for an auditable
  comparison. Missing metadata remains neutral, while reliable severe
  mismatches such as implausible rim diameters receive a bounded penalty.
- Added retrieval regression coverage proving that a severe diameter mismatch
  can evict the nominal shape-only boundary candidate before expensive scoring.

## 2026-08-03

### Incremental review safety and profile recovery

- Fixed Review Profiles so proposal generation is missing-only by default and
  can be limited to one source PDF. Reviewed accepted masks are never replaced,
  including when automatic proposals are explicitly refreshed.
- Restored the 1,099 approved Project 18 profiles from the immutable curator
  archive after the old UI accidentally regenerated them. The restore created
  a timestamped backup and left all 139 cards from `Eight_new_Figures_Hesban.pdf`
  unresolved as the only new review work.
- Integrated the curator's actual high-resolution workflow into Review
  Profiles: crops are rendered directly from the source PDF at 600 DPI, edits
  and guided recovery operate at that resolution, and masks are reduced to the
  project card size only on save. The large curator-style editor/preview layout,
  Pan, Original Auto, Autofill (G), cursor outlines, undo, and keyboard controls
  are available in the main application.
- Fixed Review & Link source selection so each imported PDF is selectable, with
  the newest source selected initially. Source-specific linking preserves all
  unrelated existing figure state.
- Made canonical-contour construction incremental: unchanged valid contours are
  preserved and only missing or genuinely changed contours are built.

### Documentation and validation checkpoint

- Added a consolidated research-evidence checkpoint covering the 1,058-reference
  v9-v11 matcher runs, shape-plus-metadata comparison, corrected 360-query
  synthetic benchmark, five-photo U-Net/SAM/DeepLab comparison, exploratory
  type voting, runtime diagnostics, exceptions, and threats to validity.
- Updated the AAAI-style manuscript and research logs from the saved workbooks,
  manifests, JSON diagnostics, and changelog rather than relying on remembered
  headline results.
- Fresh combined linkage runs now keep unchanged, fully approved figures frozen
  while processing appended or unfinished figures. Geometry changes still use
  the existing invalidation path and correctly return affected figures to
  review.
- Added and visually verified a 24-page Hesban PDF containing Figures 4.5,
  3.53, 3.65, 3.76, 3.25, 3.56, 3.88, and 3.84 with p, p+1, and p+2 for each.

- Added a leakage-controlled five-photo segmentation benchmark covering the
  original U-Net, topology-safe enclosed-hole filling, SAM 2.1 Tiny prompted
  only from U-Net geometry, and a DeepLabV3+ model trained on the separate 68
  labelled photographs. It exports identical per-image metrics, aggregate
  metrics, masks, colour overlays, a visual HTML gallery, and a Markdown
  summary without changing the manual gold masks.

- Added an isolated five-photograph real-sherd segmentation pilot. It preserves
  the original JPGs and hashes, runs the existing ResNet-50 U-Net on CUDA when
  available, proposes adjustable crops, and provides an independent zoomable
  outline/brush/eraser gold-mask editor. U-Net predictions remain hidden until
  the gold mask is saved. The pilot then exports lightly smoothed manual and
  U-Net query blobs plus Dice, IoU, precision, recall, boundary F1, HD95, mean
  surface distance, and area error reports in CSV and JSON form.

- Added conservative matcher v11. It restores the validated two-channel
  outline/ribbon retrieval path, preserves every ordinary score top-K candidate,
  and only appends a small number of channel champions. The v10 timing and
  candidate-lineage diagnostics remain, while the unvalidated persistent-local
  retrieval channel is disabled pending held-out testing.
- Added a separate Hesban query-metadata curator that displays the saved query,
  fracture trace, rim point, and the read-only catalogue metadata already stored
  for its known Figure + Item, then stores optional noisy human observations
  without changing the query geometry. The 30-query batch now exports both
  shape-only and shape-plus-metadata rankings from the same fine-scored shape
  pool. Shape and metadata now form one continuous cost with shape carrying the
  larger share; metadata can move candidates out of the final five but cannot
  import shape-rejected references. Blank fields are neutral.

- Versioned the shape matcher as `two-wall-joint-fgw-v10`. Local curvature now
  distinguishes cross-scale persistent fine features from unstable pixel
  stair-steps instead of broadly smoothing small contour movement. Retrieval
  adds a persistent-local-shape channel without enlarging the 300-candidate
  pool; the cascade reserves bounded champions from every retrieval channel
  through pruning and returns to pure score ordering for the final five.
- Added per-stage candidate lineage, known-target ranks, and retrieval/coarse/
  medium/fine timings to matcher results. The medium pass now uses one exact
  composite hypothesis consistent with coarse pruning, leaving the diverse
  expensive hypothesis search for finalists and materially reducing repeated
  work. Confidence thresholds and UI labels remain intentionally deferred.

- Added an optional uncertainty-aware Hesban metadata comparison and fusion
  module. It supports rim diameter, Munsell colour, non-plastics, voids,
  manufacture, surface treatment, decoration qualifiers, and firing. Missing
  fields are neutral; noisy numeric and ordinal differences are gradual rather
  than hard thresholds; correlated fields are capped; every contribution is
  explained. Fusion preserves every shape candidate and adapts its influence to
  available, reliable evidence instead of applying a fixed 50/50 weight. The
  current matcher remains shape-only until a held-out metadata experiment is
  explicitly enabled.

## 2026-08-02

### Changed

- Added a resumable sequential runner for the exact 30 reviewed Hesban query
  artifacts. It reuses their saved fracture/rim annotations, scores a known
  target separately only when absent from the top five, and writes an isolated
  Excel workbook without changing the prepared query artifacts.
- Added a one-time local Hesban query curator for drawing and saving the purple
  fracture trace and gold rim point across Queries 1-30 before batch matching.

- Versioned the shape matcher as `two-wall-joint-fgw-v9` and expanded the
  metadata-free retrieval safety pool from 150 to 300 candidates. The coarse,
  medium, and fine stages now retain 40, 12, and 5 candidates respectively.
- Added bounded deterministic candidate concurrency. The cheap coarse stage
  uses up to four workers; POT-based exact stages remain sequential because
  local timing showed native-solver contention when they were threaded.
- Added a mild 6% squared unmatched-reference completeness component so a
  candidate receives a small penalty for explaining only a convenient short
  interval. The contribution is stored in result diagnostics and displayed in
  Match Query without changing existing saved runs.
- Added a seeded synthetic rim-query benchmark and visual preflight galleries.
  Rim selection now compares both end caps in original upright publication
  coordinates instead of assuming the contour's nominal seam is archaeological
  truth.

### Validation

- All 183 automated tests pass. A 90-query clean synthetic retrieval check
  retained 87 parents in the 300-candidate pool (96.7%), compared with 83 in
  the former 150-candidate pool (92.2%). A one-query full rerank also confirmed
  that runtime and intermediate scoring still require further work; it is not
  presented as an accuracy estimate.

## 2026-07-31

### Added

- Added a non-destructive U-Net dataset preparation command that renders only
  reviewed vessel rectangles directly from their source PDF as configurable,
  lossless PNG crops.
- Added a 20-profile DPI pilot report for comparing existing JPEG-derived cards
  with direct 600-DPI PDF crops before preparing the full corpus.
- Added a resumable local training curator with brush, eraser, pan, zoom, undo,
  approve, reject, draft-save, navigation hotkeys, atomic decisions, immutable
  approved snapshots, and SHA-256 provenance.
- Added thin high-contrast brush/eraser cursor rings, conservative isolated-speck
  cleanup on approval, and an undoable high-DPI recovery action for omitted
  fracture/profile ink that leaves saved masks untouched until approval.
- Added compact mixed-precision U-Net training with deterministic train,
  validation, and unseen test splits; automatic threshold selection and early
  stopping; a color-coded holdout preview; and separate, resumable predictions
  for pending masks. Production predictions are anchored to the migrated draft
  location to reject unrelated drawings without overwriting approved labels.
- Restored the original SherdScope automatic mask as the curator default after
  corpus spot-checking showed the 320-pixel U-Net could lose thin detail on very
  wide crops. Added explicit `O`/`U` controls for optional side-by-side workflow
  comparison while retaining all generated predictions separately.
- Added an **All PDFs (combined)** metadata-linking mode for complementary PDF
  chunks. Combined runs preserve per-page PDF provenance and stop table
  lookahead at source boundaries while presenting one unified figure queue.
- Added a confirmed **Approve all flagged contours** action to canonical-contour
  review. It accepts all successfully built flagged contours atomically while
  leaving failed extractions unresolved for repair.
- Fixed Review & Link autosave conflicts caused by the same browser racing its
  own diameter, column, or background-review action. No-op autosaves no longer
  increment revisions, same-tab mutations rebase automatically, and genuinely
  different browser tabs still receive stale-overwrite protection.
- Fixed Review & Link approval for card filenames containing decimal-like dots
  (such as `3.1-3.50`). Card IDs now remove only real image extensions, and
  legacy figure-key collisions are repaired without discarding review data.
- Fixed the browser-side approval check so an intentionally accepted
  `unexpected_table_row` warning no longer leaves an otherwise-ready figure's
  approval button disabled.
- Combined Review & Link runs now collapse an exact figure repeated across two
  PDF chunks when both copies contain the same complete, unique vessel-number
  set, preventing false duplicate-number blockers such as Figure 3.43. When
  only one source supplies the accepted table pages, its drawing copy is kept
  (as required for Figure 3.44), and candidate lookahead stays within each PDF.
- Fixed stale per-figure `queued`/`processing` labels after a completed linkage
  job has no active queue, so a figure such as 3.50 no longer appears stuck.
- Fixed Match Query metadata joins for dotted PDF-derived filenames. All
  references now retain their full card key, allowing available labels to show
  as `Figure X Item Y` instead of falling back to cryptic filenames.
- Match Query now omits contour references that have no Figure/Item metadata
  once `mask_info.csv` exists, so obsolete or not-yet-linked duplicates cannot
  appear as filename-only matches.

## 2026-07-30

### Added

- Added canonical contour cleaning for approved profile masks, with derived
  matcher artifacts stored separately from the accepted binary masks.
- Added a **Match Query** workflow for PNG query masks, including preprocessing,
  query metadata capture, top-five ranked results, and diagnostic overlays.
- Added development lookup tools for inspecting a specific `Figure` and `Item`
  reference and running matcher diagnostics against it even when it is outside
  the top-five results.
- Added larger, zoomable query-outline controls and automatic smoothed outline
  tracing so researchers can provide or correct exterior, interior, and fracture
  guidance more easily.
- Added adaptive rim/split hypotheses so ranking is less dependent on one exact
  hand-placed gold point.
- Added a cheap retrieval stage before expensive matching. The stage builds a
  cached index of continuous-outline and two-wall/ribbon descriptors, searches
  the full reference pool quickly, and forwards a bounded candidate set to the
  full matcher.
- Added retrieval diagnostics to result cards, including retrieval rank,
  outline-retrieval rank and score, ribbon-retrieval rank and score, and the
  method that selected the candidate.

### Changed

- Removed query Form from matcher input so shape-only validation is not given
  the answer as metadata.
- Kept cleaned contour previews in their source orientation; large alignment
  rotations now belong to the matcher stage rather than the cleaning stage.
- Reduced reliance on the centreline as a matching curve because query
  centrelines can bend toward accidental fracture ends.
- Treated query fracture markings as boundaries for useful evidence rather than
  as diagnostic wall curves.
- Tightened matching so both wall curves must align over one shared ordered
  interval, with extra penalties for unstable transforms, scale disagreement,
  and implausible tail alignment.
- Displayed reference identities as `Figure X.X Item Y` where linked metadata is
  available.

### Fixed

- Fixed matcher crashes caused by out-of-bounds contour indexing on short
  curves.
- Fixed matcher errors that displayed only `0.0` instead of useful failure
  detail.
- Fixed several diagnostic-view cases where cleaned references appeared flipped
  or unexpectedly rotated.
- Fixed result cards to show the cleaned reference artifact used by the matcher
  instead of the original pixelated mask.

### Validation

- Focused contour and matcher tests passed.
- Full Python test suite passed.
- Python compilation passed for the matcher module.
- JavaScript syntax validation passed for the matcher tab.
- On the 208-reference development project, cheap retrieval retained the known
  correct reference for all 80 saved gold-point variants of Queries 1, 3, 7,
  and 9. Cached retrieval searches took roughly 0.08 seconds after the first
  index build.

## 2026-07-24

### Added

- Added a **Profiles** workspace that turns approved rectangular vessel crops
  into diagnostic side-profile mask proposals for later shape-retrieval
  research.
- Added thick-body segmentation that uses strict and loose ink thresholds,
  distance-transform support, hole filling, and thin-branch removal to retain
  the filled ceramic cross-section while suppressing attached diameter and
  construction lines.
- Added reviewer tools for zooming, drawing and erasing mask pixels, removing a
  thin connected branch, rerunning segmentation inside a selected rectangle,
  undoing edits, approving a proposal, saving an edited mask, or marking a crop
  as having no usable profile.
- Stored automatic proposals separately from accepted masks and persisted
  proposal confidence, reasons, algorithm version, review status, and notes in
  `profile_review.json`.
- Added export support for choosing either whole-vessel crops or accepted
  side-profile masks. Profile exports retain the linked catalogue metadata and
  identify the image mode and profile-review status.

### Validation

- Manually reviewed approximately 50 varied Hesban profiles. About half were
  accepted without correction and the other half required only minor edits
  averaging approximately 10 seconds each; none in this pilot required full
  manual reconstruction.
- Focused profile-segmentation suite: 7 tests passed, covering connected thin
  lines, thin stubs, faint-outline restoration, ambiguous thick candidates,
  persistence, reviewed-mask protection, and explicit forced regeneration.

## 2026-07-23

### Fixed

- The manual measurement endpoint editor now shows the first point immediately when automatic measurement found no usable top line. A second click or double-click creates the other endpoint, and a new **Reset endpoints** control lets the researcher start a fresh pair.
- Row extraction now recovers a missed sequential row number only when the drawing sequence expects it and an abnormally tall physical gap contains room for it. This prevents a missed row 1 from being merged into row 2 while avoiding invented rows on continuation pages.
- Review & Link now deduplicates figure summaries by their stable figure key in both the API and browser, preventing temporary pause/resume snapshots from displaying the same figure more than once.
- Fixed a Review & Link request race where clicking a new figure before the previous figure detail finished loading could cache the old figure under the new figure's stable key. Stale responses and mismatched detail records are now discarded instead of replacing another sidebar figure.
- Every visible Review & Link warning can now be individually marked as reviewed and ignored with a saved reason and optional note. Ignoring a genuinely missing table row approves and exports the remaining resolved vessels while leaving that vessel unlinked instead of writing blank table values.
- Review & Link sidebar badges now use fresh summary status instead of stale cached figure details and continue updating while a table field is focused or has unsaved edits. The update changes only existing badge text and classes, preserving the selected page, table inputs, scrolling, overlays, and zoom.
- Fixed completed table rereads leaving an old evidence image without column or row lines. Focus left on the Re-read button no longer freezes evidence refreshes; only focused editable fields and unsaved edits defer workspace replacement.
- Dataset ZIP downloads now stream directly through the browser instead of first buffering the entire archive into a JavaScript Blob, avoiding intermittent zero-byte exports in embedded desktop browsers.
- Dataset ZIPs are now completely prepared and verified on disk before Chrome receives a stable download URL. The Export button shows **Preparing ZIP…**, reports preparation errors directly, and keeps the final same-page download target alive for the transfer, preventing Chrome’s “File wasn’t available on site” failure.
- Export now shows an indeterminate preparation bar followed by real byte-based ZIP download progress. Chrome receives the file only after SherdScope has downloaded and size-checked 100% of the prepared archive; export thumbnails also load lazily so they do not compete with the ZIP transfer.
- Fixed Chrome exposing an empty JavaScript stream for a prepared ZIP carrying an attachment header. Progress transfers now use a separate non-attachment response and reliable browser progress events before creating the final local download.
- Vessel-diameter Verify now saves immediately through the existing per-figure save queue. Manual endpoint and scale saves automatically retry once on the newest figure revision, preserving newer unrelated review work without making the researcher place the points again.
- Table row and column lines are now mandatory publication-viewer evidence and remain visible through number edits, measurement changes, saves, rerenders, and vessel-box visibility changes. Active rereads show a clear recalculation message; incomplete grids show the required next action.
- Fixed Adjust columns occasionally opening as a blank white canvas when the full-resolution table image loaded before its event handler was attached. The editor now registers loading/error handlers first and displays an explicit loading or failure message.
- Warning review and other ordinary autosaves now preserve the server-owned table boundary, manual column override, and diagnostic references instead of silently deleting the grid without scheduling a reread.

### Changed

- Removed the bottom-of-screen Advanced legacy tools section from Review & Link. The hidden legacy implementation and saved-project compatibility remain intact; PaddleOCR is the normal reading path.

### Tests

- Full Python suite: 131 tests passed, including a regression test proving warning autosaves preserve table grids and do not queue OCR work.

## 2026-07-22

### Added

- Added persistent per-detection vessel records containing an immutable detection ID, stable legacy-compatible vessel ID, original-page box, confidence, review state, and optional instance-mask provenance.
- Added box-first vessel review with add, delete, move, resize, individual approval, approve-all, and a separate optional mask-evidence toggle.
- Added original-resolution rectangular crops for approved boxes, a configurable pixel margin, and `cards/vessel_crops.json` provenance containing page coordinates, crop coordinates, confidence, identity, and mask evidence.
- Added regression coverage for nearby and overlapping detections, reordered YOLO output, ID mismatch rejection, box editing, coordinate translation, clipped margins, unmasked crop pixels, and legacy reviewed-data migration.

### Changed

- Stopped merging YOLO instances as the source of vessel identity. The combined page mask remains available only as visual evidence; new card extraction reads approved vessel boxes directly.
- Kept `page_mask_layer_N` identifiers as compatibility aliases for existing linkage, corrections, measurements, and exports while using immutable detection UUIDs internally.
- Limited connected-component recovery to a one-time migration path for projects created before per-detection sidecars existed.

### Fixed

- Prevented nearby or overlapping vessels from fusing into one card or changing identity because of mask connectivity.
- Prevented stale browser data from rebinding one detection ID to another vessel ID, and retained omitted, reviewed, deleted, or temporarily unmatched records across model reruns.

### Tests

- Full Python suite: 124 tests passed. Ruff, Python compilation, JavaScript syntax, diff validation, and desktop/narrow-width UI checks also passed.

## 2026-07-21

### Added

- Added one readable Hesban column profile shared by OCR, Review & Link, and CSV export, including the `Sq`/`Area` alias and repeated `Den`/`Color` sequencing.
- Added a full-resolution 23-line column editor with dragging, keyboard adjustment, reset, persistence, and priority page rereading.
- Added a persistent single-worker linkage queue with priority ordering, duplicate coalescing, safe checkpoint preemption, restart recovery, failure isolation, and automatic bulk resumption.
- Added lazy selected-figure and page-diagnostic endpoints, persistent job feedback, figure status badges, and one-click 2× publication-image zoom.
- Added a reviewer override layer that preserves edited cells and added/deleted rows across OCR rereads.

### Changed

- Replaced proportional and midpoint column fallback with strict detection of all 22 ordered anchors and exactly 23 edges. Incomplete or invalid headers now fail closed, skip row OCR, and request manual adjustment without stopping later figures.
- Moved per-cell OCR evidence from the main linkage state into atomic page sidecars and reduced frequent state polling to lightweight figure summaries.
- Changed linkage reread and boundary APIs to persistent asynchronous `202` jobs, and simplified Review & Link by placing legacy extraction controls in a collapsed Advanced section.
- Renamed the public square export field to `Sq/Area` while retaining the internal `table_square` key for existing projects.

### Fixed

- Made repeated headings such as `Den` and `Color` follow their physical left-to-right order instead of OCR confidence, preventing later columns from stealing earlier labels.
- Split long merged header tokens containing up to nine headings, including the real `Typ Siz Shap Den Ty/Sz Den` pattern found in the supplied Hesban PDF.
- Resolved duplicate vessel-number readings that occupy the same printed row, so page-number fragments such as `1` beside `11` no longer create one-pixel rows or empty records.
- Changed page-versus-cell OCR selection to geometry-first arbitration. A clean whole-page token can still beat a noisy focused crop, but a token crossing a column boundary cannot overwrite a safer cell reading or leak into a neighboring column.
- Added exact incomplete-header, row-conflict, page-geometry, accepted-source, and decision-reason information to lazy page diagnostics.
- Reused one thread-safe PaddleOCR engine per process and added an actual import/model health check so the UI reports the real local OCR problem instead of merely checking whether the package name exists.
- Kept the Review & Link layout free of horizontal page overflow at a 390-pixel viewport and verified the desktop and narrow layouts without browser console errors.
- Prevented a long-running desktop server from starting table OCR after its Python source files change. Review & Link now asks for a restart instead of silently using the older in-memory header splitter.
- Added an extractor version to saved page boundaries and marks figures produced by older OCR logic for rereading.
- Kept table rows in natural printed-number order after targeted page rereads. Existing saved states are also ordered when loaded, with publication-page order used only to resolve duplicate or blank labels.

### Tests

- Added deterministic coverage for strict header sequencing, aliases, repeated labels, group exclusion, scaled offsets, incomplete-page handling, manual overrides, priority/preemption, restart recovery, failure isolation, schema migration, and lightweight API responses.
- Real-corpus diagnostics on `Hesban Corpus-pages-split-4.pdf` detected all 22 anchors on 9 of 10 table pages. Page 16 now fails closed only because its printed `Sq` heading is not readable by OCR, and remains available for manual adjustment.
- A complete Figure 3.47 reread produced exactly rows 1-21, no one-pixel rows, no blank extracted cells, and no accepted cross-column page tokens.
- The current extractor found all 22 headings on all 13 table pages in `Hesban_Corpus-pages-split-3.pdf`, including merged Non-Plastics/Voids and Surface Treatment headings.
- Full Python suite: 114 tests passed; Python compilation, Ruff, JavaScript syntax, and desktop/narrow-width visual acceptance also passed.

## 2026-07-19

### Changed

- Reorganized the Python implementation into purpose-based packages: catalogue linkage, measurements, sidecars, and research export now live in `catalog/`; OCR, model architecture, PDF handling, configuration, and image-processing code live in `processors/`; and project workspace management lives in `services/`.
- Kept `app.py` as the single obvious root-level Python launcher and preserved the existing Flask routes, file formats, browser asset paths, and platform launch commands.
- Updated application, route, and test imports for the new package layout, and revised the README structure guide so it matches the repository.

### Removed

- Removed the unreferenced duplicate model downloader, obsolete private sidecar alias, commented-out duplicate implementations, and broad compatibility re-exports after repository-wide reference tracing showed that the application did not use them.
- Removed the old root-level implementation modules after moving their active code into the focused packages. No compatibility shims were retained because no repository, documentation, test, or launcher consumer used the old import paths; unknown external scripts must update their imports.

### Tests

- Full Python suite: 93 tests passed.
- Ruff lint/static analysis passed for the product and test code.
- Python compilation passed for all tracked and newly organized source and test files, and importing `app` succeeded with all 78 Flask routes registered.
- JavaScript syntax validation passed for all 8 tracked scripts.
- Launch-path, template asset, stale-import, and Git diff validation passed. Diff validation reported only the repository's existing Windows line-ending warnings.

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
