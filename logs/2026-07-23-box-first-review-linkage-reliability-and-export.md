# SherdScope box-first review, linkage reliability, and export

Date: 2026-07-23

## Summary

This implementation pass made reviewed vessel bounding boxes authoritative,
strengthened row and figure identity handling, made Review & Link state safer
during edits and rereads, and rebuilt dataset ZIP delivery so Chrome receives a
complete verified archive with visible progress. Existing saved projects remain
compatible and do not need to be recreated.

## Bounding-box-first vessel records

- Preserved every YOLO detection independently with its original bounding box,
  confidence, immutable detection ID, stable vessel ID, and optional mask
  provenance.
- Stopped merging nearby detections and reconstructing vessels with connected
  components during normal processing.
- Made rectangular boxes the primary review method. Researchers can add,
  delete, move, resize, and approve boxes.
- Kept masks as optional supporting evidence only. Masks do not erase pixels
  from later crops.
- Added original-resolution rectangular crops from approved reviewed boxes with
  a configurable margin.
- Stored crop coordinates, original page coordinates, confidence, vessel
  identity, and optional mask provenance.
- Preserved legacy `page_mask_layer_N` vessel identifiers so existing linkage,
  corrections, measurements, and exports remain compatible.
- Protected reviewed, deleted, manual, and temporarily unmatched detections
  across model reruns, including detection-ID mismatch checks.
- Added coverage for nearby and overlapping detections, reordered model output,
  box editing, coordinate translation, crop margins, unmasked crop pixels, and
  migration of existing reviewed data.

## Measurement review

- Fixed the manual endpoint editor when automatic rim measurement had not found
  a top line.
- The first point now appears immediately; a second click or double-click adds
  the other endpoint.
- Added Reset endpoints.
- Verify saves immediately through the existing per-figure save queue.
- Manual endpoint and scale saves retry once on the newest reviewer revision
  while preserving unrelated newer work.

## Row and identity reliability

- Added conservative recovery for a missed sequential row only when both the
  expected drawing-number sequence and a physically oversized row gap support
  the missing number.
- Kept continuation pages from inventing earlier rows.
- Bumped the OCR extractor version so older boundaries are visibly marked for
  rereading.
- Deduplicated figure summaries by stable figure key on both the server and
  browser.
- Fixed a browser request race that could cache an older figure under a newly
  selected figure and make one sidebar entry appear to replace another.
- Discarded stale figure-detail responses whose returned stable key does not
  match the requested key.

## Warning review and approvals

- Made every visible warning individually reviewable with a server-validated
  reason and optional note.
- Allowed a genuinely missing table row to be reviewed and ignored. Resolved
  vessels can then be approved and exported, while the ignored vessel remains
  explicitly unlinked instead of receiving blank metadata.
- Preserved warning decisions and their original server-owned timestamps across
  unrelated autosaves.
- Updated readiness and sidebar badges from fresh lightweight summaries without
  replacing active inputs, page selection, scrolling, overlay state, or zoom.

## Persistent table evidence

- Made row and column lines core review evidence rather than an optional
  overlay.
- Kept the grid visible through number edits, cell edits, measurement changes,
  warning decisions, approvals, saves, rerenders, and vessel-box visibility
  changes.
- Found and fixed the underlying autosave bug: the browser submitted
  `table_pages` with every save, and the server rebuilt those records without
  their boundary, manual-column, and diagnostic fields.
- Ordinary saves now preserve the complete server-owned table-page record and
  do not queue OCR.
- Existing pages whose boundaries were already removed by the old bug require
  one final reread; later ordinary saves retain the replacement grid.
- Active rereads show that rows and columns are being recalculated. Incomplete
  evidence shows the required next action.
- Fixed Adjust columns opening as a white canvas by attaching image load/error
  handlers before assigning the image source.
- The manual column editor continues to request the raw full-resolution image
  and draws its own 23 draggable lines.

## Review & Link cleanup

- Removed the bottom Advanced legacy-tools section from the main Review & Link
  screen while retaining saved-project compatibility.
- Kept PaddleOCR as the normal extraction path.
- Limited editing-state protection to actual editable controls and unsaved
  drafts, so focus left on an action button no longer freezes completed reread
  refreshes.

## Dataset export reliability

- Confirmed that actual project archives were structurally valid before
  changing the browser workflow.
- Replaced direct generation-inside-download with a two-stage process:
  SherdScope first builds the complete ZIP atomically under
  `exports/prepared`, then exposes a stable finished-file URL.
- Added an indeterminate preparation bar and a byte-based 0--100 percent
  transfer bar.
- Added an exact byte-count check before the browser receives the final local
  download.
- Separated progress transfer from attachment delivery because Chrome could
  consume an attachment response itself while exposing an empty body to
  JavaScript.
- Progress transfer now uses a non-attachment response and browser progress
  events. The normal attachment route remains available.
- Export thumbnails load lazily so a large review list does not compete for all
  browser connections.
- Prepared archives older than 24 hours are eligible for cleanup during later
  preparation.
- The current saved project produced a verified 3,464,703-byte ZIP whose
  reported and downloaded sizes matched exactly and whose archive integrity
  check passed.

## Existing-project compatibility

- A fresh project is not required.
- Restarting SherdScope loads the new implementation.
- Existing reviewed boxes, manually added boxes, stable vessel IDs, linkage
  corrections, manual table boundaries, warning decisions, approvals, and
  measurements remain in their existing sidecars.
- Explicit rereads run the new OCR/linkage code. A simple resume can retain
  completed units, so use Re-read this figure or Save and re-read when a
  completed result must be recalculated.
- Prepared ZIP files do not change reviewed research state.

## Validation

- Complete Python suite: 131 passed.
- Ruff: passed.
- Python compilation: passed.
- JavaScript syntax: passed.
- Git diff validation: passed.
- Real saved-project ZIP preparation and transfer: passed.

## Research boundary

This pass prepares clean reviewed bounding-box crops for the future
diagnostic-profile stage. It does not implement diagnostic-blob separation,
three-curve profile extraction, or catalogue-profile matching.
