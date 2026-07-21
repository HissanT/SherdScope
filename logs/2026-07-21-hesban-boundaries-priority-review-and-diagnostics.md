# SherdScope Hesban boundaries, priority processing, review UI, and diagnostics

Date: 2026-07-21

## Requested outcome

Make Hesban table extraction accurate and scalable, stop column drift, allow manual boundary repair, preserve researcher edits across rereads, prioritize interactive work over long corpus jobs, reduce polling payloads, and simplify Review & Link.

## Implemented features

- Added one editable column profile shared by OCR, UI labels, group labels, and CSV export.
- Defined the 22 physical anchors in order: No., Type, Sq/Area, Loc, Pail, Reg, Exterior, Core, Interior, Typ, Siz, Shap, Den, Ty/Sz, Den, Man, Ext, Color, Int, Color, Decor, and Fire.
- Kept Fabric Color, Non-Plastics, Voids, and Surface Treatment as grouping labels only. They do not create column edges.
- Kept `table_square` as the stable internal field and changed the public label to `Sq/Area`.
- Replaced proportional/midpoint fallback boundaries with 23 edges derived from the actual ordered headings and the table rule.
- Required complete, unique, monotonic header evidence. Incomplete pages fail closed and remain manually editable while later figures continue.
- Added long merged-header splitting and physical-order handling for repeated `Den` and `Color` labels.
- Added a full-resolution 23-line column editor with mouse dragging, keyboard movement, reset, cancel, and Save and re-read.
- Persisted normalized manual boundaries as authoritative overrides.
- Added a reviewer override layer so edited cells and added/deleted rows survive compatible OCR rereads.
- Added a persistent single-worker linkage queue with priority 0 boundary rereads, priority 10 figure/measurement rereads, and priority 100 bulk work.
- Added duplicate coalescing, FIFO ordering at equal priority, safe checkpoint preemption, restart recovery, failure isolation, and bulk resumption.
- Moved heavy cell diagnostics into atomic page sidecars and made selected-figure and diagnostic data load lazily.
- Changed interactive reread endpoints to asynchronous `202` jobs.
- Added persistent job/action feedback, figure status badges, stable viewer state, and click-to-zoom publication images.
- Simplified the primary Review & Link workspace and moved legacy controls into Advanced.

## OCR diagnostic corrections

- Made physical left-to-right position authoritative when repeated headings have different OCR confidence.
- Split merged strings such as `Typ Siz Shap Den Ty/Sz Den` into their six physical anchors.
- Clustered competing row-number readings on the same printed line so readings such as `11` and `1` cannot create a one-pixel false row.
- Changed page-versus-cell arbitration to geometry first. Whole-page OCR remains useful, but a token crossing a column boundary cannot overwrite a safer focused-cell reading.
- Kept focused-cell OCR from becoming automatically authoritative because narrow cell crops can be noisy.
- Added diagnostic explanations for missing headings, row conflicts, token geometry, accepted source, and selection reason.
- Reused one thread-safe PaddleOCR engine and exposed real import/model initialization failures.
- Added extractor-version and edited-source restart checks so a running desktop process cannot silently use stale Python logic.

## Supplied-PDF findings

### Split-4

- Checked ten table pages at 400 DPI.
- Nine pages detected all 22 anchors after the repeated/merged-heading corrections.
- Page 16 safely remained incomplete because OCR did not provide the required `Sq` anchor.
- Figure 3.47 produced rows 1 through 21 with no collapsed one-pixel rows, no blank accepted cells in the diagnostic run, and no unsafe cross-column page tokens.

### Split-3

- The current extractor detected all 22 headings on all 13 table pages.
- An older still-running backend had saved incomplete boundaries because its splitter handled only shorter merged tokens. Restart detection now prevents that stale-code failure.
- Figure 3.22 row 1 was verified as a separate 136-pixel band (`y=691–827`) with all 22 accepted cells.
- Figure 3.22 row 2 was verified as its own band (`y=827–963`). Its repeated fabric, non-plastic, void, surface, decor, and firing values match the publication; its identifying values are distinct:
  - Row 1: `A.1 / 58 / 136 / 1900`
  - Row 2: `A.6 / 44 / 46 / 155`
- The apparent missing row 1 was a recombination-order problem, not an OCR failure. A targeted reread retained page-266 rows 25–31, then appended reread page-265 rows 1–24.
- Added a shared natural printed-number ordering invariant. Fresh extraction, retries, targeted rereads, manual-boundary rereads, and existing saved states now produce `1, 2, 3 …` order. Publication-page order is a stable tie-breaker for duplicate or blank labels.
- Confirmed that the Surface Treatment `Int` edge follows the intended next-heading rule. On Figure 3.22, `Int` ends at x=2788, its right edge is x=2846, and the following `Color` begins at x=2854. The field is narrow because the printed table gives code fields less space than color-description fields.

## Verification

- Full Python suite: 114 passed.
- Ruff: passed.
- Python compilation: passed.
- JavaScript syntax and responsive browser checks from the main implementation pass: passed.
- Added regression coverage for complete/merged/repeated headings, incomplete-header fail-closed behavior, row-anchor conflicts, manual boundaries, priority jobs, migration, lazy API payloads, reviewer overrides, and natural ordering after targeted page replacement.

## Important behavior for researchers

- Restart SherdScope after Python source changes; the UI now warns when this is required.
- Missing required headings do not trigger guessed extraction.
- Manual boundaries override automatic boundaries until reset.
- Interactive rereads wait only for the current atomic OCR call, then take priority.
- Existing reviewer corrections are reapplied after compatible rereads.
- Existing misordered saved rows are corrected when their linkage state is loaded.
