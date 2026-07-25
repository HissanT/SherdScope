# 2026-07-24 — Diagnostic Profile Segmentation and Manual Pilot

## Purpose

SherdScope previously stopped at approved rectangular crops. Those crops
preserved the publication evidence safely, but they still contained diameter
lines, construction axes, labels, and other printed material that should not
enter a shape matcher. The new Profiles stage creates a reviewed binary mask of
the filled ceramic cross-section while keeping the original crop unchanged.

## Implemented workflow

The Profiles workspace operates on approved vessel crops. It generates an
automatic mask proposal, shows the proposal over the original crop, and keeps
the automatic and accepted versions separate.

The automatic method does more than retain the largest connected component. It
uses strict and loose grayscale thresholds, a distance transform to locate a
thick ceramic core, hole filling, nearby-outline restoration, and
thickness-supported branch removal. Candidate blobs are ranked using area,
thickness, compactness, position, and vertical extent. Ambiguous or unusually
small selections receive diagnostic reasons and reduced confidence.

The reviewer can:

- approve an unchanged proposal;
- draw or erase mask pixels;
- click a thin attached branch for removal;
- select a rectangle and rerun the proposal only within that area;
- undo edits or reset to the automatic proposal;
- save an edited mask or mark the crop as having no usable profile.

SherdScope saves proposal evidence, algorithm version, review status, and notes
in `profile_review.json`. Automatic masks live in `profiles/auto/`; accepted
masks live in `profiles/accepted/`. Ordinary generation preserves reviewed
work unless forced regeneration is explicitly requested.

Export now supports whole-vessel crops or accepted side-profile masks. The
profile option keeps the same reviewed catalogue rows and records the image
mode and profile status in the exported dataset.

## Manual validation

Approximately 50 varied Hesban profiles were reviewed manually:

- roughly half were correct without any edit;
- roughly half needed only a minor correction;
- minor corrections averaged approximately 10 seconds each;
- no profile in this pilot required complete manual reconstruction.

Across the full sample, the observed correction burden was therefore about five
seconds per profile before navigation and decision time. This is a practical
workflow result from a development sample, not an independent corpus-wide pixel
accuracy measurement.

## Interpretation and decision

The profile-preparation approach is promising enough to support the first
retrieval experiment. Its value is not fully automatic segmentation; its value
is turning a slow manual tracing task into rapid proposal review and correction.
Further heuristic tuning is paused until a broader or genuinely different
sample exposes a repeated failure mode.

The immediate research dataset will be deliberately small. Approximately 30
Hesban drawings will be selected because corresponding real sherds and
independently identified parallels are available with some confidence. Each
real-shard–Hesban pair will be documented before matching, and the accepted
profile masks will form the frozen pilot reference set.

## Validation

The focused automated profile-segmentation suite passed 7 tests. It covers
connected construction lines, thin stubs, faint outline recovery, ambiguous
candidates, sidecar and mask persistence, protection of reviewed masks, and
explicit forced regeneration.
