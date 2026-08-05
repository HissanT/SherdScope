# Project 18 incremental review recovery

The Review Profiles client was sending `force: true` for every Generate action.
That reset the review state and replaced accepted masks for all 1,099 previously
approved cards. The earlier reviewed masks remained available in the immutable
`profile_unet_v1` curator drafts, so they were restored at their current card
dimensions. A complete pre-restore copy of the review JSON and accepted-mask
directory is stored under the project's `cards/profile_recovery_backups` folder.

After recovery, Project 18 contains 1,238 cards: 1,099 restored approved cards
and 139 unresolved cards belonging to `Eight_new_Figures_Hesban.pdf`. No old
card must be reviewed again. Proposal generation now defaults to missing-only,
supports a source-PDF filter, and will not overwrite reviewed accepted masks.
The main reviewer now uses the curator's full high-resolution path rather than
only copying its G button. Each crop is rendered directly from its source PDF
at 600 DPI. Editing, previewing, rectangle reruns, and Autofill (G) operate on
that high-resolution image; the accepted binary mask is downsampled safely only
when it is saved to the existing project. Its workspace also follows the
curator's wide editor/narrow mask-preview layout and includes Pan (P), Original
Auto (O), visible brush/eraser rings, zoom, undo, and the review hotkeys.

Review & Link now lists individual PDFs and starts with the newest source.
Running it for that source preserves figures belonging to the other PDFs.
Canonical contour construction likewise skips unchanged complete entries and
builds only missing or changed entries.

Validation completed with Python and JavaScript syntax checks and 103 focused
tests covering profile generation/recovery, contours, linkage, and linkage API
behavior. All 103 passed.
