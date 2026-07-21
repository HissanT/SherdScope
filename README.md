# SherdScope

<p align="center">
  <img src="static/imgs/sherdscope-logo.svg" alt="SherdScope logo" width="150">
</p>

**An explainable archaeological pottery workflow for turning published catalogues into reviewed research datasets.**

SherdScope helps researchers extract vessel-profile drawings from pottery
publication PDFs, connect each drawing to its printed catalogue-table row, and
export the reviewed result as a clean CSV and image dataset.

The long-term research goal is broader: compare real pottery sherd profiles
against a reviewed catalogue and return ranked published parallels. This
repository currently implements the catalogue digitization part of that goal.
Shape normalization and similarity retrieval are planned next; they are not
presented here as completed features.

## Current status

The Hesban catalogue-digitization MVP is operational and under active
real-corpus validation. The current workflow can:

- render publication PDFs at configurable high resolution;
- segment vessel drawings using the PyPotteryLens detection models;
- preserve each card's source page, bounding box, and stable internal identity;
- recognize printed figure and vessel numbers with local PaddleOCR;
- locate all 22 configured Hesban headings, exact column boundaries, rows,
  continuation pages, and closing rules using deterministic image processing;
- fail closed and request column adjustment when any required heading is
  missing, duplicated, ambiguous, or out of order;
- link drawings to table rows through validated `(figure, vessel number)` keys;
- autosave reviewer corrections without allowing background OCR refreshes to
  erase the active figure, scroll position, or draft;
- calibrate the standard `0-10 CM` publication ruler and estimate illustrated
  rim diameter, with manual correction when detection needs attention;
- expose targeted OCR evidence for difficult cells, including the
  Non-Plastics Type column;
- queue interactive rereads ahead of bulk extraction, checkpoint the bulk run,
  and resume it automatically after the priority work finishes;
- approve figures individually and withhold ambiguous joins;
- export a clean 25-column research CSV or a complete dataset ZIP.

PaddleOCR and OpenCV run locally. Generative AI and paid API calls are not
required for the normal Hesban workflow.

## Why review remains important

Catalogue scans vary in quality and layout. OCR can misread small characters,
segmentation can join nearby marks, and an automatic diameter can be ambiguous.
SherdScope therefore treats automation as a draft-producing tool:

- vessel and table numbers must form a unique match;
- unresolved or duplicate identities cannot be approved;
- researchers can edit numbers, rows, cells, scale, and diameter;
- only approved figures enter the final research dataset;
- OCR crops, boundaries, source pages, and internal provenance remain available
  for auditing even though technical fields are omitted from the clean export.

## Installation

### Requirements

- Python 3.12
- 8 GB RAM minimum; 16 GB recommended
- a modern browser
- an optional NVIDIA GPU for faster vessel detection

Model weights, copyrighted publication PDFs, generated projects, and private
research data are intentionally not stored in this repository.

### Windows

Double-click `SherdScope_WIN.bat`, or run it from PowerShell:

```powershell
.\SherdScope_WIN.bat
```

### Linux or macOS

```bash
chmod +x SherdScope_UNIX.sh
./SherdScope_UNIX.sh
```

### Manual setup

```bash
python -m venv venv
```

Activate the environment, then install the application and local OCR
dependencies:

```bash
pip install -r requirements.txt
pip install -r requirements-ocr.txt
python app.py
```

Open [http://localhost:5001](http://localhost:5001).

The required vessel-detection and orientation models are downloaded from the
original PyPotteryLens Hugging Face repository on first use.

## Research workflow

### 1. Create and prepare a project

1. Create a project from the SherdScope home screen.
2. Upload the publication PDF.
3. Render the PDF pages.
4. Run vessel detection.
5. Review the detected masks and correct them when necessary.
6. Extract the drawing cards.

SherdScope records PDF-page order in `page_manifest.json`, including split-page
information where applicable.

### 2. Review and link the catalogue

Open **Review & Link** and select **Read and Link Tables**. Figures are processed
one at a time so completed figures can be reviewed while later figures wait.
The persistent job strip reports queued, running, paused, completed, and failed
work. A figure or measurement reread can be queued during the bulk run; it starts
after the current OCR call finishes, then the bulk run resumes from its saved
checkpoint. Interrupted work is recovered from `metadata_jobs.json` on restart.

The desktop launcher intentionally does not hot-reload Python because doing so
can initialize the OCR and vision models twice. If backend source files change
while SherdScope is open, Review & Link blocks new OCR work and asks for a
restart. Saved boundaries carry an extractor version; figures made by older OCR
logic are clearly marked for a fresh **Read and Link Tables** run.

For the selected figure:

- use the publication viewer to compare the drawing boxes and printed numbers;
- correct any unresolved or incorrect vessel number;
- inspect the detected scale or rim diameter only when it needs attention;
- edit table cells, add or delete rows, and undo deletions;
- select **Adjust columns** when a page is marked for column review, drag any of
  the 23 lines (or use the arrow keys), then choose **Save and re-read**;
- use the group shortcuts to move between Identity, Fabric, Non-Plastics,
  Voids, Surface, and Finish;
- approve the figure when the readiness checklist confirms a unique join.

For Hesban 11, the default table search window is the drawing page through the
next two logical pages. Less common actions, including assigning another page
or rereading one figure, are available from **More**.

### 3. Diagnose a difficult OCR cell

If any value is missing, suspicious, or appears to come from a neighboring
column:

1. open the figure in **Review & Link**;
2. expand the OCR diagnostic inspector;
3. compare the exact crop, raw OCR token, confidence, overlapping page tokens,
   page geometry, accepted value, and the reason that source won;
4. correct the cell manually if the printed page and OCR disagree;
5. use **More -> Re-read this figure** after an OCR improvement or page
   reassignment.

The diagnostic is saved in a per-page sidecar and loaded only when this inspector
opens, so an empty cell can be investigated without placing the complete corpus
evidence in every status poll.

Whole-page OCR and focused-cell OCR are deliberately not trusted blindly. A
clear whole-page token remains useful when a focused crop is noisy or blank, but
it is accepted only when its printed box safely belongs to that cell. A token
that crosses a column boundary is withheld instead of being copied into the
neighboring column. The focused crop therefore does not automatically win; safe
geometry is considered first, then recognition quality.

The application also checks that PaddleOCR can actually be imported and its
shared local model can be initialized. When the local installation is broken,
Review & Link displays the real startup error rather than saying only that OCR
is unavailable.

### Hesban column profile

`catalog/profiles.py` is the single editable source for extraction, review
headings, grouping, and CSV labels. Each `ColumnSpec` declares a stable storage
key, accepted OCR aliases, header tier, group, UI label, and CSV label. The
ordered Hesban anchors are:

```text
No. Type Sq|Area Loc Pail Reg
Exterior Core Interior Typ Siz Shap Den Ty/Sz Den
Man
Ext Color Int Color
Decor Fire
```

`Fabric Color`, `Non-Plastics`, `Voids`, and `Surface Treatment` are validation
and display groups, not boundary anchors. All 22 anchors are required. The left
edge and each internal edge are placed a DPI-scaled lead distance before the
corresponding heading; the detected table-rule endpoint supplies the final edge.
Saved manual edges take precedence on every later reread. Internal projects keep
the stable `table_square` key, while exports call the field `Sq/Area`.

### 4. Export the research dataset

Open **Export** after approving the figures you want to use.

- Search or review vessel masks.
- Include or exclude individual masks.
- Select **Download CSV** for metadata only.
- Select **Download Dataset ZIP** for metadata plus images and documentation.

The ZIP contains:

```text
metadata.csv
images/
data_dictionary.csv
export_summary.txt
```

Exported image names are stable and readable, for example
`HES_Fig2-1_No3.png`. The same filename appears in `metadata.csv`.

## Researcher-facing CSV

The final CSV is created from approved linkage data rather than exposing the
internal working table. It uses UTF-8 with a byte-order mark for Excel
compatibility, preserves punctuation and multiline archaeological values, and
exports blank cells as genuinely blank.

The fixed export fields are:

1. Image Filename
2. Figure
3. No.
4. Vessel Type
5. Rim Diameter (cm)
6. Sq/Area
7. Locus (Loc)
8. Pail
9. Registration (Reg)
10. Fabric Color - Exterior
11. Fabric Color - Core
12. Fabric Color - Interior
13. Non-Plastics - Type
14. Non-Plastics - Size
15. Non-Plastics - Shape
16. Non-Plastics - Density
17. Voids - Type/Size
18. Voids - Density
19. Manufacture
20. Surface Treatment - Exterior
21. Surface Treatment - Exterior Color
22. Surface Treatment - Interior
23. Surface Treatment - Interior Color
24. Decoration
25. Firing

Technical keys, fingerprints, pixel coordinates, raw OCR evidence, reviewer
revisions, and internal filenames stay out of this final schema.

## Local project data

Runtime projects are stored locally and ignored by Git:

```text
projects/<project-id>/
|-- page_manifest.json
|-- pdf_source/
|-- images/
|-- masks/
|-- cards/
|   |-- mask_info.csv
|   |-- mask_info_annots.csv
|   |-- metadata_linkage.json
|   |-- metadata_jobs.json
|   `-- metadata_diagnostics/<figure>/<page>.json
|-- export_settings.json
`-- exports/
```

The internal `mask_info.csv` remains backward compatible. Approved linkage,
measurement evidence, reviewer edits, and warnings are staged in
`metadata_linkage.json`; large per-cell evidence lives in page sidecars, and the
resumable queue lives in `metadata_jobs.json`. Schema-v1 linkage files are
migrated atomically with one recovery backup. The clean public schema is
generated only at export.

## Validation

Install the development requirements and run:

```bash
pip install -r requirements-dev.txt
pytest -q
```

At the current milestone:

- the complete Python test suite passes;
- JavaScript syntax checks pass for the main, Review & Link, and Export scripts;
- Python compilation and Git diff validation pass;
- a local PP-OCRv5 acceptance check read all 19 Non-Plastics Type values in
  Hesban Figure 2.1 across its two table pages.

The automated suite covers manifests, table geometry, OCR parsing and retries,
figure normalization, unique joins, autosave revisions, measurements, warning
handling, approval, CSV persistence, per-figure rereads, and dataset export.
Real-corpus results still require researcher review.

## Repository structure

The repository root is intentionally limited to the Flask launcher,
documentation, dependency lists, version metadata, and platform launchers.
Implementation is grouped by responsibility:

- `app.py` - the obvious Flask entry point and application API assembly;
- `catalog/` - figure/table linkage, measurements, sidecar persistence, and
  clean research-dataset export;
- `processors/` - PDF rendering, OCR, model architecture, and the image
  processing pipeline;
- `routes/` - focused Flask route groups;
- `services/` - project-workspace management;
- `static/` and `templates/` - browser assets and the application page;
- `tests/` - unit and Flask integration coverage.

See [CHANGELOG.md](CHANGELOG.md) for detailed implementation history.

## Known limits and next research stage

- The publication profile is currently designed and tested around Hesban 11;
  other catalogue layouts will require their own profiles.
- Automatic segmentation, small-cell OCR, scale calibration, and rim diameter
  can still require manual correction on poor scans.
- SherdScope currently builds the reviewed reference corpus. It does not yet
  take a photographed sherd and return ranked matches.
- The next planned research stage is to represent each cross-section as
  exterior, interior, and fracture curves; exclude the accidental fracture from
  the main shape score; normalize scale and orientation; and compare
  explainable geometric retrieval methods before considering learned
  embeddings.

## Attribution and license

SherdScope is derived from
[PyPotteryLens](https://github.com/lrncrd/PyPotteryLens) by Lorenzo Cardarelli
and contributors. SherdScope has a separate name and interface, while retaining
the upstream copyright, license, and attribution required by the GNU GPL. See
[NOTICE.md](NOTICE.md) for details.

SherdScope is distributed under the
[GNU General Public License v3.0](LICENSE).
