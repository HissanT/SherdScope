# SherdScope

**An explainable archaeological pottery pipeline for building reviewed reference corpora and retrieving published parallels.**

SherdScope is being developed to connect photographed pottery sherds with
published archaeological catalogues. The long-term system will segment a
sherd's diagnostic profile, normalize its shape, search a reviewed catalogue,
combine geometric and archaeological metadata evidence, and return ranked
published parallels for expert assessment.

This repository currently contains the production-ready **catalogue
digitization and metadata-linking module**. It converts pottery-publication PDFs
into individual drawing cards and links each drawing to its corresponding table
row through a reviewable, provenance-preserving workflow.

## Current capabilities

- Render publication PDFs at configurable 200–600 DPI.
- Detect and segment pottery drawings with the PyPotteryLens vision models.
- Review and correct detected masks in the browser.
- Preserve stable card identities, source pages, and bounding boxes.
- Recognize printed figure and vessel numbers with local PaddleOCR.
- Detect Hesban table boundaries using deterministic image processing.
- Link drawings and metadata using validated `(figure, vessel number)` keys.
- Review, correct, approve, or reject extracted figure-table matches.
- Export approved drawing metadata to CSV while preserving manual edits.
- Keep ambiguous or incomplete matches out of the final dataset.

Generative AI is optional. The Hesban linking workflow uses local OCR and
OpenCV-based table geometry by default.

## Why this matters

Published pottery catalogues contain valuable drawings and archaeological
metadata, but the two are usually connected only through printed figure and
item numbers. SherdScope turns that relationship into structured, auditable
data without treating OCR output as unquestioned ground truth. Human approval
remains required before extracted metadata enters the working CSV.

## Installation

### Requirements

- Python 3.12
- 8 GB RAM minimum; 16 GB recommended
- A modern browser
- Optional NVIDIA GPU for faster drawing detection

### Windows

Run:

```text
SherdScope_WIN.bat
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

Activate the environment and install the application dependencies:

```bash
pip install -r requirements.txt
pip install -r requirements-ocr.txt
python app.py
```

Then open [http://localhost:5001](http://localhost:5001).

The required PyPotteryLens drawing-detection and orientation models are
downloaded automatically from the original project's Hugging Face repository
on first use. Model weights are intentionally not stored in Git.

## Workflow

1. Create a project.
2. Upload and render a pottery-publication PDF.
3. Run drawing detection.
4. Review and correct the masks.
5. Extract drawing cards.
6. Open the Tabular workflow and run **Link Figure Tables**.
7. Review figure IDs, vessel numbers, table boundaries, and extracted cells.
8. Approve validated figures.
9. Export the reviewed cards and metadata.

For Hesban 11, the linker searches the drawing page and the following two
logical pages by default. Reviewers can manually assign other table pages and
rerun OCR for a single figure.

## Project data

Runtime projects are stored locally and ignored by Git:

```text
projects/<project-id>/
├── page_manifest.json
├── pdf_source/
├── images/
├── masks/
├── cards/
│   ├── mask_info.csv
│   ├── mask_info_annots.csv
│   └── metadata_linkage.json
├── cards_modified/
└── exports/
```

No copyrighted corpus pages, private research datasets, generated projects, or
model weights are included in this repository.

## Validation

Install the development requirements and run:

```bash
pip install -r requirements-dev.txt
pytest -q
```

The test suite covers page manifests, table-boundary detection, OCR result
parsing, figure-number normalization, unique joins, review edits, CSV
persistence, per-figure reruns, and export behavior. Real-corpus results still
require human review because scan quality and publication layout vary.

## Roadmap

- Validate a reviewed Hesban pilot corpus across varied layouts and periods.
- Add normalized diagnostic-profile representations.
- Compare explainable geometric retrieval methods.
- Integrate photographed-sherd segmentation as a separate evaluated module.
- Add metadata-aware reranking and top-k parallel suggestions.

The intended output is decision support for archaeologists, not automatic
identification: every result should remain traceable to its published drawing
and catalogue evidence.

## Attribution and license

The catalogue application is derived from
[PyPotteryLens](https://github.com/lrncrd/PyPotteryLens) by Leonardo Cardarelli
and contributors. See [NOTICE.md](NOTICE.md) for details.

SherdScope is distributed under the [GNU General Public License v3.0](LICENSE).

