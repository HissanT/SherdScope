"""Build the Hesban Classical Queries PDF with p, p+1, and p+2 per query."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import pypdfium2 as pdfium
from pypdf import PdfReader, PdfWriter


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE = SCRIPT_DIR.parents[1]
SOURCE_PDF = WORKSPACE / "DL ArchProject" / "Hesban Corpus.pdf"
OUTPUT_PDF = SCRIPT_DIR / "Hesban_Classical_Queries_p-p1-p2.pdf"
MANIFEST_CSV = SCRIPT_DIR / "Hesban_Classical_Queries_manifest.csv"

QUERIES = [
    "3.1.8",
    "3.2.4",
    "3.3.6",
    "3.4.9",
    "3.5.8",
    "3.6.10",
    "3.7.7",
    "3.8.5",
    "3.9.8",
    "3.11.1",
    "3.12.5",
    "3.13.6",
    "3.14.10",
    "3.15.4",
    "3.16.1",
    "3.17.9",
    "3.18.5",
    "3.19.5",
    "3.20.20",
    "3.21.16",
    "3.22.30",
    "3.23.2",
    "3.24.18",
    "3.25.4",
    "3.26.12",
    "3.27.19",
    "3.29.4",
    "3.43.4",
    "3.47.2",
    "3.81.5",
]

FIGURE_HEADING_RE = re.compile(r"Figure\s+(\d+\.\d+)", re.IGNORECASE)


def figure_from_query(query: str) -> str:
    major, figure, _item = query.split(".")
    return f"{major}.{figure}"


def build_figure_index(pdf_path: Path) -> dict[str, int]:
    """Map each figure to the first PDF page bearing its top-of-page heading."""
    document = pdfium.PdfDocument(pdf_path)
    index: dict[str, int] = {}
    for pdf_page in range(1, len(document) + 1):
        page = document[pdf_page - 1]
        width, height = page.get_size()
        top_text = page.get_textpage().get_text_bounded(
            left=0, bottom=height - 145, right=width, top=height
        )
        match = FIGURE_HEADING_RE.search(top_text)
        if match:
            index.setdefault(match.group(1), pdf_page)
    return index


def main() -> None:
    if len(QUERIES) != 30 or len(set(QUERIES)) != 30:
        raise RuntimeError("Expected exactly 30 unique Classical Queries.")
    if not SOURCE_PDF.exists():
        raise FileNotFoundError(SOURCE_PDF)

    figure_index = build_figure_index(SOURCE_PDF)
    missing = [
        query for query in QUERIES if figure_from_query(query) not in figure_index
    ]
    if missing:
        raise RuntimeError(f"Figure headings not found for: {', '.join(missing)}")

    reader = PdfReader(SOURCE_PDF)
    writer = PdfWriter()
    manifest_rows: list[dict[str, object]] = []

    for query_number, query in enumerate(QUERIES, start=1):
        figure = figure_from_query(query)
        source_page = figure_index[figure]
        source_pages = [source_page, source_page + 1, source_page + 2]
        if source_pages[-1] > len(reader.pages):
            raise RuntimeError(f"Page range exceeds source PDF for {query}")

        output_start = len(writer.pages) + 1
        for pdf_page in source_pages:
            writer.add_page(reader.pages[pdf_page - 1])
        writer.add_outline_item(
            title=f"Query {query_number}: {query} (Figure {figure})",
            page_number=output_start - 1,
        )

        manifest_rows.append(
            {
                "query_number": query_number,
                "query": query,
                "figure": figure,
                "source_p": source_page,
                "source_p_plus_1": source_page + 1,
                "source_p_plus_2": source_page + 2,
                "output_pages": f"{output_start}-{output_start + 2}",
            }
        )

    with OUTPUT_PDF.open("wb") as file:
        writer.write(file)

    with MANIFEST_CSV.open("w", newline="", encoding="utf-8") as file:
        csv_writer = csv.DictWriter(file, fieldnames=manifest_rows[0].keys())
        csv_writer.writeheader()
        csv_writer.writerows(manifest_rows)

    verified = PdfReader(OUTPUT_PDF)
    expected_pages = len(QUERIES) * 3
    if len(verified.pages) != expected_pages:
        raise RuntimeError(
            f"Output verification failed: expected {expected_pages} pages, "
            f"found {len(verified.pages)}"
        )

    print(f"Verified {len(QUERIES)} queries and {len(set(figure_from_query(q) for q in QUERIES))} figures.")
    print(f"Wrote {len(verified.pages)} pages to {OUTPUT_PDF}")
    print(f"Wrote manifest to {MANIFEST_CSV}")


if __name__ == "__main__":
    main()
