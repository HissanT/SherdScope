"""Build one deduplicated Hesban PDF for eight requested figures.

Each figure contributes its heading page (p) and the following two pages
(p+1 and p+2). Shared source pages are written only once.
"""

from __future__ import annotations

import csv
from pathlib import Path

from pypdf import PdfReader, PdfWriter

from build_hesban_classical_split import SOURCE_PDF, build_figure_index


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR.parent / "output" / "pdf"
OUTPUT_PDF = OUTPUT_DIR / (
    "Hesban_Eight_New_Figures_4-5_3-53_3-65_3-76_"
    "3-25_3-56_3-88_3-84_p-p1-p2.pdf"
)
MANIFEST_CSV = OUTPUT_DIR / "Hesban_Eight_New_Figures_manifest.csv"

FIGURES = ["4.5", "3.53", "3.65", "3.76", "3.25", "3.56", "3.88", "3.84"]


def main() -> None:
    if len(FIGURES) != 8 or len(set(FIGURES)) != 8:
        raise RuntimeError("Expected exactly eight unique requested figures.")
    if not SOURCE_PDF.exists():
        raise FileNotFoundError(SOURCE_PDF)

    figure_index = build_figure_index(SOURCE_PDF)
    missing = [figure for figure in FIGURES if figure not in figure_index]
    if missing:
        raise RuntimeError(f"Missing figure headings: {', '.join(missing)}")

    reader = PdfReader(SOURCE_PDF)
    writer = PdfWriter()
    source_to_output: dict[int, int] = {}
    manifest_rows: list[dict[str, object]] = []

    for request_order, figure in enumerate(FIGURES, start=1):
        source_p = figure_index[figure]
        source_pages = [source_p, source_p + 1, source_p + 2]
        if source_pages[-1] > len(reader.pages):
            raise RuntimeError(f"Page range exceeds source PDF for Figure {figure}.")

        for source_page in source_pages:
            if source_page not in source_to_output:
                writer.add_page(reader.pages[source_page - 1])
                source_to_output[source_page] = len(writer.pages)

        output_pages = [source_to_output[page] for page in source_pages]
        writer.add_outline_item(
            title=f"Figure {figure}: source PDF pages {source_p}-{source_p + 2}",
            page_number=output_pages[0] - 1,
        )
        manifest_rows.append({
            "request_order": request_order,
            "figure": figure,
            "source_p": source_p,
            "source_p_plus_1": source_p + 1,
            "source_p_plus_2": source_p + 2,
            "output_pages": "; ".join(map(str, output_pages)),
        })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PDF.open("wb") as file:
        writer.write(file)
    with MANIFEST_CSV.open("w", newline="", encoding="utf-8") as file:
        csv_writer = csv.DictWriter(file, fieldnames=manifest_rows[0].keys())
        csv_writer.writeheader()
        csv_writer.writerows(manifest_rows)

    verified = PdfReader(OUTPUT_PDF)
    if len(verified.pages) != len(source_to_output):
        raise RuntimeError(
            f"Expected {len(source_to_output)} unique pages, found {len(verified.pages)}."
        )
    output_index = build_figure_index(OUTPUT_PDF)
    missing_from_output = [figure for figure in FIGURES if figure not in output_index]
    if missing_from_output:
        raise RuntimeError(
            "Output verification could not find: " + ", ".join(missing_from_output)
        )

    print(f"Verified all eight figures: {', '.join(FIGURES)}")
    print(f"Wrote {len(verified.pages)} unique pages to {OUTPUT_PDF}")
    print(f"Wrote manifest to {MANIFEST_CSV}")


if __name__ == "__main__":
    main()
