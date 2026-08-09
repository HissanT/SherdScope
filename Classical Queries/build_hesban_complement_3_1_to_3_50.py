"""Build the non-repeating complement of Hesban figures 3.1 through 3.50."""

from __future__ import annotations

import csv
from pathlib import Path

from pypdf import PdfReader, PdfWriter

from build_hesban_classical_split import SOURCE_PDF, build_figure_index


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PDF = SCRIPT_DIR / "Hesban_Complement_3.1-3.50_p-p1-p2_deduplicated.pdf"
MANIFEST_CSV = SCRIPT_DIR / "Hesban_Complement_3.1-3.50_manifest.csv"

EXCLUDED_QUERIES = [
    "3.1.8", "3.2.4", "3.3.6", "3.4.9", "3.5.8", "3.6.10", "3.7.7",
    "3.8.5", "3.9.8", "3.11.1", "3.12.5", "3.13.6", "3.14.10",
    "3.15.4", "3.16.1", "3.17.9", "3.18.5", "3.19.5", "3.20.20",
    "3.21.16", "3.22.30", "3.23.2", "3.24.18", "3.25.4", "3.26.12",
    "3.27.19", "3.29.4", "3.43.4", "3.47.2", "3.81.5",
]


def figure_from_query(query: str) -> str:
    major, figure, _item = query.split(".")
    return f"{major}.{figure}"


def main() -> None:
    all_figures = [f"3.{number}" for number in range(1, 51)]
    excluded_figures = {figure_from_query(query) for query in EXCLUDED_QUERIES}
    included_figures = [
        figure for figure in all_figures if figure not in excluded_figures
    ]

    if len(excluded_figures & set(all_figures)) != 29:
        raise RuntimeError("Expected 29 excluded figures within the 3.1-3.50 range.")
    if len(included_figures) != 21 or set(included_figures) & excluded_figures:
        raise RuntimeError("Complement validation failed.")

    figure_index = build_figure_index(SOURCE_PDF)
    missing = [figure for figure in included_figures if figure not in figure_index]
    if missing:
        raise RuntimeError(f"Missing figure headings: {', '.join(missing)}")

    reader = PdfReader(SOURCE_PDF)
    writer = PdfWriter()
    source_to_output: dict[int, int] = {}
    manifest_rows: list[dict[str, object]] = []

    for figure in included_figures:
        source_p = figure_index[figure]
        source_pages = [source_p, source_p + 1, source_p + 2]
        for source_page in source_pages:
            if source_page not in source_to_output:
                writer.add_page(reader.pages[source_page - 1])
                source_to_output[source_page] = len(writer.pages)

        output_pages = [source_to_output[page] for page in source_pages]
        writer.add_outline_item(
            title=f"Figure {figure}: source pages {source_p}-{source_p + 2}",
            page_number=output_pages[0] - 1,
        )
        manifest_rows.append(
            {
                "figure": figure,
                "source_p": source_p,
                "source_p_plus_1": source_p + 1,
                "source_p_plus_2": source_p + 2,
                "output_pages": "; ".join(map(str, output_pages)),
            }
        )

    with OUTPUT_PDF.open("wb") as file:
        writer.write(file)
    with MANIFEST_CSV.open("w", newline="", encoding="utf-8") as file:
        csv_writer = csv.DictWriter(file, fieldnames=manifest_rows[0].keys())
        csv_writer.writeheader()
        csv_writer.writerows(manifest_rows)

    verified = PdfReader(OUTPUT_PDF)
    expected_pages = len(source_to_output)
    if len(verified.pages) != expected_pages:
        raise RuntimeError(
            f"Expected {expected_pages} unique pages, found {len(verified.pages)}."
        )
    if len(source_to_output) != len(set(source_to_output)):
        raise RuntimeError("Duplicate source pages were written.")

    print(f"Included figures ({len(included_figures)}): {', '.join(included_figures)}")
    print(f"Excluded figures in range (29): {', '.join(sorted(excluded_figures & set(all_figures), key=lambda value: int(value.split('.')[1])))}")
    print(f"Wrote {len(verified.pages)} unique pages to {OUTPUT_PDF}")
    print(f"Wrote manifest to {MANIFEST_CSV}")


if __name__ == "__main__":
    main()
