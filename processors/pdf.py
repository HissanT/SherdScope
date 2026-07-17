"""Publication PDF rendering.

This module contains only the PDF-to-image responsibility. Keeping it separate
from detection, mask extraction, OCR, and export makes the input stage easier
to understand and test without changing its established file naming scheme.
"""

import os
from pathlib import Path

import fitz
from PIL import Image

from .config import PDFConfig


class PDFProcessor:
    """Convert PDF pages to the JPEG images used by later stages."""

    def __init__(self, config: PDFConfig):
        self.config = config

    def process_pdf(
        self, pdf_path: str, split_pages: bool = False, render_dpi: int = None
    ) -> str:
        """Convert a PDF into the configured output directory."""
        try:
            pdf_file_name = Path(pdf_path).stem
            output_folder = self.config.output_dir / pdf_file_name
            os.makedirs(output_folder, exist_ok=True)

            dpi = int(render_dpi or self.config.render_dpi)
            if not 200 <= dpi <= 600:
                raise ValueError("Render DPI must be between 200 and 600")

            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc[page_num]
                pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))
                image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                if split_pages:
                    self._process_split_page(
                        image, pdf_file_name, page_num, output_folder
                    )
                else:
                    self._process_single_page(
                        image, pdf_file_name, page_num, output_folder
                    )
            doc.close()
            return f"PDF file {pdf_file_name} has been converted to JPG"
        except Exception as exc:
            return f"Error processing PDF: {str(exc)}"

    def process_pdf_to_folder(
        self,
        pdf_path: str,
        output_folder: str,
        split_pages: bool = False,
        project_name: str = None,
        render_dpi: int = None,
    ) -> str:
        """Convert a PDF into a specified project image folder."""
        try:
            base_name = project_name if project_name else Path(pdf_path).stem
            output_path = Path(output_folder)
            os.makedirs(output_path, exist_ok=True)

            dpi = int(render_dpi or self.config.render_dpi)
            if not 200 <= dpi <= 600:
                raise ValueError("Render DPI must be between 200 and 600")

            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc[page_num]
                pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))
                image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                if split_pages:
                    self._process_split_page(image, base_name, page_num, output_path)
                else:
                    self._process_single_page(image, base_name, page_num, output_path)
            doc.close()
            return f"PDF file {base_name} has been converted to JPG"
        except Exception as exc:
            return f"Error processing PDF: {str(exc)}"

    @staticmethod
    def _process_single_page(
        image: Image.Image, pdf_name: str, page_num: int, output_folder: Path
    ) -> None:
        output_name = f"{pdf_name}_page_{page_num}.jpg"
        image.save(output_folder / output_name, "JPEG", quality=90, optimize=True)

    @staticmethod
    def _process_split_page(
        image: Image.Image, pdf_name: str, page_num: int, output_folder: Path
    ) -> None:
        width, height = image.size
        midpoint = width // 2
        left_page = image.crop((0, 0, midpoint, height))
        right_page = image.crop((midpoint, 0, width, height))
        left_page.save(
            output_folder / f"{pdf_name}_page_{page_num}a.jpg",
            "JPEG",
            quality=90,
            optimize=True,
        )
        right_page.save(
            output_folder / f"{pdf_name}_page_{page_num}b.jpg",
            "JPEG",
            quality=90,
            optimize=True,
        )
