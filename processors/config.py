"""Configuration values for the document and image-processing stages."""

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class PDFConfig:
    """Configuration for rendering publication PDFs."""

    output_dir: Path
    render_dpi: int = 400


@dataclass
class ModelConfig:
    """Configuration for vessel-detection model execution."""

    models_dir: Path
    pred_output_dir: Path
    confidence: float = 0.5
    kernel_size: int = 2
    iterations: int = 10
    diagnostic: bool = False
    device: str = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


@dataclass
class MaskExtractionConfig:
    """Configuration for turning detected masks into vessel cards."""

    pdfimg_output_dir: Path
    pred_output_dir: Path
    min_area_ratio: float = 0.0002
    closing_kernel_size: int = 3
    output_suffix: str = "_card"
    mask_suffix: str = "_mask"


@dataclass
class AnnotationConfig:
    """Configuration for browser mask review and editing."""

    pred_output_dir: Path


@dataclass
class TabularConfig:
    """Configuration for editable card metadata."""

    pdfimg_output_dir: Path
    pred_output_dir: Path
    max_workers: int = 4
    cache_size: int = 32
