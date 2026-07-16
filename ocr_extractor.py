"""Local OCR extraction for Hesban 11 figure/table linking.

PaddleOCR only reads text and its coordinates. The deterministic code in this
module uses each printed Hesban 11 header to build page-specific table columns;
it does not use a generative model to infer or rewrite metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
from statistics import median
from typing import Any, Iterable, Optional

import numpy as np
import cv2
from PIL import Image, ImageFilter, ImageOps

from metadata_linker import (
    HESBAN_TABLE_COLUMNS,
    Hesban11Profile,
    StructuredExtractor,
    normalize_figure_id,
    normalize_vessel_number,
)


class OCRUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class OCRToken:
    text: str
    score: float
    bbox: tuple[float, float, float, float]

    @property
    def center_x(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2

    @property
    def center_y(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2

    @property
    def height(self) -> float:
        return max(1.0, self.bbox[3] - self.bbox[1])


@dataclass(frozen=True)
class TableBoundary:
    """A visually verified Hesban table region, relative to the supplied image."""

    left: int
    right: int
    upper_header_rule: int
    lower_header_rule: int
    data_top: int
    data_bottom: int
    closing_rule: Optional[int]
    header_text: str
    column_bounds: tuple[tuple[int, int], ...]
    column_source: str
    header_anchors: tuple[dict[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "table_bounds": [self.left, self.upper_header_rule, self.right, self.data_bottom],
            "upper_header_rule": self.upper_header_rule,
            "lower_header_rule": self.lower_header_rule,
            "data_start_y": self.data_top,
            "data_end_y": self.data_bottom,
            "closing_rule_y": self.closing_rule,
            "header_confirmed": True,
            "has_closing_rule": self.closing_rule is not None,
            "continues": self.closing_rule is None,
            "header_text": self.header_text,
            "column_bounds": [[self.left + left, self.left + right]
                              for left, right in self.column_bounds],
            "column_source": self.column_source,
            "header_anchors": [
                {**anchor, "bbox": list(anchor.get("bbox", []))}
                for anchor in self.header_anchors
            ],
        }


def _prepare_image(image: Image.Image, min_height: int = 0) -> Image.Image:
    gray = ImageOps.autocontrast(image.convert("L"), cutoff=1).filter(ImageFilter.SHARPEN)
    if min_height and gray.height < min_height:
        scale = min(4.0, min_height / max(1, gray.height))
        gray = gray.resize((max(1, round(gray.width * scale)), round(gray.height * scale)),
                           Image.Resampling.LANCZOS)
    return gray.convert("RGB")


def _prepare_compact_cell(image: Image.Image) -> Image.Image:
    """Enlarge an isolated categorical glyph without teaching OCR its value.

    Hesban columns such as Non-Plastics ``Typ`` often contain a single narrow
    letter. Whole-row text detection can legitimately skip that tiny component
    even when the column boundary is correct. Crop to the actual ink, surround
    it with clean whitespace, then enlarge it so PaddleOCR still reads the
    printed character rather than a hard-coded publication-specific default.
    """
    gray = ImageOps.autocontrast(image.convert("L"), cutoff=1)
    pixels = np.asarray(gray)
    ys, xs = np.where(pixels < 205)
    if len(xs) and len(ys):
        pad = max(3, round(min(gray.width, gray.height) * .04))
        left, right = max(0, int(xs.min()) - pad), min(gray.width, int(xs.max()) + pad + 1)
        top, bottom = max(0, int(ys.min()) - pad), min(gray.height, int(ys.max()) + pad + 1)
        gray = gray.crop((left, top, right, bottom))
    white_pad = max(12, round(max(gray.width, gray.height) * .25))
    canvas = Image.new("L", (gray.width + white_pad * 2, gray.height + white_pad * 2), 255)
    canvas.paste(gray, (white_pad, white_pad))
    prepared = _prepare_image(canvas, min_height=128)
    if prepared.width < 96:
        padded = Image.new("RGB", (96, prepared.height), "white")
        padded.paste(prepared, ((96 - prepared.width) // 2, 0))
        prepared = padded
    return prepared


def _compact_code(value: str) -> str:
    """Return a short printed category code, or blank for noisy OCR output."""
    compact = re.sub(r"\s+", "", value or "")
    return compact if re.fullmatch(r"[A-Za-z][A-Za-z0-9./-]{0,7}", compact) else ""


def _remove_cross_column_code(value: str, code: str) -> str:
    """Remove a verified next-column code from an OCR token crossing its edge."""
    lines = (value or "").splitlines()
    if not lines or not code:
        return value or ""
    lines[0] = re.sub(rf"{re.escape(code)}\s*$", "", lines[0],
                      count=1, flags=re.IGNORECASE).rstrip()
    return "\n".join(lines).strip()


def _poly_bbox(poly: Any) -> tuple[float, float, float, float]:
    points = np.asarray(poly, dtype=float).reshape(-1, 2)
    return (float(points[:, 0].min()), float(points[:, 1].min()),
            float(points[:, 0].max()), float(points[:, 1].max()))


def _parse_v3_result(result: Any) -> list[OCRToken]:
    payload = getattr(result, "json", result)
    if callable(payload):
        payload = payload()
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except (TypeError, ValueError, json.JSONDecodeError):
            return []
    if not isinstance(payload, dict):
        return []
    data = payload.get("res", payload)
    if not isinstance(data, dict):
        return []
    texts = data.get("rec_texts", [])
    scores = data.get("rec_scores", [])
    polys = data.get("rec_polys", data.get("dt_polys", []))
    tokens = []
    for index, text in enumerate(texts):
        if index >= len(polys):
            continue
        score = scores[index] if index < len(scores) else 0.0
        tokens.append(OCRToken(str(text).strip(), float(score), _poly_bbox(polys[index])))
    return [token for token in tokens if token.text]


def _parse_legacy_result(result: Any) -> list[OCRToken]:
    tokens: list[OCRToken] = []
    items = result or []
    if len(items) == 1 and isinstance(items[0], list):
        items = items[0]
    for item in items:
        try:
            poly, recognition = item
            text, score = recognition
            if str(text).strip():
                tokens.append(OCRToken(str(text).strip(), float(score), _poly_bbox(poly)))
        except (TypeError, ValueError, IndexError):
            continue
    return tokens


class PaddleOCREngine:
    """Small lazy adapter supporting PaddleOCR 3.x and its older result shape."""

    def __init__(self):
        self._model = None
        self._uses_predict = False

    def _load(self):
        if self._model is not None:
            return
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise OCRUnavailableError(
                "PaddleOCR is not installed. Install requirements-ocr.txt, then restart SherdScope."
            ) from exc
        try:
            self._model = PaddleOCR(
                # PaddleOCR 3.7 defaults to substantially heavier "medium"
                # models.  The mobile pair is a much better CPU-only default
                # for this fixed, high-resolution publication layout.
                text_detection_model_name="PP-OCRv5_mobile_det",
                text_recognition_model_name="PP-OCRv5_mobile_rec",
                text_det_limit_side_len=3000,
                text_det_limit_type="max",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False, use_textline_orientation=False)
            self._uses_predict = hasattr(self._model, "predict")
        except TypeError:
            # Compatibility with PaddleOCR 2.x installations.
            self._model = PaddleOCR(lang="en", use_angle_cls=False, show_log=False)
            self._uses_predict = False

    def recognize(self, image: Image.Image) -> list[OCRToken]:
        self._load()
        prepared = np.asarray(_prepare_image(image))
        if self._uses_predict:
            results = list(self._model.predict(prepared))
            return [token for result in results for token in _parse_v3_result(result)]
        return _parse_legacy_result(self._model.ocr(prepared, cls=False))

    def recognize_many(self, images: Iterable[Image.Image]) -> list[list[OCRToken]]:
        images = list(images)
        self._load()
        if not images:
            return []
        if not self._uses_predict:
            return [self.recognize(image) for image in images]
        parsed: list[list[OCRToken]] = []
        # PaddleOCR 3.x accepts image lists.  Batching avoids rebuilding the
        # detection/recognition pipeline for every small crop while bounding RAM.
        for start in range(0, len(images), 24):
            batch = [np.asarray(_prepare_image(image)) for image in images[start:start+24]]
            results = list(self._model.predict(batch))
            parsed.extend(_parse_v3_result(result) for result in results)
        return parsed


def _tokens_to_text(tokens: list[OCRToken], minimum_score: float = 0.25) -> tuple[str, float]:
    usable = [token for token in tokens if token.text and token.score >= minimum_score]
    if not usable:
        return "", 0.0
    heights = [token.height for token in usable]
    tolerance = max(5.0, median(heights) * 0.65)
    lines: list[list[OCRToken]] = []
    for token in sorted(usable, key=lambda item: (item.center_y, item.bbox[0])):
        if not lines or abs(token.center_y - sum(t.center_y for t in lines[-1]) / len(lines[-1])) > tolerance:
            lines.append([token])
        else:
            lines[-1].append(token)
    text = "\n".join(" ".join(token.text for token in sorted(line, key=lambda item: item.bbox[0]))
                     for line in lines)
    return text.strip(), min(token.score for token in usable)


def _leading_row_number(text: str) -> Optional[re.Match]:
    """Match a row number even when OCR glues it to a capitalized Type."""
    normal = re.match(r"\s*(\d{1,3}[a-z]?)\b", text or "", re.IGNORECASE)
    if normal:
        return normal
    return re.match(r"\s*(\d{1,3})(?=[A-Z])", text or "")


class PaddleOCRStructuredExtractor(StructuredExtractor):
    """Hesban-specific coordinate parser backed by compact local OCR models."""

    # Column centers measured from the repeated Hesban 11 table template.  The
    # original measurements are page-relative; _column_bounds converts them to
    # coordinates relative to the detected long-rule width.
    PAGE_COLUMN_CENTERS = (
        .148, .176, .214, .254, .273, .300, .347, .399, .453, .493, .515,
        .540, .569, .596, .623, .648, .670, .703, .745, .775, .806, .843,
    )
    PAGE_TABLE_LEFT = .135
    PAGE_TABLE_RIGHT = .860
    # When two neighboring headings leave whitespace between them, give most
    # of that gap to the column on the left. Printed values such as
    # ``Cooking pot`` are often much wider than the short ``Type`` heading,
    # while the next column begins close to its own heading.
    HEADER_GAP_SHARE_TO_PREVIOUS = .82
    DIRECT_HEADER_ALIASES = {
        0: ("no", "number"), 1: ("type",), 2: ("sq", "square"),
        3: ("loc", "locus"), 4: ("pail",), 5: ("reg", "registration"),
        15: ("man", "manufacture"), 20: ("decor", "decoration"),
        21: ("fire",),
    }
    SUBHEADER_SEQUENCE = (
        (6, ("exterior",)), (7, ("core",)), (8, ("interior",)),
        (9, ("typ", "type")), (10, ("siz", "size")),
        (11, ("shap", "shape")), (12, ("den", "density")),
        (13, ("tysz", "typesize")), (14, ("den", "density")),
        (16, ("ext", "exterior")), (17, ("color", "colour")),
        (18, ("int", "interior")), (19, ("color", "colour")),
    )

    def __init__(self, engine: Optional[PaddleOCREngine] = None):
        self.engine = engine or PaddleOCREngine()

    def extract_drawing_identifiers(self, image_path: Path, cards: list[dict[str, Any]],
                                    page_context: dict[str, Any]) -> dict[str, Any]:
        with Image.open(image_path) as source_image:
            source = source_image.convert("RGB")
        width, height = source.size
        drawings = {}
        crop_jobs = []
        for card in cards:
            x1, y1, x2, y2 = [int(value) for value in card["bbox"]]
            box_width, box_height = x2 - x1, y2 - y1
            margin_x = max(45, round(box_width * .30))
            margin_top = max(25, round(box_height * .10))
            margin_bottom = max(110, round(box_height * .45))
            left, top = max(0, x1-margin_x), max(0, y1-margin_top)
            right, bottom = min(width, x2+margin_x), min(height, y2+margin_bottom)
            crop = source.crop((left, top, right, bottom))
            prepared_crop = _prepare_image(crop, min_height=360)
            scale_x = prepared_crop.width / max(1, crop.width)
            scale_y = prepared_crop.height / max(1, crop.height)
            crop_jobs.append((card, left, top, crop, prepared_crop, scale_x, scale_y))

        # Read all drawing-number neighborhoods in batches. Apart from being
        # faster than one model call per card, this keeps the scoring rules
        # identical for every vessel on a crowded page.
        recognized = self.engine.recognize_many(job[4] for job in crop_jobs)
        for (card, left, top, crop, prepared_crop, scale_x, scale_y), tokens in zip(crop_jobs, recognized):
            x1, y1, x2, y2 = [int(value) for value in card["bbox"]]
            candidates = []
            for token in tokens:
                number = normalize_vessel_number(token.text)
                if not re.fullmatch(r"\d{1,3}[a-z]?", number):
                    continue
                # Printed identifiers normally sit directly beneath the
                # drawing. Penalize horizontally adjacent vessel numbers; the
                # old vertical-only score could swap neighbors in dense rows.
                relative_left = (x1 - left) * scale_x
                relative_right = (x2 - left) * scale_x
                relative_center = (relative_left + relative_right) / 2
                relative_drawing_bottom = (y2 - top) * scale_y
                if token.center_y < (y1 - top) * scale_y + (y2-y1) * scale_y * .62:
                    continue
                horizontal_distance = (0 if relative_left <= token.center_x <= relative_right
                                       else min(abs(token.center_x-relative_left),
                                                abs(token.center_x-relative_right)))
                vertical_distance = abs(token.center_y - relative_drawing_bottom)
                score = (token.score
                         - vertical_distance / max(220, prepared_crop.height * .75)
                         - horizontal_distance / max(180, prepared_crop.width * .55)
                         - abs(token.center_x-relative_center) / max(700, prepared_crop.width * 2.0))
                candidates.append((score, number))
            best = max(candidates, default=None)
            drawings[card["mask_file"]] = {"number": best[1] if best and best[0] >= 0.15 else None}

        profile = Hesban11Profile()
        figure_id = normalize_figure_id(page_context.get("figure_id"))
        caption = page_context.get("figure_caption") or page_context.get("caption", "")
        printed_page = page_context.get("printed_page", "")
        if not figure_id or not printed_page:
            header = source.crop((0, 0, width, min(height, round(height * .22))))
            text, _ = _tokens_to_text(self.engine.recognize(header))
            visual = profile.detect_figure_context(text)
            figure_id = figure_id or visual.get("figure_id", "")
            caption = caption or visual.get("caption", "")
            printed_page = printed_page or profile.detect_printed_page(text)
        return {
            "figure_id": figure_id,
            "figure_caption": caption,
            "printed_page": printed_page,
            "drawings": drawings,
        }

    @classmethod
    def _column_bounds(cls, width: int) -> list[tuple[int, int]]:
        """Legacy proportional bounds used only when header anchoring fails."""
        span = cls.PAGE_TABLE_RIGHT - cls.PAGE_TABLE_LEFT
        centers = tuple((center - cls.PAGE_TABLE_LEFT) / span
                        for center in cls.PAGE_COLUMN_CENTERS)
        edges = [0.0] + [(centers[index] + centers[index+1]) / 2
                         for index in range(len(centers)-1)] + [1.0]
        # Hesban's Man column is extremely narrow. The generic midpoint gave
        # it too much space and let a merged ``W **`` token steal the Ext
        # marks. Move only the Man/Ext divider slightly toward Man.
        edges[16] = (.655 - cls.PAGE_TABLE_LEFT) / span
        return [(max(0, round(edges[index] * width)), min(width, round(edges[index+1] * width)))
                for index in range(len(centers))]

    @staticmethod
    def _normalized_header_word(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", (value or "").lower())

    @classmethod
    def _header_match_score(cls, text: str, aliases: tuple[str, ...]) -> float:
        normalized = cls._normalized_header_word(text)
        if not normalized:
            return 0.0
        scores = []
        for alias in aliases:
            alias = cls._normalized_header_word(alias)
            if normalized == alias:
                scores.append(1.0)
            elif normalized.startswith(alias) or alias.startswith(normalized):
                scores.append(.84 if min(len(normalized), len(alias)) >= 3 else .70)
            else:
                scores.append(SequenceMatcher(None, normalized, alias).ratio())
        return max(scores, default=0.0)

    @classmethod
    def _split_header_tokens(cls, tokens: list[OCRToken]) -> list[OCRToken]:
        """Split OCR boxes such as ``No Type`` into usable word anchors."""
        known = {
            alias for aliases in cls.DIRECT_HEADER_ALIASES.values() for alias in aliases
        } | {
            alias for _, aliases in cls.SUBHEADER_SEQUENCE for alias in aliases
        } | {"fabric", "non", "plastics", "void", "voids", "surface", "treatment"}
        output: list[OCRToken] = []
        for token in tokens:
            parts = re.findall(r"[A-Za-z]+(?:/[A-Za-z]+)?", token.text)
            normalized_parts = [cls._normalized_header_word(part) for part in parts]
            if (2 <= len(parts) <= 4 and
                    sum(part in known for part in normalized_parts) >= 2):
                total = sum(max(1, len(part)) for part in parts)
                cursor = token.bbox[0]
                span = token.bbox[2] - token.bbox[0]
                for part in parts:
                    part_width = span * max(1, len(part)) / total
                    output.append(OCRToken(
                        part, token.score,
                        (cursor, token.bbox[1], cursor + part_width, token.bbox[3])))
                    cursor += part_width
            else:
                output.append(token)
        return output

    @classmethod
    def _column_bounds_from_header(
            cls, tokens: list[OCRToken], width: int,
            upper_rule_y: float) -> tuple[list[tuple[int, int]], str, list[dict[str, Any]]]:
        """Build this page's 22 columns from its actual two-level header."""
        tokens = cls._split_header_tokens([token for token in tokens if token.text])
        tolerance = (max(5.0, median([token.height for token in tokens]) * .45)
                     if tokens else 5.0)
        upper_tokens = [token for token in tokens
                        if token.center_y <= upper_rule_y + tolerance]
        lower_tokens = sorted(
            [token for token in tokens if token.center_y > upper_rule_y - tolerance],
            key=lambda token: token.center_x)

        anchors: dict[int, OCRToken] = {}
        for column_index, aliases in cls.DIRECT_HEADER_ALIASES.items():
            candidates = [(cls._header_match_score(token.text, aliases), token)
                          for token in upper_tokens]
            score, token = max(
                candidates, default=(0.0, None),
                key=lambda item: item[0] * (item[1].score if item[1] else 0))
            if token is not None and score >= .68:
                anchors[column_index] = token

        # Repeated Den and Color headings are disambiguated by printed order.
        cursor = -1.0
        for column_index, aliases in cls.SUBHEADER_SEQUENCE:
            candidates = []
            for token in lower_tokens:
                if token.center_x <= cursor:
                    continue
                score = cls._header_match_score(token.text, aliases)
                if score >= .68:
                    candidates.append((token.center_x, -score * token.score, token))
            if candidates:
                _, _, token = min(candidates)
                anchors[column_index] = token
                cursor = token.center_x

        direct_count = sum(index in anchors for index in cls.DIRECT_HEADER_ALIASES)
        subgroup_count = sum(index in anchors for index, _ in cls.SUBHEADER_SEQUENCE)
        if direct_count < 5 or subgroup_count < 7 or len(anchors) < 13:
            return cls._column_bounds(width), "fixed_fallback", []

        fallback = cls._column_bounds(width)
        fallback_centers = [(left + right) / 2 for left, right in fallback]
        observed = {index: token.center_x for index, token in anchors.items()}
        centers: list[float] = []
        for index, fallback_center in enumerate(fallback_centers):
            if index in observed:
                centers.append(observed[index])
                continue
            previous = max((item for item in observed if item < index), default=None)
            following = min((item for item in observed if item > index), default=None)
            if previous is not None and following is not None:
                ratio = ((fallback_center - fallback_centers[previous]) /
                         max(1.0, fallback_centers[following] - fallback_centers[previous]))
                center = observed[previous] + ratio * (observed[following] - observed[previous])
            elif previous is not None:
                center = observed[previous] + fallback_center - fallback_centers[previous]
            elif following is not None:
                center = observed[following] + fallback_center - fallback_centers[following]
            else:
                center = fallback_center
            centers.append(center)

        minimum_gap = max(3.0, width * .0025)
        for index in range(1, len(centers)):
            centers[index] = max(centers[index], centers[index-1] + minimum_gap)
        if centers[-1] >= width:
            return cls._column_bounds(width), "fixed_fallback", []

        edges = [0.0]
        for index in range(len(centers)-1):
            current = anchors.get(index)
            following = anchors.get(index+1)
            if current and following and current.bbox[2] <= following.bbox[0]:
                gap = following.bbox[0] - current.bbox[2]
                edge = current.bbox[2] + gap * cls.HEADER_GAP_SHARE_TO_PREVIOUS
            else:
                edge = (centers[index] + centers[index+1]) / 2
            edges.append(max(edges[-1] + 1, min(width-1, edge)))
        edges.append(float(width))
        bounds = [(round(edges[index]), round(edges[index+1]))
                  for index in range(len(centers))]
        if len(bounds) != len(HESBAN_TABLE_COLUMNS) or any(left >= right for left, right in bounds):
            return cls._column_bounds(width), "fixed_fallback", []

        evidence = [{
            "column": HESBAN_TABLE_COLUMNS[index],
            "text": token.text,
            "bbox": [round(value) for value in token.bbox],
        } for index, token in sorted(anchors.items())]
        return bounds, "header_ocr", evidence

    @staticmethod
    def _horizontal_rules(image: Image.Image) -> list[tuple[int, int, int, int]]:
        """Return long horizontal rules as (y, left, right, thickness).

        Detection is performed on a bounded-size copy so 400-DPI pages remain
        cheap. A horizontal close reconnects scan gaps and a long opening
        removes text, vessel strokes, and short underlines.
        """
        gray = np.asarray(ImageOps.autocontrast(image.convert("L"), cutoff=1))
        source_height, source_width = gray.shape
        scale = min(1.0, 2200.0 / max(1, source_width))
        if scale < 1.0:
            gray = cv2.resize(gray, (round(source_width * scale), round(source_height * scale)),
                              interpolation=cv2.INTER_AREA)
        height, width = gray.shape
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
            max(15, (round(min(width, height) * .012) // 2) * 2 + 1), 13)
        # Correct only small scan skew. Larger angles are likely drawing strokes
        # rather than page rotation and are intentionally ignored.
        hough = cv2.HoughLinesP(binary, 1, np.pi / 1800,
                                threshold=max(40, round(width * .12)),
                                minLineLength=max(80, round(width * .35)),
                                maxLineGap=max(10, round(width * .025)))
        angles = []
        for line in hough if hough is not None else []:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2-y1, x2-x1))
            if abs(angle) <= 3.0:
                angles.append(float(angle))
        skew = float(np.median(angles)) if angles else 0.0
        if .12 <= abs(skew) <= 3.0:
            matrix = cv2.getRotationMatrix2D((width / 2, height / 2), skew, 1.0)
            binary = cv2.warpAffine(binary, matrix, (width, height),
                                    flags=cv2.INTER_NEAREST,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        close_width = max(9, round(width * .018))
        closed = cv2.morphologyEx(
            binary, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (close_width, 1)))
        opened = cv2.morphologyEx(
            closed, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, round(width * .36)), 1)))
        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        raw = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w >= width * .55 and h <= max(18, height * .012):
                raw.append((y + h // 2, x, x + w, max(1, h)))
        if not raw:
            return []
        longest = max(right-left for _, left, right, _ in raw)
        raw = [rule for rule in raw if rule[2]-rule[1] >= longest * .70]
        merged: list[list[int]] = []
        y_tolerance = max(3, round(height * .0018))
        for y, left, right, thickness in sorted(raw):
            if merged and abs(y-merged[-1][0]) <= y_tolerance:
                old = merged[-1]
                old[0] = round((old[0] + y) / 2)
                old[1] = min(old[1], left)
                old[2] = max(old[2], right)
                old[3] = max(old[3], thickness)
            else:
                merged.append([y, left, right, thickness])
        inverse = 1.0 / scale
        return [(round(y*inverse), round(left*inverse), round(right*inverse),
                 max(1, round(thickness*inverse))) for y, left, right, thickness in merged]

    @staticmethod
    def _header_is_hesban(text: str) -> bool:
        normalized = re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()
        if not re.search(r"\bno\b", normalized) or not re.search(r"\btype\b", normalized):
            return False
        main_patterns = (
            r"\bsq\b", r"\bloc\b", r"\bpail\b", r"\breg\b",
            r"fabric\s+color", r"non\s+plastics", r"\bvoids?\b",
            r"\bman\b", r"surface\s+treatment", r"\bdecor\b", r"\bfire\b",
        )
        subgroup_patterns = (
            r"\bexterior\b", r"\bcore\b", r"\binterior\b", r"\btyp\b",
            r"\bsiz\b", r"\bshap\b", r"\bden\b", r"ty\s*sz", r"\bext\b", r"\bint\b",
        )
        main_count = sum(bool(re.search(pattern, normalized)) for pattern in main_patterns)
        subgroup_count = sum(bool(re.search(pattern, normalized)) for pattern in subgroup_patterns)
        return main_count >= 4 and subgroup_count >= 2

    def detect_table_boundary(self, image: Image.Image) -> Optional[TableBoundary]:
        rules = self._horizontal_rules(image)
        if len(rules) < 2:
            return None
        height = image.height
        pairs = []
        for index in range(len(rules)-1):
            upper, lower = rules[index], rules[index+1]
            gap = lower[0] - upper[0]
            if max(8, height * .0015) <= gap <= max(80, height * .045):
                overlap = min(upper[2], lower[2]) - max(upper[1], lower[1])
                if overlap >= .70 * max(upper[2]-upper[1], lower[2]-lower[1]):
                    pairs.append((upper, lower, gap))
        for upper, lower, gap in pairs:
            header_top = max(0, round(upper[0] - max(gap * 2.5, height * .018)))
            header_bottom = min(height, lower[0] + max(3, lower[3]))
            header_crop = image.crop((max(0, min(upper[1], lower[1])), header_top,
                                      min(image.width, max(upper[2], lower[2])), header_bottom))
            header_tokens = self.engine.recognize(header_crop)
            header_text, _ = _tokens_to_text(header_tokens, minimum_score=.18)
            if not self._header_is_hesban(header_text):
                continue
            left = max(0, min(upper[1], lower[1]))
            right = min(image.width, max(upper[2], lower[2]))
            data_top = min(height, lower[0] + max(2, lower[3] // 2 + 1))
            closing = next((rule for rule in rules if rule[0] > data_top + max(8, height*.002)), None)
            closing_y = closing[0] if closing else None
            data_bottom = closing_y if closing_y is not None else height
            column_bounds, column_source, header_anchors = self._column_bounds_from_header(
                header_tokens, right-left, upper[0]-header_top)
            for anchor in header_anchors:
                anchor["bbox"][1] += header_top
                anchor["bbox"][3] += header_top
            return TableBoundary(left, right, upper[0], lower[0], data_top,
                                 data_bottom, closing_y, header_text,
                                 tuple(column_bounds), column_source,
                                 tuple(header_anchors))
        return None

    @staticmethod
    def _last_row_bottom(image: Image.Image, last_anchor: float, default_gap: float) -> int:
        gray = np.asarray(image.convert("L"))
        width = gray.shape[1]
        left, right = round(width * .14), round(width * .89)
        if right <= left:
            return image.height
        dark_fraction = (gray[:, left:right] < 110).mean(axis=1)
        candidates = np.where(dark_fraction > .48)[0]
        candidates = candidates[candidates > last_anchor + 5]
        if len(candidates):
            return int(candidates[0])
        return min(image.height, round(last_anchor + max(45, default_gap * 1.35)))

    def _number_component_tokens(
            self, image: Image.Image,
            bounds: Optional[list[tuple[int, int]]] = None) -> list[OCRToken]:
        """OCR each small item in the printed No column.

        Whole-page detectors occasionally merge ``4`` with ``Pithos`` or miss
        the very narrow ``1``.  A simple vertical ink projection finds those
        number-sized components without guessing their values; OCR still reads
        the actual printed characters.
        """
        width = image.width
        bounds = bounds or self._column_bounds(width)
        left, right = bounds[0]
        gray = np.asarray(image.convert("L"))[:, left:right]
        active = np.where((gray < 170).sum(axis=1) >= 2)[0]
        runs: list[tuple[int, int]] = []
        start = previous = None
        for y in active:
            if start is None or y > previous + 4:
                if start is not None:
                    runs.append((int(start), int(previous)))
                start = y
            previous = y
        if start is not None:
            runs.append((int(start), int(previous)))
        runs = [(top, bottom) for top, bottom in runs if 8 <= bottom-top+1 <= 50]
        # Include Type as context. PaddleOCR often cannot recognize a narrow
        # isolated ``1``, but reliably reads ``1 Pithos``; _row_anchors keeps
        # only the leading printed number.
        context_right = bounds[1][1]
        crops = [image.crop((left, max(0, top-10), context_right, min(image.height, bottom+11)))
                 for top, bottom in runs]
        output = []
        for (top, bottom), tokens in zip(runs, self.engine.recognize_many(crops)):
            text, confidence = _tokens_to_text(tokens)
            match = _leading_row_number(text)
            if match:
                output.append(OCRToken(match.group(1), confidence,
                              (left, top, right, bottom)))
        return output

    def _row_anchors(self, image: Image.Image, expected_numbers: list[str],
                     page_tokens: Optional[list[OCRToken]] = None,
                     restrict_to_expected: bool = False,
                     bounds: Optional[list[tuple[int, int]]] = None) -> list[tuple[str, OCRToken]]:
        width = image.width
        bounds = bounds or self._column_bounds(width)
        number_left, number_right = bounds[0]
        tokens = page_tokens if page_tokens is not None else self.engine.recognize(image)
        component_tokens = self._number_component_tokens(image, bounds)
        expected = {normalize_vessel_number(number) for number in expected_numbers}
        candidates: list[tuple[str, OCRToken]] = []
        for token in tokens + component_tokens:
            leading = _leading_row_number(token.text)
            number = normalize_vessel_number(leading.group(1) if leading else token.text)
            if not re.fullmatch(r"\d{1,3}[a-z]?", number):
                continue
            if int(re.match(r"\d+", number).group(0)) <= 0:
                continue
            if restrict_to_expected and expected and number not in expected:
                continue
            merged_no_type = bool(
                leading and number in expected and token.text[leading.end():].strip()
                and token.bbox[0] < number_right and token.bbox[2] > number_left)
            if not (number_left <= token.center_x < number_right):
                if not merged_no_type:
                    continue
                token = OCRToken(token.text, token.score,
                                 (number_left, token.bbox[1], number_right, token.bbox[3]))
            # A row number must line up with actual row content. This rejects a
            # stray page number even if a damaged crop happens to place it near
            # the first column.
            vertical_tolerance = max(18.0, token.height * 1.35)
            has_row_content = merged_no_type or any(
                item is not token and item.center_x >= bounds[1][0]
                and abs(item.center_y-token.center_y) <= vertical_tolerance
                for item in tokens)
            if not has_row_content:
                continue
            candidates.append((number, token))
        # Collapse duplicate readings of the same printed row at the same height.
        selected: list[tuple[str, OCRToken]] = []
        for number, token in sorted(candidates, key=lambda item: item[1].center_y):
            duplicate = next((index for index, (old_number, old_token) in enumerate(selected)
                              if old_number == number and abs(old_token.center_y-token.center_y) < 18), None)
            if duplicate is None:
                selected.append((number, token))
            elif token.score > selected[duplicate][1].score:
                selected[duplicate] = (number, token)
        return selected

    def extract_table(self, image_path: Path, crop: Optional[tuple[int, int, int, int]],
                      figure_id: str, expected_numbers: list[str],
                      page_context: dict[str, Any]) -> dict[str, Any]:
        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB")
        crop_left = crop_top = 0
        if crop:
            left, top, right, bottom = [int(value) for value in crop]
            if bottom-top < 80:
                return {"is_table": False, "rows": []}
            crop_left, crop_top = left, top
            image = image.crop((left, top, right, bottom))

        boundary = self.detect_table_boundary(image)
        if boundary is None:
            return {
                "is_table": False, "rows": [],
                "warnings": [{"code": "table_header_not_found",
                              "message": f"No verified Hesban table header was found for figure {figure_id}."}],
            }

        data_image = image.crop((boundary.left, boundary.data_top,
                                 boundary.right, boundary.data_bottom))
        if data_image.width < 100 or data_image.height < 20:
            return {"is_table": False, "rows": [], "boundary": boundary.as_dict()}

        # One OCR pass over only the verified data rectangle supplies both row
        # anchors and cell text. Header, page-number, and footer text never
        # enters row parsing.
        # per-cell approach could require hundreds of inference calls per page.
        page_tokens = self.engine.recognize(data_image)
        bounds = list(boundary.column_bounds)
        # The first pass must retain unexpected rows so validation can detect a
        # wrong/extra table. Only the targeted truncation retry may filter to
        # the explicitly missing numbers.
        anchors = self._row_anchors(
            data_image, expected_numbers, page_tokens,
            restrict_to_expected=bool(page_context.get("retry_missing")),
            bounds=bounds)
        if not anchors:
            return {"is_table": False, "rows": [], "boundary": boundary.as_dict()}
        differences = [anchors[index+1][1].bbox[1] - anchors[index][1].bbox[1]
                       for index in range(len(anchors)-1) if anchors[index+1][1].bbox[1] > anchors[index][1].bbox[1]]
        default_gap = median(differences) if differences else data_image.height * .09
        rows = []
        warnings = ([] if boundary.column_source == "header_ocr" else [{
            "code": "column_header_fallback",
            "message": (f"Figure {figure_id} used fallback column widths because "
                        "the printed subheadings were not read reliably."),
        }])
        refinement_images = []
        refinement_targets: list[tuple[dict[str, str], str, Optional[dict[str, Any]]]] = []
        interior_overlap_rows: set[int] = set()
        ocr_diagnostics: list[dict[str, Any]] = []
        for index, (number, token) in enumerate(anchors):
            row_top = max(0, round(token.bbox[1] - 4))
            if index+1 < len(anchors):
                row_bottom = max(row_top+5, round(anchors[index+1][1].bbox[1] - 3))
            else:
                row_bottom = data_image.height
            row = {column: "" for column in HESBAN_TABLE_COLUMNS}
            row["table_no"] = number
            for column_index, (column, (x1, x2)) in enumerate(
                    zip(HESBAN_TABLE_COLUMNS[1:], bounds[1:]), start=1):
                cell_tokens = [item for item in page_tokens
                               if x1 <= item.center_x < x2
                               and row_top <= item.center_y < row_bottom]
                value, confidence = _tokens_to_text(cell_tokens)
                row[column] = value
                if (column_index == 8 and
                        any(item.bbox[2] > bounds[9][0] for item in cell_tokens)):
                    interior_overlap_rows.add(id(row))
                if value and confidence < .45:
                    warnings.append({
                        "code": "low_ocr_confidence",
                        "message": f"Review figure {figure_id}, row {number}, field {column}.",
                    })
            # Whole-page OCR often skips repeated Type words and joins narrow
            # adjacent columns (Pail/Reg and Man/Surface Ext). Re-read those
            # cells independently in one batch.
            for column_index in (1, 4, 5, 9, 15, 16):
                column = HESBAN_TABLE_COLUMNS[column_index]
                x1, x2 = bounds[column_index]
                diagnostic = None
                if column_index == 9:
                    # Typ is a short row-level code printed on the first text
                    # line. The complete row can be hundreds of pixels tall
                    # because Size/Shape cells wrap over many lines. Restrict
                    # the retry around the row-number baseline so lower-line
                    # fragments from neighboring columns cannot become h/1/7.
                    inset = max(2, round((x2 - x1) * .08))
                    band_top = max(row_top, round(token.center_y - token.height * .9))
                    band_bottom = min(
                        row_bottom,
                        max(band_top + 8, round(token.center_y + token.height * 1.1)),
                    )
                    crop_x1, crop_x2 = x1 + inset, x2 - inset
                    cell_image = data_image.crop(
                        (crop_x1, band_top, crop_x2, band_bottom))
                    overlapping = [
                        item for item in page_tokens
                        if item.bbox[2] > x1 and item.bbox[0] < x2
                        and row_top <= item.center_y < row_bottom
                    ]
                    diagnostic = {
                        "row": number,
                        "field": column,
                        "crop": [
                            boundary.left + crop_x1 + crop_left,
                            boundary.data_top + band_top + crop_top,
                            boundary.left + crop_x2 + crop_left,
                            boundary.data_top + band_bottom + crop_top,
                        ],
                        "page_overlap_tokens": [{
                            "text": item.text, "confidence": round(item.score, 4),
                            "bbox": [
                                round(boundary.left + item.bbox[0] + crop_left),
                                round(boundary.data_top + item.bbox[1] + crop_top),
                                round(boundary.left + item.bbox[2] + crop_left),
                                round(boundary.data_top + item.bbox[3] + crop_top),
                            ],
                        } for item in overlapping],
                    }
                    prepared = _prepare_compact_cell(cell_image)
                else:
                    cell_image = data_image.crop(
                        (x1, row_top, x2, max(row_top + 1, row_bottom)))
                    prepared = _prepare_image(cell_image, min_height=100)
                refinement_images.append(prepared)
                refinement_targets.append((row, column, diagnostic))
            rows.append(row)
        for (row, column, diagnostic), tokens in zip(
                refinement_targets, self.engine.recognize_many(refinement_images)):
            value, confidence = _tokens_to_text(tokens)
            if column == "nonplastics_type":
                accepted = _compact_code(value)
                if accepted:
                    row[column] = accepted
                    if id(row) in interior_overlap_rows:
                        row["fabric_interior"] = _remove_cross_column_code(
                            row.get("fabric_interior", ""), accepted)
                if diagnostic is not None:
                    diagnostic.update({
                        "retry_tokens": [{
                            "text": item.text,
                            "confidence": round(item.score, 4),
                            "bbox": [round(value) for value in item.bbox],
                        } for item in tokens],
                        "raw_text": value,
                        "confidence": round(confidence, 4),
                        "accepted_value": accepted,
                        "status": "accepted" if accepted else "needs_review",
                    })
                    ocr_diagnostics.append(diagnostic)
            elif value:
                row[column] = value
        # If the page-level OCR merged No. into Type and the isolated retry was
        # empty, keep the useful type text but remove the duplicated row ID.
        for row in rows:
            value = row.get("table_type", "")
            number = re.escape(row.get("table_no", ""))
            if value and number:
                row["table_type"] = re.sub(
                    rf"^\s*{number}\s*[.)]?\s*", "", value, count=1,
                    flags=re.IGNORECASE).strip()
        boundary_data = boundary.as_dict()
        # Evidence coordinates are stored in original-page space even when the
        # linker supplied a same-page crop below the drawings.
        boundary_data["table_bounds"] = [
            boundary.left + crop_left, boundary.upper_header_rule + crop_top,
            boundary.right + crop_left, boundary.data_bottom + crop_top,
        ]
        for key in ("upper_header_rule", "lower_header_rule", "data_start_y", "data_end_y"):
            boundary_data[key] += crop_top
        if boundary_data["closing_rule_y"] is not None:
            boundary_data["closing_rule_y"] += crop_top
        boundary_data["column_bounds"] = [
            [left + crop_left, right + crop_left]
            for left, right in boundary_data.get("column_bounds", [])
        ]
        for anchor in boundary_data.get("header_anchors", []):
            anchor["bbox"] = [
                anchor["bbox"][0] + boundary.left + crop_left,
                anchor["bbox"][1] + crop_top,
                anchor["bbox"][2] + boundary.left + crop_left,
                anchor["bbox"][3] + crop_top,
            ]
        return {
            "is_table": True,
            "figure_id": page_context.get("figure_id", ""),
            "figure_caption": page_context.get("figure_caption", ""),
            "printed_page": page_context.get("printed_page", ""),
            "rows": rows,
            "warnings": warnings,
            "boundary": boundary_data,
            "ocr_diagnostics": ocr_diagnostics,
        }
