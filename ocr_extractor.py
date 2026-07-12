"""Local OCR extraction for Hesban 11 figure/table linking.

PaddleOCR only reads text and its coordinates.  The deterministic code in this
module uses the fixed Hesban 11 layout to turn those coordinates into table cells;
it does not use a generative model to infer or rewrite metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
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
        }


def _prepare_image(image: Image.Image, min_height: int = 0) -> Image.Image:
    gray = ImageOps.autocontrast(image.convert("L"), cutoff=1).filter(ImageFilter.SHARPEN)
    if min_height and gray.height < min_height:
        scale = min(4.0, min_height / max(1, gray.height))
        gray = gray.resize((max(1, round(gray.width * scale)), round(gray.height * scale)),
                           Image.Resampling.LANCZOS)
    return gray.convert("RGB")


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
                "PaddleOCR is not installed. Install requirements-ocr.txt, then restart PyPotteryLens."
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
            header_text, _ = _tokens_to_text(self.engine.recognize(header_crop), minimum_score=.18)
            if not self._header_is_hesban(header_text):
                continue
            left = max(0, min(upper[1], lower[1]))
            right = min(image.width, max(upper[2], lower[2]))
            data_top = min(height, lower[0] + max(2, lower[3] // 2 + 1))
            closing = next((rule for rule in rules if rule[0] > data_top + max(8, height*.002)), None)
            closing_y = closing[0] if closing else None
            data_bottom = closing_y if closing_y is not None else height
            return TableBoundary(left, right, upper[0], lower[0], data_top,
                                 data_bottom, closing_y, header_text)
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

    def _number_component_tokens(self, image: Image.Image) -> list[OCRToken]:
        """OCR each small item in the printed No column.

        Whole-page detectors occasionally merge ``4`` with ``Pithos`` or miss
        the very narrow ``1``.  A simple vertical ink projection finds those
        number-sized components without guessing their values; OCR still reads
        the actual printed characters.
        """
        width = image.width
        left, right = self._column_bounds(width)[0]
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
        context_right = self._column_bounds(width)[1][1]
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
                     restrict_to_expected: bool = False) -> list[tuple[str, OCRToken]]:
        width = image.width
        bounds = self._column_bounds(width)
        number_left, number_right = bounds[0]
        tokens = page_tokens if page_tokens is not None else self.engine.recognize(image)
        component_tokens = self._number_component_tokens(image)
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
        # The first pass must retain unexpected rows so validation can detect a
        # wrong/extra table. Only the targeted truncation retry may filter to
        # the explicitly missing numbers.
        anchors = self._row_anchors(
            data_image, expected_numbers, page_tokens,
            restrict_to_expected=bool(page_context.get("retry_missing")))
        if not anchors:
            return {"is_table": False, "rows": [], "boundary": boundary.as_dict()}
        differences = [anchors[index+1][1].bbox[1] - anchors[index][1].bbox[1]
                       for index in range(len(anchors)-1) if anchors[index+1][1].bbox[1] > anchors[index][1].bbox[1]]
        default_gap = median(differences) if differences else data_image.height * .09
        bounds = self._column_bounds(data_image.width)
        rows = []
        warnings = []
        refinement_images = []
        refinement_targets: list[tuple[dict[str, str], str]] = []
        for index, (number, token) in enumerate(anchors):
            row_top = max(0, round(token.bbox[1] - 4))
            if index+1 < len(anchors):
                row_bottom = max(row_top+5, round(anchors[index+1][1].bbox[1] - 3))
            else:
                row_bottom = data_image.height
            row = {column: "" for column in HESBAN_TABLE_COLUMNS}
            row["table_no"] = number
            for column, (x1, x2) in zip(HESBAN_TABLE_COLUMNS[1:], bounds[1:]):
                cell_tokens = [item for item in page_tokens
                               if x1 <= item.center_x < x2
                               and row_top <= item.center_y < row_bottom]
                value, confidence = _tokens_to_text(cell_tokens)
                row[column] = value
                if value and confidence < .45:
                    warnings.append({
                        "code": "low_ocr_confidence",
                        "message": f"Review figure {figure_id}, row {number}, field {column}.",
                    })
            # Whole-page OCR often skips repeated Type words and joins narrow
            # adjacent columns (Pail/Reg and Man/Surface Ext). Re-read those
            # cells independently in one batch.
            for column_index in (1, 4, 5, 15, 16):
                column = HESBAN_TABLE_COLUMNS[column_index]
                x1, x2 = bounds[column_index]
                refinement_images.append(
                    _prepare_image(data_image.crop((x1, row_top, x2, max(row_top+1, row_bottom))),
                                   min_height=100))
                refinement_targets.append((row, column))
            rows.append(row)
        for (row, column), tokens in zip(
                refinement_targets, self.engine.recognize_many(refinement_images)):
            value, _ = _tokens_to_text(tokens)
            if value:
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
        return {
            "is_table": True,
            "figure_id": page_context.get("figure_id", ""),
            "figure_caption": page_context.get("figure_caption", ""),
            "printed_page": page_context.get("printed_page", ""),
            "rows": rows,
            "warnings": warnings,
            "boundary": boundary_data,
        }
