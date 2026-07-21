import cv2
import numpy as np
from PIL import Image, ImageDraw

from catalog.linkage import HESBAN_TABLE_COLUMNS
from processors.ocr import (
    OCRToken,
    PaddleOCRStructuredExtractor,
    _english_ocr_text,
    _parse_v3_result,
    _prepare_table_cell,
    _tokens_to_text,
)


class DrawingEngine:
    def recognize(self, image):
        return [OCRToken("1", .96, (20, image.height-55, 45, image.height-25))]

    def recognize_many(self, images):
        return [self.recognize(image) for image in images]


class TableEngine:
    cell_rows = [
        {"table_no": "1", "table_type": "Pithos", "table_locus": "143",
         "nonplastics_type": "L"},
        {"table_no": "2", "table_type": "Bowl", "table_locus": "144",
         "nonplastics_type": "L"},
    ]

    def recognize(self, image):
        if image.height < 220:
            headings = ["No.", "Type", "Sq", "Loc", "Pail", "Reg", "Exterior",
                        "Core", "Interior", "Typ", "Siz", "Shap", "Den", "Ty/Sz",
                        "Den", "Man", "Ext", "Color", "Int", "Color", "Decor", "Fire"]
            primary = {0, 1, 2, 3, 4, 5, 15, 20, 21}
            return [OCRToken(text, .99,
                             (5 + index*35, 20 if index in primary else 110,
                              30 + index*35, 40 if index in primary else 130))
                    for index, text in enumerate(headings)]
        if image.width > 500:
            return [
                OCRToken("1", .98, (10, 60, 24, 82)),
                OCRToken("Pithos", .98, (45, 60, 100, 82)),
                OCRToken("2", .97, (10, 220, 24, 242)),
                OCRToken("Bowl", .98, (45, 220, 90, 242)),
            ]
        return []

    def recognize_many(self, images):
        images = list(images)
        if images and len(images) % 22 == 0:
            row_count = len(images) // 22
            selected_rows = (self.cell_rows if row_count == len(self.cell_rows)
                             else self.cell_rows[-row_count:])
            output = []
            columns = list(HESBAN_TABLE_COLUMNS)
            for row in selected_rows:
                for column in columns:
                    value = row.get(column, "")
                    output.append([OCRToken(value, .97, (10, 10, 80, 35))]
                                  if value else [])
            return output
        return [[] for _ in images]


class ExtraRowEngine(TableEngine):
    cell_rows = TableEngine.cell_rows + [
        {"table_no": "99", "table_type": "Footer-like row"},
    ]

    def recognize(self, image):
        result = super().recognize(image)
        if image.height >= 220:
            result += [OCRToken("99", .99, (10, 380, 30, 402)),
                       OCRToken("Footer-like row", .99, (45, 380, 150, 402))]
        return result


class RetryRowEngine(ExtraRowEngine):
    cell_rows = [TableEngine.cell_rows[1]]


class WeakNumberCellEngine(TableEngine):
    """Model the blank and sub-threshold isolated-number failures."""

    def recognize_many(self, images):
        output = super().recognize_many(images)
        if len(output) >= 23:
            output[0] = []
            output[22] = [OCRToken("7", .13, (10, 10, 30, 35))]
        return output


class BoundaryTouchingNumberEngine(WeakNumberCellEngine):
    """Return narrow digits whose OCR boxes touch the No. cell boundary."""

    def recognize(self, image):
        result = super().recognize(image)
        if image.height >= 220 and image.width > 500:
            result[0] = OCRToken("1", .998, (-20, 60, 50, 82))
        return result


class PageAuthorityEngine(TableEngine):
    cell_rows = [dict(TableEngine.cell_rows[0], surface_exterior_color="**")]

    def recognize(self, image):
        result = super().recognize(image)
        if image.height >= 220 and image.width > 500:
            index = list(HESBAN_TABLE_COLUMNS).index("surface_exterior_color")
            left, right = index * 35, (index + 1) * 35
            result.append(OCRToken("**", .95, (left + 2, 60, right - 2, 82)))
        return result

    def recognize_many(self, images):
        output = super().recognize_many(images)
        if len(output) == 22:
            index = list(HESBAN_TABLE_COLUMNS).index("surface_exterior_color")
            output[index] = [OCRToken("术", .38, (10, 10, 35, 35))]
        return output


class MergedRowTypeEngine(TableEngine):
    cell_rows = [TableEngine.cell_rows[0]]

    def recognize(self, image):
        result = super().recognize(image)
        if image.height >= 220 and image.width > 500:
            result = [item for item in result if item.text not in {"1", "Pithos"}]
            result.append(OCRToken(
                "1 Pithos", .98,
                (1, 60, 69, 82)))
        return result

    def recognize_many(self, images):
        output = super().recognize_many(images)
        if len(output) == 22:
            output[1] = []
        return output


class BlankCellPageEngine(TableEngine):
    cell_rows = [TableEngine.cell_rows[0]]

    def recognize(self, image):
        result = super().recognize(image)
        if image.height >= 220 and image.width > 500:
            index = list(HESBAN_TABLE_COLUMNS).index("surface_exterior_color")
            left, right = index * 35, (index + 1) * 35
            result.append(OCRToken("**", .95, (left - 10, 60, right + 10, 82)))
        return result


class BoundaryNoisyCellEngine(BlankCellPageEngine):
    def recognize_many(self, images):
        output = super().recognize_many(images)
        if len(output) == 22:
            index = list(HESBAN_TABLE_COLUMNS).index("surface_exterior_color")
            output[index] = [OCRToken("noise*", .45, (10, 10, 60, 35))]
        return output


class BoundaryCleanCellEngine(BlankCellPageEngine):
    def recognize_many(self, images):
        output = super().recognize_many(images)
        if len(output) == 22:
            index = list(HESBAN_TABLE_COLUMNS).index("surface_exterior_color")
            output[index] = [OCRToken("**", .97, (10, 10, 35, 35))]
        return output


class FiringZeroEngine(TableEngine):
    cell_rows = [dict(TableEngine.cell_rows[0], fire="0")]


def make_table_image(path, closing=True):
    image = Image.new("RGB", (1000, 900), "white")
    draw = ImageDraw.Draw(image)
    draw.line((100, 100, 900, 100), fill="black", width=4)
    draw.line((100, 145, 900, 145), fill="black", width=4)
    if closing:
        draw.line((100, 760, 900, 760), fill="black", width=4)
    image.save(path)


def _dark_component_count(image):
    dark = (np.asarray(image.convert("L")) < 210).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    return sum(1 for index in range(1, count)
               if stats[index, cv2.CC_STAT_AREA] >= 10)


def test_local_ocr_reads_drawing_number_without_generative_model(tmp_path):
    image_path = tmp_path / "page.jpg"
    Image.new("RGB", (1000, 1200), "white").save(image_path)
    extractor = PaddleOCRStructuredExtractor(DrawingEngine())
    result = extractor.extract_drawing_identifiers(
        image_path,
        [{"mask_file": "page_mask_layer_0", "bbox": [100, 100, 400, 300]}],
        {"figure_id": "2.1", "figure_caption": "Figure 2.1", "printed_page": "19"},
    )
    assert result["figure_id"] == "2.1"
    assert result["drawings"]["page_mask_layer_0"]["number"] == "1"


def test_hesban_fixed_layout_builds_rows_from_ocr_coordinates(tmp_path):
    image_path = tmp_path / "table.jpg"
    make_table_image(image_path)
    extractor = PaddleOCRStructuredExtractor(TableEngine())
    result = extractor.extract_table(
        image_path, None, "2.1", ["1", "2"],
        {"figure_id": "2.1", "figure_caption": "Figure 2.1", "printed_page": "20"},
    )
    assert result["is_table"] is True
    assert [row["table_no"] for row in result["rows"]] == ["1", "2"]
    assert all(len(row) == 22 for row in result["rows"])
    row_bounds = result["boundary"]["row_bounds"]
    assert [item["row"] for item in row_bounds] == ["1", "2"]
    assert row_bounds[0]["top"] == result["boundary"]["data_start_y"]
    assert row_bounds[0]["bottom"] == row_bounds[1]["top"]
    assert row_bounds[-1]["bottom"] <= result["boundary"]["data_end_y"]
    cells = result["boundary"]["cell_diagnostics"]
    assert len(cells) == 44
    assert cells[0]["field"] == "table_no"
    assert cells[0]["crop"][1] == row_bounds[0]["top"]
    assert cells[-1]["crop"][3] == row_bounds[-1]["bottom"]


def test_every_cell_uses_the_same_cell_ocr_pass(tmp_path):
    image_path = tmp_path / "table.jpg"
    make_table_image(image_path)
    result = PaddleOCRStructuredExtractor(TableEngine()).extract_table(
        image_path, None, "2.1", ["1", "2"],
        {"figure_id": "2.1", "figure_caption": "Figure 2.1", "printed_page": "20"},
    )
    assert [row["nonplastics_type"] for row in result["rows"]] == ["L", "L"]
    assert [row["table_locus"] for row in result["rows"]] == ["143", "144"]
    cells = result["boundary"]["cell_diagnostics"]
    assert len(cells) == 44
    assert all(cell["accepted_source"] == "row_anchor"
               for cell in cells if cell["field"] == "table_no")
    assert all(cell["accepted_source"] == "cell_pass"
               for cell in cells
               if cell["field"] != "table_no" and cell["focused_text"])
    assert all("focused_text" in cell for cell in cells)
    assert result["ocr_diagnostics"] == []


def test_safe_page_token_survives_blank_or_weak_cell_detection(tmp_path):
    image_path = tmp_path / "table.jpg"
    make_table_image(image_path)
    result = PaddleOCRStructuredExtractor(WeakNumberCellEngine()).extract_table(
        image_path, None, "2.1", ["1", "2"],
        {"figure_id": "2.1", "figure_caption": "Figure 2.1",
         "printed_page": "20"},
    )
    assert [row["table_no"] for row in result["rows"]] == ["1", "2"]
    number_cells = [cell for cell in result["boundary"]["cell_diagnostics"]
                    if cell["field"] == "table_no"]
    assert [cell["accepted_source"] for cell in number_cells] == [
        "row_anchor", "row_anchor"]
    assert number_cells[1]["focused_text"] == ""
    assert number_cells[1]["focused_tokens"][0]["text"] == "7"


def test_boundary_touching_page_number_is_not_erased_by_blank_cell(tmp_path):
    image_path = tmp_path / "table.jpg"
    make_table_image(image_path)
    result = PaddleOCRStructuredExtractor(
        BoundaryTouchingNumberEngine()).extract_table(
            image_path, None, "2.1", ["1", "2"],
            {"figure_id": "2.1", "figure_caption": "Figure 2.1",
             "printed_page": "20"},
        )
    first_number = next(
        cell for cell in result["boundary"]["cell_diagnostics"]
        if cell["row"] == "1" and cell["field"] == "table_no")
    assert first_number["safe_initial_text"] == ""
    assert first_number["initial_text"] == "1"
    assert first_number["accepted_text"] == "1"
    assert first_number["accepted_source"] == "row_anchor"


def test_strong_page_value_beats_noisy_low_confidence_cell_value(tmp_path):
    image_path = tmp_path / "table.jpg"
    make_table_image(image_path)
    result = PaddleOCRStructuredExtractor(PageAuthorityEngine()).extract_table(
        image_path, None, "2.1", ["1"], {"figure_id": "2.1"})
    cell = next(item for item in result["boundary"]["cell_diagnostics"]
                if item["field"] == "surface_exterior_color")
    assert cell["initial_text"] == "**"
    assert cell["focused_text"] == ""
    assert cell["focused_tokens"][0]["text"] == "术"
    assert cell["accepted_text"] == "**"
    assert cell["accepted_source"] == "page_pass_blank_cell"


def test_non_english_glyphs_are_removed_from_values_but_raw_tokens_survive():
    tokens = [OCRToken("\u6c34*", .93, (10, 10, 40, 35))]
    text, confidence = _tokens_to_text(tokens)
    assert text == "*"
    assert confidence == .93
    assert tokens[0].text == "\u6c34*"
    assert _english_ocr_text("\uff21\uff11\uff0a") == "A1*"


def test_merged_row_anchor_is_removed_from_page_cell_candidate(tmp_path):
    image_path = tmp_path / "table.jpg"
    make_table_image(image_path)
    result = PaddleOCRStructuredExtractor(MergedRowTypeEngine()).extract_table(
        image_path, None, "2.1", ["1"], {"figure_id": "2.1"})
    cell = next(item for item in result["boundary"]["cell_diagnostics"]
                if item["field"] == "table_type")
    assert cell["initial_tokens"][0]["text"] == "1 Pithos"
    assert cell["initial_text"] == "Pithos"
    assert cell["focused_text"] == ""
    assert cell["accepted_text"] == "Pithos"
    assert cell["accepted_source"] == "page_pass_blank_cell"


def test_cross_column_page_value_is_withheld_when_cell_is_blank(tmp_path):
    image_path = tmp_path / "table.jpg"
    make_table_image(image_path)
    result = PaddleOCRStructuredExtractor(BlankCellPageEngine()).extract_table(
        image_path, None, "2.1", ["1"], {"figure_id": "2.1"})
    cell = next(item for item in result["boundary"]["cell_diagnostics"]
                if item["field"] == "surface_exterior_color")
    assert cell["focused_text"] == ""
    assert cell["safe_initial_text"] == ""
    assert cell["initial_text"] == "**"
    assert cell["accepted_text"] == ""
    assert cell["accepted_source"] == "cross_column_page_withheld"
    assert cell["page_geometry_reliable"] is False
    assert "crossed a cell boundary" in cell["decision_reason"]


def test_cross_column_page_value_does_not_beat_noisy_cell_reading(tmp_path):
    image_path = tmp_path / "table.jpg"
    make_table_image(image_path)
    result = PaddleOCRStructuredExtractor(BoundaryNoisyCellEngine()).extract_table(
        image_path, None, "2.1", ["1"], {"figure_id": "2.1"})
    cell = next(item for item in result["boundary"]["cell_diagnostics"]
                if item["field"] == "surface_exterior_color")
    assert cell["safe_initial_text"] == ""
    assert cell["initial_text"] == "**"
    assert cell["initial_confidence"] == .95
    assert cell["focused_text"] == "noise*"
    assert cell["focused_confidence"] == .45
    assert cell["accepted_text"] == "noise*"
    assert cell["accepted_source"] == "cell_pass"
    assert cell["page_geometry_reliable"] is False


def test_cross_column_page_value_does_not_replace_clear_cell_reading(tmp_path):
    image_path = tmp_path / "table.jpg"
    make_table_image(image_path)
    result = PaddleOCRStructuredExtractor(BoundaryCleanCellEngine()).extract_table(
        image_path, None, "2.1", ["1"], {"figure_id": "2.1"})
    cell = next(item for item in result["boundary"]["cell_diagnostics"]
                if item["field"] == "surface_exterior_color")
    assert cell["initial_text"] == "**"
    assert cell["safe_initial_text"] == ""
    assert cell["focused_text"] == "**"
    assert cell["accepted_text"] == "**"
    assert cell["accepted_source"] == "cell_pass"


def test_firing_zero_is_strictly_normalized_to_letter_o(tmp_path):
    image_path = tmp_path / "table.jpg"
    make_table_image(image_path)
    result = PaddleOCRStructuredExtractor(FiringZeroEngine()).extract_table(
        image_path, None, "2.1", ["1"], {"figure_id": "2.1"})
    cell = next(item for item in result["boundary"]["cell_diagnostics"]
                if item["field"] == "fire")
    assert cell["focused_text"] == "0"
    assert cell["accepted_text"] == "O"
    assert cell["normalizations"] == ["fire_zero_to_letter_o"]


def test_cell_input_expands_but_assigns_crossing_ink_group_only_once():
    image = Image.new("L", (220, 70), 255)
    draw = ImageDraw.Draw(image)
    # Six close marks form one word-like group. Its final mark crosses the
    # calculated boundary at x=150. The separate mark at x=190 belongs to the
    # following cell.
    for left in (90, 102, 114, 126, 138, 150):
        draw.rectangle((left, 20, left + 7, 49), fill=0)
    draw.rectangle((190, 20, 199, 49), fill=0)

    left_cell = _prepare_table_cell(image, [40, 0, 150, 70])
    right_cell = _prepare_table_cell(image, [150, 0, 210, 70])

    left_components = _dark_component_count(left_cell)
    right_components = _dark_component_count(right_cell)
    assert left_components == 6
    assert right_components == 1


def test_hesban_column_template_has_one_band_per_csv_field():
    bounds = PaddleOCRStructuredExtractor._column_bounds(4000)
    assert len(bounds) == 22
    assert all(left < right for left, right in bounds)
    assert all(bounds[index][1] == bounds[index+1][0] for index in range(21))


def test_merged_header_words_follow_visible_ink_valleys():
    image = Image.new("L", (380, 70), 255)
    draw = ImageDraw.Draw(image)
    word_starts = (12, 92, 180, 292)
    letter_counts = (3, 3, 4, 3)
    expected = []
    for word_start, letter_count in zip(word_starts, letter_counts):
        for index in range(letter_count):
            left = word_start + index * 10
            draw.rectangle((left, 18, left + 6, 49), fill=0)
        expected.append((word_start, word_start + (letter_count - 1) * 10 + 7))
    merged = OCRToken("Typ Siz Shap Den", .99, (5, 10, 330, 58))

    split = PaddleOCRStructuredExtractor._split_header_tokens([merged], image)

    assert [token.text for token in split] == ["Typ", "Siz", "Shap", "Den"]
    assert [(token.bbox[0], token.bbox[2]) for token in split] == expected


def test_long_merged_nonplastics_and_voids_header_is_split():
    merged = OCRToken("Typ Siz Shap Den Ty/Sz Den", .99, (10, 10, 610, 58))
    split = PaddleOCRStructuredExtractor._split_header_tokens([merged])
    assert [token.text for token in split] == [
        "Typ", "Siz", "Shap", "Den", "Ty/Sz", "Den"]
    assert all(left.center_x < right.center_x for left, right in zip(split, split[1:]))


def test_full_header_accepts_merged_second_half_groups():
    primary_labels = ["No", "Type", "Sq", "Loc", "Pail", "Reg"]
    tokens = [OCRToken(label, .98, (20 + index * 75, 20,
                                    60 + index * 75, 55))
              for index, label in enumerate(primary_labels)]
    tokens.extend([
        OCRToken("Exterior", .98, (500, 95, 570, 130)),
        OCRToken("Core", .98, (600, 95, 650, 130)),
        OCRToken("Interior", .98, (690, 95, 760, 130)),
        OCRToken("Typ Siz Shap Den Ty/Sz Den", .98, (790, 95, 1110, 130)),
        OCRToken("Man", .98, (1135, 20, 1175, 55)),
        OCRToken("Ext Color Int Color", .98, (1190, 95, 1390, 130)),
        OCRToken("Decor", .98, (1410, 20, 1470, 55)),
        OCRToken("Fire", .98, (1500, 20, 1545, 55)),
    ])

    bounds, source, evidence = (
        PaddleOCRStructuredExtractor._column_bounds_from_header(
            tokens, 1580, upper_rule_y=70))

    assert source == "header_detected"
    assert len(bounds) == 22
    assert [item["column"] for item in evidence] == HESBAN_TABLE_COLUMNS


def test_single_header_token_is_not_changed_by_visual_splitter():
    image = Image.new("L", (120, 50), 255)
    token = OCRToken("Typ", .99, (10, 10, 60, 35))
    assert PaddleOCRStructuredExtractor._split_header_tokens(
        [token], image) == [token]


def test_only_an_abnormally_tall_final_row_is_capped():
    anchors = [
        ("1", OCRToken("1", .99, (0, 60, 10, 80))),
        ("2", OCRToken("2", .99, (0, 160, 10, 180))),
        ("3", OCRToken("3", .99, (0, 260, 10, 280))),
    ]
    bounds = PaddleOCRStructuredExtractor._row_bounds(anchors, 1000)
    assert bounds[0] == ("1", 0, 157)
    assert bounds[1] == ("2", 157, 257)
    assert bounds[2] == ("3", 257, 557)


def test_normal_final_row_keeps_the_verified_table_ending():
    anchors = [
        ("1", OCRToken("1", .99, (0, 60, 10, 80))),
        ("2", OCRToken("2", .99, (0, 220, 10, 240))),
    ]
    bounds = PaddleOCRStructuredExtractor._row_bounds(anchors, 600)
    assert bounds[-1] == ("2", 217, 543)


def test_columns_follow_this_pages_header_positions():
    # Deliberately non-uniform centers model the differently stretched tables
    # in the supplied Hesban examples.
    centers = [18, 88, 174, 226, 278, 340, 430, 520, 615, 682, 724,
               772, 818, 865, 910, 950, 985, 1028, 1070, 1118, 1170, 1225]
    labels = ["No.", "Type", "Sq", "Loc", "Pail", "Reg", "Exterior", "Core",
              "Interior", "Typ", "Siz", "Shap", "Den", "Ty/Sz", "Den", "Man",
              "Ext", "Color", "Int", "Color", "Decor", "Fire"]
    direct = {0, 1, 2, 3, 4, 5, 15, 20, 21}
    tokens = [OCRToken(label, .99,
                       (center-10, 20 if index in direct else 105,
                        center+10, 40 if index in direct else 125))
              for index, (center, label) in enumerate(zip(centers, labels))]
    bounds, source, evidence = PaddleOCRStructuredExtractor._column_bounds_from_header(
        tokens, 1260, upper_rule_y=75)
    assert source == "header_detected"
    assert len(bounds) == 22
    assert len(evidence) == 22
    # A boundary approaches the next heading instead of splitting the gap in
    # half, so long Type values retain the whitespace before Sq.
    assert bounds[0] == (4, 74)
    assert bounds[1] == (74, 160)
    assert bounds[-1][1] == 1260


def test_repeated_headers_follow_left_to_right_order_not_confidence():
    centers = [18 + index * 52 for index in range(22)]
    labels = ["No.", "Type", "Sq", "Loc", "Pail", "Reg", "Exterior", "Core",
              "Interior", "Typ", "Siz", "Shap", ")Den", "Ty/Sz", "Den", "Man",
              "Ext", "Color", "Int", "Color", "Decor", "Fire"]
    primary = {0, 1, 2, 3, 4, 5, 15, 20, 21}
    confidence = [0.99] * 22
    confidence[12] = .72
    confidence[14] = .999
    confidence[17] = .71
    confidence[19] = .999
    tokens = [OCRToken(label, confidence[index],
                       (center - 12, 20 if index in primary else 105,
                        center + 18, 42 if index in primary else 127))
              for index, (center, label) in enumerate(zip(centers, labels))]

    bounds, source, evidence = PaddleOCRStructuredExtractor._column_bounds_from_header(
        tokens, 1200, upper_rule_y=75)

    assert source == "header_detected"
    assert len(bounds) == 22
    anchors = {item["column"]: item for item in evidence}
    assert anchors["nonplastics_density"]["text"] == ")Den"
    assert anchors["voids_density"]["bbox"][0] > anchors["nonplastics_density"]["bbox"][0]
    assert anchors["surface_exterior_color"]["bbox"][0] < anchors["surface_interior_color"]["bbox"][0]


def test_area_alias_replaces_sq_without_changing_storage_column():
    centers = [18 + index * 52 for index in range(22)]
    labels = ["No.", "Type", "Area", "Loc", "Pail", "Reg", "Exterior", "Core",
              "Interior", "Typ", "Siz", "Shap", "Den", "Ty/Sz", "Den", "Man",
              "Ext", "Color", "Int", "Color", "Decor", "Fire"]
    primary = {0, 1, 2, 3, 4, 5, 15, 20, 21}
    tokens = [OCRToken(label, .99, (center - 12, 20 if index in primary else 105,
                                    center + 18, 42 if index in primary else 127))
              for index, (center, label) in enumerate(zip(centers, labels))]
    bounds, source, evidence = PaddleOCRStructuredExtractor._column_bounds_from_header(
        tokens, 1160, upper_rule_y=75)
    assert source == "header_detected"
    assert len(bounds) == 22
    assert evidence[2]["column"] == "table_square"
    assert evidence[2]["text"] == "Area"


def test_header_lead_offset_scales_with_rendered_text_height():
    labels = ["No.", "Type", "Sq", "Loc", "Pail", "Reg", "Exterior", "Core",
              "Interior", "Typ", "Siz", "Shap", "Den", "Ty/Sz", "Den", "Man",
              "Ext", "Color", "Int", "Color", "Decor", "Fire"]
    primary = {0, 1, 2, 3, 4, 5, 15, 20, 21}
    small = [OCRToken(label, .99, (8 + index * 50,
                                   20 if index in primary else 105,
                                   32 + index * 50,
                                   40 if index in primary else 125))
             for index, label in enumerate(labels)]
    large = [OCRToken(label, .99, (16 + index * 100,
                                   40 if index in primary else 210,
                                   64 + index * 100,
                                   80 if index in primary else 250))
             for index, label in enumerate(labels)]
    small_bounds, _, _ = PaddleOCRStructuredExtractor._column_bounds_from_header(
        small, 1120, upper_rule_y=75)
    large_bounds, _, _ = PaddleOCRStructuredExtractor._column_bounds_from_header(
        large, 2240, upper_rule_y=150)
    assert large_bounds[0][0] == small_bounds[0][0] * 2
    assert large_bounds[1][0] == small_bounds[1][0] * 2


def test_group_headings_never_become_column_anchors():
    centers = [18 + index * 52 for index in range(22)]
    labels = ["No.", "Type", "Sq", "Loc", "Pail", "Reg", "Exterior", "Core",
              "Interior", "Typ", "Siz", "Shap", "Den", "Ty/Sz", "Den", "Man",
              "Ext", "Color", "Int", "Color", "Decor", "Fire"]
    primary = {0, 1, 2, 3, 4, 5, 15, 20, 21}
    tokens = [OCRToken(label, .99, (center - 10, 20 if index in primary else 105,
                                    center + 16, 42 if index in primary else 127))
              for index, (center, label) in enumerate(zip(centers, labels))]
    tokens += [OCRToken("Fabric Color", .99, (300, 12, 500, 34)),
               OCRToken("Non-Plastics", .99, (510, 12, 720, 34)),
               OCRToken("Voids", .99, (720, 12, 820, 34)),
               OCRToken("Surface Treatment", .99, (830, 12, 1050, 34))]
    bounds, source, evidence = PaddleOCRStructuredExtractor._column_bounds_from_header(
        tokens, 1160, upper_rule_y=75)
    assert source == "header_detected"
    assert len(bounds) == len(evidence) == 22
    assert not {"Fabric Color", "Non-Plastics", "Voids", "Surface Treatment"} & {
        item["text"] for item in evidence}


def test_damaged_header_fails_closed_for_manual_review():
    tokens = [OCRToken("No.", .99, (5, 10, 20, 30)),
              OCRToken("Type", .99, (30, 10, 70, 30))]
    bounds, source, evidence = PaddleOCRStructuredExtractor._column_bounds_from_header(
        tokens, 1000, upper_rule_y=60)
    assert source == "headers_incomplete"
    assert bounds == []
    assert [item["column"] for item in evidence] == ["table_no", "table_type"]


def test_row_anchor_recovers_number_merged_with_type():
    extractor = PaddleOCRStructuredExtractor(TableEngine())
    image = Image.new("RGB", (800, 500), "white")
    bounds = extractor._column_bounds(image.width)
    merged = OCRToken("14Jar/Jug", .96,
                      (bounds[0][0], 80, bounds[1][1], 105))
    anchors = extractor._row_anchors(image, ["14"], [merged])
    assert [number for number, _ in anchors] == ["14"]


def test_conflicting_row_numbers_at_same_height_create_only_one_row(monkeypatch):
    extractor = PaddleOCRStructuredExtractor(TableEngine())
    image = Image.new("RGB", (800, 500), "white")
    bounds = extractor._column_bounds(image.width)
    page_tokens = [
        OCRToken("11", .98, (bounds[0][0] + 2, 80, bounds[0][1] - 2, 104)),
        OCRToken("Cooking pot", .98, (bounds[1][0] + 2, 80,
                                      bounds[1][1] - 2, 104)),
        OCRToken("12", .98, (bounds[0][0] + 2, 170, bounds[0][1] - 2, 194)),
        OCRToken("Cooking pot", .98, (bounds[1][0] + 2, 170,
                                      bounds[1][1] - 2, 194)),
    ]
    monkeypatch.setattr(extractor, "_number_component_tokens", lambda *_: [
        OCRToken("1", .99, (bounds[0][0], 81, bounds[0][1], 103)),
        OCRToken("2", .99, (bounds[0][0], 171, bounds[0][1], 193)),
    ])
    conflicts = []

    anchors = extractor._row_anchors(
        image, ["1", "2", "11", "12"], page_tokens,
        bounds=bounds, conflicts_out=conflicts)

    assert [number for number, _ in anchors] == ["11", "12"]
    assert [item["chosen"] for item in conflicts] == ["11", "12"]
    assert [bottom - top for _, top, bottom in extractor._row_bounds(anchors, 400)] == [167, 233]


def test_row_bounds_start_at_data_rule_and_split_before_next_number():
    anchors = [
        ("1", OCRToken("1", .99, (0, 40, 12, 60))),
        ("2", OCRToken("2", .99, (0, 150, 12, 170))),
        ("3", OCRToken("3", .99, (0, 310, 12, 330))),
    ]
    assert PaddleOCRStructuredExtractor._row_bounds(anchors, 500) == [
        ("1", 0, 147), ("2", 147, 307), ("3", 307, 500),
    ]


def test_initial_ocr_pass_keeps_unexpected_rows_for_validation(tmp_path):
    image_path = tmp_path / "table.jpg"
    make_table_image(image_path)
    result = PaddleOCRStructuredExtractor(ExtraRowEngine()).extract_table(
        image_path, None, "2.1", ["1", "2"], {"figure_id": "2.1"})
    assert [row["table_no"] for row in result["rows"]] == ["1", "2", "99"]


def test_retry_pass_limits_rows_to_requested_missing_numbers(tmp_path):
    image_path = tmp_path / "table.jpg"
    make_table_image(image_path)
    result = PaddleOCRStructuredExtractor(RetryRowEngine()).extract_table(
        image_path, None, "2.1", ["2"], {"figure_id": "2.1", "retry_missing": True})
    assert [row["table_no"] for row in result["rows"]] == ["2"]
    rows = {item["row"]: item for item in result["boundary"]["row_bounds"]}
    number_cell = next(item for item in result["boundary"]["cell_diagnostics"]
                       if item["field"] == "table_no")
    assert number_cell["row"] == "2"
    assert number_cell["crop"][1] == rows["2"]["top"]


def test_malformed_paddle_result_is_withheld_instead_of_crashing():
    assert _parse_v3_result("{not valid json") == []
    assert _parse_v3_result({"res": []}) == []


def test_page_without_verified_header_is_not_a_table(tmp_path):
    image_path = tmp_path / "not-table.jpg"
    Image.new("RGB", (1000, 900), "white").save(image_path)
    result = PaddleOCRStructuredExtractor(TableEngine()).extract_table(
        image_path, None, "2.1", ["1", "2"], {"figure_id": "2.1"})
    assert result["is_table"] is False
    assert result["warnings"][0]["code"] == "table_header_not_found"


class MissingHeaderEngine(TableEngine):
    def recognize(self, image):
        result = super().recognize(image)
        if image.height < 220:
            return [token for token in result if token.text != "Core"]
        return result


def test_incomplete_page_header_is_flagged_without_guessing_rows(tmp_path):
    image_path = tmp_path / "incomplete.jpg"
    make_table_image(image_path)
    result = PaddleOCRStructuredExtractor(MissingHeaderEngine()).extract_table(
        image_path, None, "2.1", ["1", "2"], {"figure_id": "2.1"})
    assert result["is_table"] is False
    assert result["needs_column_review"] is True
    assert result["rows"] == []
    assert result["warnings"][0]["code"] == "column_headers_incomplete"
    assert len(result["boundary"]["header_anchors"]) == 21
    assert result["boundary"]["image_size"] == [1000, 900]
    status = result["boundary"]["diagnostic_status"]
    assert status["found_count"] == 21
    assert status["required_count"] == 22
    assert status["missing_labels"] == ["Core"]
    assert "did not start" in status["message"]


def test_saved_manual_edges_override_detected_header_edges(tmp_path):
    image_path = tmp_path / "manual.jpg"
    make_table_image(image_path)
    page_edges = [100 + index * 36 for index in range(22)] + [900]
    result = PaddleOCRStructuredExtractor(TableEngine()).extract_table(
        image_path, None, "2.1", ["1", "2"], {
            "figure_id": "2.1",
            "manual_column_edges": [edge / 1000 for edge in page_edges],
        })
    assert result["is_table"] is True
    assert result["boundary"]["column_source"] == "manual"
    assert result["boundary"]["column_edges"] == page_edges


def test_missing_closing_rule_is_recorded_as_continuation(tmp_path):
    image_path = tmp_path / "continued.jpg"
    make_table_image(image_path, closing=False)
    result = PaddleOCRStructuredExtractor(TableEngine()).extract_table(
        image_path, None, "2.1", ["1", "2"], {"figure_id": "2.1"})
    assert result["is_table"] is True
    assert result["boundary"]["header_confirmed"] is True
    assert result["boundary"]["has_closing_rule"] is False
    assert result["boundary"]["continues"] is True


class OutsideNoiseEngine(TableEngine):
    def recognize(self, image):
        result = super().recognize(image)
        # This simulates page-number/footer tokens that would only be returned
        # by a whole-page OCR call. The verified data crop is shorter.
        if image.height > 800:
            result += [OCRToken("20", .99, (10, 10, 30, 30)),
                       OCRToken("CHAPTER HEADING", .99, (45, 10, 200, 30))]
        elif image.height >= 220:
            # A number in a metadata column must not become a No. anchor.
            result += [OCRToken("99", .99, (400, 360, 425, 382)),
                       OCRToken("text", .99, (450, 360, 500, 382))]
        return result


def test_page_number_footer_and_numbers_outside_no_column_are_excluded(tmp_path):
    image_path = tmp_path / "bounded.jpg"
    make_table_image(image_path)
    result = PaddleOCRStructuredExtractor(OutsideNoiseEngine()).extract_table(
        image_path, None, "2.1", ["1", "2"], {"figure_id": "2.1"})
    assert [row["table_no"] for row in result["rows"]] == ["1", "2"]


def test_broken_faint_header_rules_are_merged(tmp_path):
    image_path = tmp_path / "broken.png"
    image = Image.new("RGB", (1000, 900), "white")
    draw = ImageDraw.Draw(image)
    for y in (100, 145, 760):
        for left in range(100, 900, 90):
            draw.line((left, y, min(left+78, 900), y), fill=(105, 105, 105), width=4)
    image.save(image_path)
    result = PaddleOCRStructuredExtractor(TableEngine()).extract_table(
        image_path, None, "2.1", ["1", "2"], {"figure_id": "2.1"})
    assert result["is_table"] is True
    assert result["boundary"]["closing_rule_y"] is not None


def test_slightly_skewed_rules_are_deskewed_for_detection(tmp_path):
    base = tmp_path / "base.png"
    image_path = tmp_path / "skewed.png"
    make_table_image(base)
    Image.open(base).rotate(1.2, resample=Image.Resampling.BICUBIC,
                            expand=False, fillcolor="white").save(image_path)
    result = PaddleOCRStructuredExtractor(TableEngine()).extract_table(
        image_path, None, "2.1", ["1", "2"], {"figure_id": "2.1"})
    assert result["is_table"] is True
