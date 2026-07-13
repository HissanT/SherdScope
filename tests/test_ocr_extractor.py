from pathlib import Path

from PIL import Image, ImageDraw

from ocr_extractor import OCRToken, PaddleOCRStructuredExtractor, _parse_v3_result


class DrawingEngine:
    def recognize(self, image):
        return [OCRToken("1", .96, (20, image.height-55, 45, image.height-25))]

    def recognize_many(self, images):
        return [self.recognize(image) for image in images]


class TableEngine:
    def recognize(self, image):
        if image.height < 220:
            headings = ["No.", "Type", "Sq", "Loc", "Fabric Color", "Non-Plastics",
                        "Voids", "Surface Treatment", "Exterior", "Core", "Interior", "Typ", "Den"]
            return [OCRToken(text, .99, (10 + index*45, 10, 45 + index*45, 28))
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
        return [[] for _ in images]


class ExtraRowEngine(TableEngine):
    def recognize(self, image):
        result = super().recognize(image)
        if image.height >= 220:
            result += [OCRToken("99", .99, (10, 380, 30, 402)),
                       OCRToken("Footer-like row", .99, (45, 380, 150, 402))]
        return result


def make_table_image(path, closing=True):
    image = Image.new("RGB", (1000, 900), "white")
    draw = ImageDraw.Draw(image)
    draw.line((100, 100, 900, 100), fill="black", width=4)
    draw.line((100, 145, 900, 145), fill="black", width=4)
    if closing:
        draw.line((100, 760, 900, 760), fill="black", width=4)
    image.save(path)


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


def test_hesban_column_template_has_one_band_per_csv_field():
    bounds = PaddleOCRStructuredExtractor._column_bounds(4000)
    assert len(bounds) == 22
    assert all(left < right for left, right in bounds)
    assert all(bounds[index][1] == bounds[index+1][0] for index in range(21))
    # Manufacture is deliberately narrower than Surface Ext so merged Ext
    # marks are not assigned to Man.
    assert bounds[15][1] - bounds[15][0] < bounds[16][1] - bounds[16][0]


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
    assert source == "header_ocr"
    assert len(bounds) == 22
    assert len(evidence) == 22
    # A boundary approaches the next heading instead of splitting the gap in
    # half, so long Type values retain the whitespace before Sq.
    assert bounds[0][1] == 69
    assert bounds[1] == (69, 152)
    assert bounds[-1][1] == 1260


def test_damaged_header_uses_explicitly_flagged_fallback():
    tokens = [OCRToken("No.", .99, (5, 10, 20, 30)),
              OCRToken("Type", .99, (30, 10, 70, 30))]
    bounds, source, evidence = PaddleOCRStructuredExtractor._column_bounds_from_header(
        tokens, 1000, upper_rule_y=60)
    assert source == "fixed_fallback"
    assert bounds == PaddleOCRStructuredExtractor._column_bounds(1000)
    assert evidence == []


def test_row_anchor_recovers_number_merged_with_type():
    extractor = PaddleOCRStructuredExtractor(TableEngine())
    image = Image.new("RGB", (800, 500), "white")
    bounds = extractor._column_bounds(image.width)
    merged = OCRToken("14Jar/Jug", .96,
                      (bounds[0][0], 80, bounds[1][1], 105))
    anchors = extractor._row_anchors(image, ["14"], [merged])
    assert [number for number, _ in anchors] == ["14"]


def test_initial_ocr_pass_keeps_unexpected_rows_for_validation(tmp_path):
    image_path = tmp_path / "table.jpg"
    make_table_image(image_path)
    result = PaddleOCRStructuredExtractor(ExtraRowEngine()).extract_table(
        image_path, None, "2.1", ["1", "2"], {"figure_id": "2.1"})
    assert [row["table_no"] for row in result["rows"]] == ["1", "2", "99"]


def test_retry_pass_limits_rows_to_requested_missing_numbers(tmp_path):
    image_path = tmp_path / "table.jpg"
    make_table_image(image_path)
    result = PaddleOCRStructuredExtractor(ExtraRowEngine()).extract_table(
        image_path, None, "2.1", ["2"], {"figure_id": "2.1", "retry_missing": True})
    assert [row["table_no"] for row in result["rows"]] == ["2"]


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
