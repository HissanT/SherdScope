from PIL import Image

import catalog.matcher_annotation as matcher_annotation
from catalog.matcher_annotation import create_annotation_app


def test_annotation_editor_accepts_40_numbered_queries(tmp_path):
    project = tmp_path / "project"
    queries = tmp_path / "queries"
    queries.mkdir()
    for number in range(1, 41):
        Image.new("L", (24, 24), 255).save(queries / f"Query{number}.png")

    app = create_annotation_app(
        project, queries, expected_count=40, set_name="hesban_40_new"
    )
    client = app.test_client()
    page = client.get("/")
    queue = client.get("/queue").get_json()["queries"]

    assert page.status_code == 200
    assert b"Prepare all 40 Hesban queries" in page.data
    assert len(queue) == 40
    assert queue[-1]["number"] == 40


def test_sorted_inverted_masks_preview_before_save(tmp_path, monkeypatch):
    project = tmp_path / "project"
    queries = tmp_path / "queries"
    queries.mkdir()
    for name in ("IMG_4612.png", "IMG_4610.png"):
        image = Image.new("L", (24, 24), 0)
        for x in range(8, 16):
            for y in range(6, 19):
                image.putpixel((x, y), 255)
        image.save(queries / name)

    seen = {}

    def trace(image, manual):
        seen["background"] = image.convert("L").getpixel((0, 0))
        seen["sherd"] = image.convert("L").getpixel((10, 10))
        return {
            "exterior": [[8, 6], [8, 18]],
            "interior": [[15, 6], [15, 18]],
            "fracture": manual["fracture"],
        }

    monkeypatch.setattr(matcher_annotation, "auto_query_wall_curves_from_fracture", trace)
    monkeypatch.setattr(
        matcher_annotation,
        "preprocess_query",
        lambda *_args, **_kwargs: {"query_id": "saved-query"},
    )
    app = create_annotation_app(
        project,
        queries,
        expected_count=2,
        set_name="real_sherds_2",
        sorted_files=True,
        invert_masks=True,
        preview_before_save=True,
        display_name="real-sherd queries",
    )
    client = app.test_client()
    queue = client.get("/queue").get_json()["queries"]
    payload = {
        "number": 1,
        "fracture": [[8, 18], [12, 19], [15, 18]],
        "rim_point": [12, 6],
    }
    preview = client.post("/preview", json=payload).get_json()
    saved = client.post("/save", json=payload).get_json()

    assert [item["filename"] for item in queue] == ["IMG_4610.png", "IMG_4612.png"]
    assert preview["success"] is True
    assert preview["curves"]["exterior"] == [[8, 6], [8, 18]]
    assert seen == {"background": 255, "sherd": 0}
    assert saved == {"success": True, "query_id": "saved-query"}
