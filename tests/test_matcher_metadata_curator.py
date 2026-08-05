import json
from pathlib import Path

from PIL import Image

from catalog.matcher_metadata_curator import create_metadata_curator_app


def test_metadata_curator_loads_points_and_saves_optional_values(tmp_path: Path):
    project = tmp_path / "project"
    images = tmp_path / "queries"
    images.mkdir()
    prepared = {}
    for number in range(1, 31):
        filename = f"Query{number}.png"
        Image.new("RGB", (24, 24), "white").save(images / filename)
        prepared[str(number)] = {
            "number": number,
            "filename": filename,
            "query_id": f"query-{number}",
            "fracture": [[1, 1], [2, 2], [3, 3]],
            "rim_point": [4, 5],
        }
    set_path = project / "matcher" / "query_sets" / "hesban_30.json"
    set_path.parent.mkdir(parents=True)
    set_path.write_text(
        json.dumps({"schema_version": 1, "queries": prepared}),
        encoding="utf-8",
    )
    cards = project / "cards"
    cards.mkdir()
    (cards / "mask_info.csv").write_text(
        "mask_file,Figure,No.,Rim Diameter (cm),Fabric Color - Core\n"
        "reference.png,3.1,8,12.6,5YR 5/4\n",
        encoding="utf-8",
    )

    client = create_metadata_curator_app(project, images).test_client()
    queue = client.get("/queue").get_json()["queries"]
    assert len(queue) == 30
    assert queue[0]["fracture"] == [[1, 1], [2, 2], [3, 3]]
    assert queue[0]["rim_point"] == [4, 5]
    assert queue[0]["target"] == {"figure": "3.1", "item": "8"}
    assert queue[0]["reference_records"] == [{
        "mask_file": "reference.png",
        "values": [
            {"key": "rim_diameter_cm", "label": "Rim diameter (cm)", "value": "12.6"},
            {"key": "fabric_core", "label": "Fabric colour - core (Munsell)", "value": "5YR 5/4"},
        ],
    }]

    response = client.post(
        "/save",
        json={
            "number": 1,
            "metadata": {
                "rim_diameter_cm": "12.4",
                "fabric_core": "5YR 5/4",
                "not_allowed": "discard me",
            },
            "diameter_uncertainty_cm": 1.8,
        },
    )
    assert response.status_code == 200
    reloaded = client.get("/queue").get_json()["queries"][0]
    assert reloaded["saved"] is True
    assert reloaded["metadata"] == {
        "rim_diameter_cm": "12.4",
        "fabric_core": "5YR 5/4",
    }
    assert reloaded["diameter_uncertainty_cm"] == 1.8
