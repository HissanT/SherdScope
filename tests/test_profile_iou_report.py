from pathlib import Path

import numpy as np
from PIL import Image

from scripts.evaluation.build_profile_iou_report import evaluate_pair


def save_mask(path: Path, values: np.ndarray) -> None:
    Image.fromarray(values.astype(np.uint8) * 255, mode="L").save(path)


def test_profile_iou_uses_binary_intersection_over_union(tmp_path):
    automatic = np.array([[1, 1], [0, 0]], dtype=bool)
    accepted = np.array([[0, 1], [1, 0]], dtype=bool)
    auto_path, accepted_path = tmp_path / "auto.png", tmp_path / "accepted.png"
    save_mask(auto_path, automatic)
    save_mask(accepted_path, accepted)

    result = evaluate_pair(auto_path, accepted_path)

    assert result["valid"] is True
    assert result["intersection_pixels"] == 1
    assert result["union_pixels"] == 3
    assert result["iou"] == 1 / 3
    assert result["dice"] == 1 / 2


def test_profile_iou_reports_dimension_mismatch_without_resampling(tmp_path):
    auto_path, accepted_path = tmp_path / "auto.png", tmp_path / "accepted.png"
    save_mask(auto_path, np.ones((2, 2), dtype=bool))
    save_mask(accepted_path, np.ones((3, 2), dtype=bool))

    result = evaluate_pair(auto_path, accepted_path)

    assert result == {
        "valid": False,
        "reason": "dimension_mismatch",
        "auto_shape": [2, 2],
        "accepted_shape": [3, 2],
    }
