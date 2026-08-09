import numpy as np

from real_sherd_pilot.benchmark import fill_enclosed_holes, standardized_metrics, unet_prompt


def test_hole_fill_does_not_move_outer_contour():
    mask = np.zeros((30, 40), dtype=bool)
    mask[5:25, 8:32] = True
    mask[12:18, 16:24] = False
    fixed = fill_enclosed_holes(mask)
    assert fixed[12:18, 16:24].all()
    assert np.array_equal(fixed[:5], mask[:5])
    assert np.array_equal(fixed[:, :8], mask[:, :8])


def test_hole_fill_preserves_open_concavity():
    mask = np.zeros((30, 40), dtype=bool)
    mask[5:25, 8:32] = True
    mask[12:18, 20:40] = False
    assert np.array_equal(fill_enclosed_holes(mask), mask)


def test_unet_prompt_uses_mask_geometry_only():
    mask = np.zeros((20, 30), dtype=bool)
    mask[4:16, 7:24] = True
    box, point = unet_prompt(mask)
    assert box == [7, 4, 24, 16]
    assert mask[point[1], point[0]]


def test_standardized_metrics_are_perfect_for_equal_masks():
    mask = np.zeros((80, 120), dtype=bool)
    mask[12:70, 25:90] = True
    metrics = standardized_metrics(mask, mask)
    assert metrics["dice"] == 1.0
    assert metrics["boundary_f1_at_3px"] == 1.0
