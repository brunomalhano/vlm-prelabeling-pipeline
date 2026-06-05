"""Tests for vlm_pipeline.metrics — box IoU, Dice, boundary F1, utility, instance metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vlm_pipeline.metrics import (
    aggregate_results,
    classify_utility,
    compute_boundary_f1,
    compute_box_iou,
    compute_dice,
    compute_instance_metrics,
)


# ---------------------------------------------------------------------------
# compute_box_iou
# ---------------------------------------------------------------------------

class TestBoxIoU:
    def test_identical_boxes(self) -> None:
        assert compute_box_iou([0, 0, 100, 100], [0, 0, 100, 100]) == 1.0

    def test_no_overlap_boxes(self) -> None:
        assert compute_box_iou([0, 0, 50, 50], [60, 60, 100, 100]) == 0.0

    def test_partial_overlap(self) -> None:
        iou = compute_box_iou([0, 0, 100, 100], [50, 50, 150, 150])
        assert 0.0 < iou < 1.0

    def test_zero_area(self) -> None:
        assert compute_box_iou([0, 0, 0, 0], [0, 0, 0, 0]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compute_dice
# ---------------------------------------------------------------------------

class TestDice:
    def test_identical_masks(self) -> None:
        mask = np.ones((100, 100), dtype=bool)
        assert compute_dice(mask, mask) == 1.0

    def test_both_empty(self) -> None:
        mask = np.zeros((100, 100), dtype=bool)
        assert compute_dice(mask, mask) == 0.0

    def test_no_overlap(self) -> None:
        m1 = np.zeros((100, 100), dtype=bool); m1[:50] = True
        m2 = np.zeros((100, 100), dtype=bool); m2[50:] = True
        assert compute_dice(m1, m2) == 0.0


# ---------------------------------------------------------------------------
# compute_boundary_f1
# ---------------------------------------------------------------------------

class TestBoundaryF1:
    def test_identical_masks(self) -> None:
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:80, 20:80] = True
        f1 = compute_boundary_f1(mask, mask, tolerance=2)
        assert f1 >= 0.95  # Should be very close to 1.0

    def test_both_empty(self) -> None:
        mask = np.zeros((100, 100), dtype=bool)
        assert compute_boundary_f1(mask, mask) == 0.0

    def test_empty_pred(self) -> None:
        empty = np.zeros((20, 20), dtype=bool)
        gt = np.zeros((20, 20), dtype=bool)
        gt[5:15, 5:15] = True
        assert compute_boundary_f1(empty, gt) == pytest.approx(0.0)

    def test_empty_gt(self) -> None:
        pred = np.zeros((20, 20), dtype=bool)
        pred[5:15, 5:15] = True
        empty = np.zeros((20, 20), dtype=bool)
        assert compute_boundary_f1(pred, empty) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# classify_utility
# ---------------------------------------------------------------------------

class TestClassifyUtility:
    def test_good(self) -> None:
        assert classify_utility(0.80) == "good"
        assert classify_utility(0.75) == "good"

    def test_correctable(self) -> None:
        assert classify_utility(0.60) == "correctable"
        assert classify_utility(0.50) == "correctable"

    def test_bad(self) -> None:
        assert classify_utility(0.30) == "bad"
        assert classify_utility(0.0) == "bad"

    def test_boundary_values(self) -> None:
        assert classify_utility(0.75) == "good"
        assert classify_utility(0.749) == "correctable"
        assert classify_utility(0.50) == "correctable"
        assert classify_utility(0.499) == "bad"


# ---------------------------------------------------------------------------
# compute_instance_metrics
# ---------------------------------------------------------------------------

class TestInstanceMetrics:
    def test_perfect_match(self) -> None:
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:80, 20:80] = True
        result = compute_instance_metrics(mask, mask)
        assert result["mask_iou"] == 1.0
        assert result["dice"] == 1.0
        assert result["utility_class"] == "good"

    def test_with_boxes(self) -> None:
        mask = np.zeros((50, 50), dtype=bool)
        mask[10:40, 10:40] = True
        result = compute_instance_metrics(
            mask, mask,
            box_pred=[10, 10, 40, 40],
            box_gt=[10, 10, 40, 40],
        )
        assert result["mask_iou"] == pytest.approx(1.0)
        assert result["dice"] == pytest.approx(1.0)
        assert result["boundary_f1"] == pytest.approx(1.0)
        assert result["box_iou"] == pytest.approx(1.0)
        assert result["utility_class"] == "good"

    def test_no_boxes(self) -> None:
        mask = np.ones((10, 10), dtype=bool)
        result = compute_instance_metrics(mask, mask)
        assert "box_iou" not in result
        assert "mask_iou" in result


# ---------------------------------------------------------------------------
# aggregate_results
# ---------------------------------------------------------------------------

class TestAggregateResults:
    def test_empty(self) -> None:
        df = aggregate_results([])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_basic_aggregation(self) -> None:
        results = [
            {
                "class_name": "cat",
                "language": "en",
                "prompt_type": "simple",
                "mask_iou": 0.80,
                "dice": 0.85,
                "boundary_f1": 0.70,
                "utility_class": "good",
            },
            {
                "class_name": "cat",
                "language": "en",
                "prompt_type": "simple",
                "mask_iou": 0.60,
                "dice": 0.65,
                "boundary_f1": 0.55,
                "utility_class": "correctable",
            },
        ]
        df = aggregate_results(results)
        assert len(df) == 1
        assert df["count"].iloc[0] == 2
        assert df["mean_mask_iou"].iloc[0] == pytest.approx(0.70)

    def test_groups_by_language_and_type(self) -> None:
        results = [
            {
                "class_name": "dog",
                "language": "en",
                "prompt_type": "direct",
                "mask_iou": 0.90,
                "dice": 0.92,
                "boundary_f1": 0.88,
                "utility_class": "good",
            },
            {
                "class_name": "dog",
                "language": "pt",
                "prompt_type": "direct",
                "mask_iou": 0.40,
                "dice": 0.45,
                "boundary_f1": 0.35,
                "utility_class": "bad",
            },
        ]
        df = aggregate_results(results)
        assert len(df) == 2  # two distinct language groups
