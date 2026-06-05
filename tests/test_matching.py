"""Tests for vlm_pipeline.matching — IoU computation and Hungarian assignment."""

import numpy as np
import pytest

from vlm_pipeline.matching import compute_mask_iou, hungarian_match


# ---------------------------------------------------------------------------
# compute_mask_iou
# ---------------------------------------------------------------------------

class TestMaskIoU:
    def test_identical_masks(self) -> None:
        """IoU of identical masks is 1.0."""
        mask = np.ones((100, 100), dtype=bool)
        assert compute_mask_iou(mask, mask) == 1.0

    def test_no_overlap(self) -> None:
        """IoU of non-overlapping masks is 0.0."""
        m1 = np.zeros((100, 100), dtype=bool)
        m2 = np.zeros((100, 100), dtype=bool)
        m1[:50, :50] = True
        m2[50:, 50:] = True
        assert compute_mask_iou(m1, m2) == 0.0

    def test_both_empty(self) -> None:
        """IoU of two empty masks is 0.0."""
        m = np.zeros((100, 100), dtype=bool)
        assert compute_mask_iou(m, m) == 0.0

    def test_partial_overlap(self) -> None:
        """IoU of partially overlapping masks is between 0 and 1."""
        m1 = np.zeros((100, 100), dtype=bool)
        m2 = np.zeros((100, 100), dtype=bool)
        m1[:60, :60] = True  # 3600 pixels
        m2[30:80, 30:80] = True  # 2500 pixels
        iou = compute_mask_iou(m1, m2)
        assert 0.0 < iou < 1.0


# ---------------------------------------------------------------------------
# hungarian_match
# ---------------------------------------------------------------------------

class TestHungarianMatch:
    def test_empty_preds(self) -> None:
        """No predictions → all GT are misses."""
        gt = [np.ones((100, 100), dtype=bool)]
        result = hungarian_match([], gt)
        assert len(result["matched"]) == 0
        assert len(result["false_positives"]) == 0
        assert len(result["misses"]) == 1

    def test_empty_gt(self) -> None:
        """No GT → all predictions are false positives."""
        pred = [np.ones((100, 100), dtype=bool)]
        result = hungarian_match(pred, [])
        assert len(result["matched"]) == 0
        assert len(result["false_positives"]) == 1
        assert len(result["misses"]) == 0

    def test_both_empty(self) -> None:
        """Empty pred and empty GT → all lists empty."""
        result = hungarian_match([], [])
        assert result == {"matched": [], "false_positives": [], "misses": []}

    def test_perfect_match(self) -> None:
        """Identical masks produce a perfect match."""
        mask = np.ones((50, 50), dtype=bool)
        result = hungarian_match([mask], [mask])
        assert len(result["matched"]) == 1
        assert result["matched"][0][2] == 1.0  # IoU = 1.0

    def test_below_threshold(self) -> None:
        """Matches with IoU below threshold become FP+miss."""
        m1 = np.zeros((100, 100), dtype=bool)
        m2 = np.zeros((100, 100), dtype=bool)
        m1[:5, :5] = True  # tiny overlap
        m2[90:, 90:] = True
        result = hungarian_match([m1], [m2], iou_threshold=0.10)
        assert len(result["matched"]) == 0
        assert len(result["false_positives"]) == 1
        assert len(result["misses"]) == 1

    def test_multiple_matches(self) -> None:
        """N pred × M GT with correct assignment."""
        # Create 3 distinct regions
        m_gt1 = np.zeros((100, 100), dtype=bool); m_gt1[:30, :30] = True
        m_gt2 = np.zeros((100, 100), dtype=bool); m_gt2[50:80, 50:80] = True
        m_p1 = np.zeros((100, 100), dtype=bool); m_p1[:30, :30] = True  # matches gt1
        m_p2 = np.zeros((100, 100), dtype=bool); m_p2[50:80, 50:80] = True  # matches gt2
        m_p3 = np.zeros((100, 100), dtype=bool); m_p3[90:, 90:] = True  # FP

        result = hungarian_match([m_p1, m_p2, m_p3], [m_gt1, m_gt2])
        assert len(result["matched"]) == 2
        assert len(result["false_positives"]) == 1
        assert len(result["misses"]) == 0
