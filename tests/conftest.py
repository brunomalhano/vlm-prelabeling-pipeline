"""Shared fixtures for vlm_pipeline tests."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture()
def full_mask() -> np.ndarray:
    """100×100 boolean mask with all pixels True."""
    return np.ones((100, 100), dtype=bool)


@pytest.fixture()
def empty_mask() -> np.ndarray:
    """100×100 boolean mask with all pixels False."""
    return np.zeros((100, 100), dtype=bool)


@pytest.fixture()
def square_mask() -> np.ndarray:
    """100×100 mask with a centered 60×60 True region (rows/cols 20–79)."""
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:80, 20:80] = True
    return mask
