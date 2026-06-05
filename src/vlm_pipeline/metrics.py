"""Metric computation: box IoU, mask quality, boundary quality, and utility classification."""

from __future__ import annotations

import numpy as np
import pandas as pd
from skimage.measure import find_contours


# ---------------------------------------------------------------------------
# Primitive metrics
# ---------------------------------------------------------------------------

def compute_box_iou(box_pred: list[float], box_gt: list[float]) -> float:
    """IoU between two bounding boxes in ``[x1, y1, x2, y2]`` format."""
    x1 = max(box_pred[0], box_gt[0])
    y1 = max(box_pred[1], box_gt[1])
    x2 = min(box_pred[2], box_gt[2])
    y2 = min(box_pred[3], box_gt[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area_pred = (box_pred[2] - box_pred[0]) * (box_pred[3] - box_pred[1])
    area_gt = (box_gt[2] - box_gt[0]) * (box_gt[3] - box_gt[1])
    union = area_pred + area_gt - inter_area

    return float(inter_area / union) if union > 0 else 0.0


def compute_dice(mask_pred: np.ndarray, mask_gt: np.ndarray) -> float:
    """Dice coefficient: ``2|P∩G| / (|P|+|G|)``.

    Returns 0.0 if both masks are empty.
    """
    pred_sum = mask_pred.sum()
    gt_sum = mask_gt.sum()
    if pred_sum + gt_sum == 0:
        return 0.0
    intersection = np.logical_and(mask_pred, mask_gt).sum()
    return float(2.0 * intersection / (pred_sum + gt_sum))


def compute_boundary_f1(
    mask_pred: np.ndarray,
    mask_gt: np.ndarray,
    tolerance: int = 2,
) -> float:
    """Boundary F1 score between two binary masks.

    Steps:
        1. Extract contour pixels from both masks via ``find_contours``.
        2. For each predicted contour pixel, check whether any GT contour
           pixel exists within *tolerance* pixels (Chebyshev distance).
        3. Compute boundary precision, recall, and their harmonic mean (F1).

    Returns 0.0 when either mask has no extractable contours.
    """
    pred_contours = find_contours(mask_pred.astype(np.float64), level=0.5)
    gt_contours = find_contours(mask_gt.astype(np.float64), level=0.5)

    if not pred_contours or not gt_contours:
        return 0.0

    # Collect all contour pixels into (N, 2) arrays
    pred_pts = np.concatenate(pred_contours, axis=0)  # (N_p, 2)
    gt_pts = np.concatenate(gt_contours, axis=0)  # (N_g, 2)

    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return 0.0

    # Precision: fraction of pred boundary pixels matched to a GT pixel
    precision = _matched_fraction(pred_pts, gt_pts, tolerance)
    # Recall: fraction of GT boundary pixels matched to a pred pixel
    recall = _matched_fraction(gt_pts, pred_pts, tolerance)

    if precision + recall == 0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def _matched_fraction(
    source: np.ndarray, target: np.ndarray, tolerance: int
) -> float:
    """Fraction of *source* points that have a *target* neighbour within *tolerance*."""
    # Chebyshev (L∞) distance via broadcasting; chunked to limit memory.
    chunk_size = 2048
    matched = 0
    for start in range(0, len(source), chunk_size):
        src_chunk = source[start : start + chunk_size]  # (C, 2)
        # Distances: (C, N_target)
        diffs = np.abs(src_chunk[:, None, :] - target[None, :, :])  # (C, T, 2)
        chebyshev = diffs.max(axis=2)  # (C, T)
        matched += int((chebyshev.min(axis=1) <= tolerance).sum())
    return matched / len(source)


# ---------------------------------------------------------------------------
# Utility classification
# ---------------------------------------------------------------------------

def classify_utility(mask_iou: float) -> str:
    """Classify a mask match into a utility bucket.

    - ``'good'``        — IoU ≥ 0.75
    - ``'correctable'`` — 0.50 ≤ IoU < 0.75
    - ``'bad'``         — IoU < 0.50
    """
    if mask_iou >= 0.75:
        return "good"
    if mask_iou >= 0.50:
        return "correctable"
    return "bad"


# ---------------------------------------------------------------------------
# Per-instance aggregation
# ---------------------------------------------------------------------------

def compute_instance_metrics(
    mask_pred: np.ndarray,
    mask_gt: np.ndarray,
    box_pred: list[float] | None = None,
    box_gt: list[float] | None = None,
    boundary_tolerance: int = 2,
) -> dict:
    """Compute all metrics for a single matched prediction / ground-truth pair.

    Returns:
        dict with keys ``mask_iou``, ``dice``, ``boundary_f1``,
        ``box_iou`` (present only when boxes are provided),
        and ``utility_class``.
    """
    from .matching import compute_mask_iou

    mask_iou = compute_mask_iou(mask_pred, mask_gt)
    dice = compute_dice(mask_pred, mask_gt)
    bf1 = compute_boundary_f1(mask_pred, mask_gt, tolerance=boundary_tolerance)

    result: dict = {
        "mask_iou": mask_iou,
        "dice": dice,
        "boundary_f1": bf1,
        "utility_class": classify_utility(mask_iou),
    }

    if box_pred is not None and box_gt is not None:
        result["box_iou"] = compute_box_iou(box_pred, box_gt)

    return result


# ---------------------------------------------------------------------------
# Cross-experiment aggregation
# ---------------------------------------------------------------------------

def aggregate_results(all_results: list[dict]) -> pd.DataFrame:
    """Aggregate per-instance result dicts into summary statistics.

    Each element in *all_results* is expected to carry at least:
    ``mask_iou``, ``dice``, ``boundary_f1``, ``utility_class``,
    and grouping keys ``class_name``, ``language``, ``prompt_type``.

    Returns a :class:`~pandas.DataFrame` with one row per group, containing
    mean metrics, detection rate (GSR), utility distribution percentages,
    and instance count.
    """
    if not all_results:
        return pd.DataFrame()

    df = pd.DataFrame(all_results)

    group_cols = ["class_name", "language", "prompt_type"]
    # Ensure grouping columns exist; fill missing with "unknown"
    for col in group_cols:
        if col not in df.columns:
            df[col] = "unknown"

    # Add a combined grouping key
    df["lang_x_type"] = df["language"] + "_" + df["prompt_type"]

    def _agg(group: pd.DataFrame) -> pd.Series:
        n = len(group)
        detected = group["detected"].sum() if "detected" in group.columns else n
        utility_counts = group["utility_class"].value_counts(normalize=True) * 100.0

        metrics: dict = {
            "count": n,
            "mean_mask_iou": group["mask_iou"].mean(),
            "mean_dice": group["dice"].mean(),
            "mean_boundary_f1": group["boundary_f1"].mean(),
            "detection_rate_gsr": detected / n if n > 0 else 0.0,
            "pct_good": utility_counts.get("good", 0.0),
            "pct_correctable": utility_counts.get("correctable", 0.0),
            "pct_bad": utility_counts.get("bad", 0.0),
        }
        if "box_iou" in group.columns:
            metrics["mean_box_iou"] = group["box_iou"].mean()
        return pd.Series(metrics)

    aggregated = df.groupby(group_cols + ["lang_x_type"]).apply(
        _agg, include_groups=False,
    ).reset_index()

    return aggregated
