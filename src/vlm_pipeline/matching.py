"""Hungarian matching between predicted and ground-truth masks."""

import numpy as np
from scipy.optimize import linear_sum_assignment

IOU_MATCH_THRESHOLD = 0.10  # Minimum IoU to accept a match


def compute_mask_iou(mask_pred: np.ndarray, mask_gt: np.ndarray) -> float:
    """IoU between two binary masks.

    Returns 0.0 if union is zero.
    """
    intersection = np.logical_and(mask_pred, mask_gt).sum()
    union = np.logical_or(mask_pred, mask_gt).sum()
    return float(intersection / union) if union > 0 else 0.0


def hungarian_match(
    pred_masks: list[np.ndarray],
    gt_masks: list[np.ndarray],
    iou_threshold: float = IOU_MATCH_THRESHOLD,
) -> dict:
    """Match predicted masks to ground-truth masks via the Hungarian algorithm.

    Steps:
        1. Build pairwise IoU matrix (N_pred × N_gt).
        2. Cost matrix = 1 − IoU (minimise cost ⟹ maximise IoU).
        3. Solve optimal assignment with ``linear_sum_assignment``.
        4. Discard matches whose IoU falls below *iou_threshold* (→ FP).
        5. Unmatched GT indices become misses.

    Returns:
        dict with keys:
            matched          – list of ``(pred_idx, gt_idx, iou)``
            false_positives  – list of pred indices without a valid GT match
            misses           – list of GT indices without a prediction
    """
    n_pred = len(pred_masks)
    n_gt = len(gt_masks)

    # --- edge cases ---------------------------------------------------------
    if n_pred == 0 and n_gt == 0:
        return {"matched": [], "false_positives": [], "misses": []}
    if n_pred == 0:
        return {"matched": [], "false_positives": [], "misses": list(range(n_gt))}
    if n_gt == 0:
        return {"matched": [], "false_positives": list(range(n_pred)), "misses": []}

    # --- pairwise IoU matrix ------------------------------------------------
    iou_matrix = np.zeros((n_pred, n_gt), dtype=np.float64)
    for i, mp in enumerate(pred_masks):
        for j, mg in enumerate(gt_masks):
            iou_matrix[i, j] = compute_mask_iou(mp, mg)

    # --- Hungarian assignment (minimise cost = 1 − IoU) ---------------------
    cost_matrix = 1.0 - iou_matrix
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    # --- filter by threshold ------------------------------------------------
    matched: list[tuple[int, int, float]] = []
    matched_pred: set[int] = set()
    matched_gt: set[int] = set()

    for r, c in zip(row_indices, col_indices):
        iou_val = iou_matrix[r, c]
        if iou_val >= iou_threshold:
            matched.append((int(r), int(c), float(iou_val)))
            matched_pred.add(int(r))
            matched_gt.add(int(c))

    false_positives = [i for i in range(n_pred) if i not in matched_pred]
    misses = [j for j in range(n_gt) if j not in matched_gt]

    return {"matched": matched, "false_positives": false_positives, "misses": misses}
