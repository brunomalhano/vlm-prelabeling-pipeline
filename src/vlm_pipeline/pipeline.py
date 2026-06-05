"""Orchestrator: chains sampling → prompts → grounding → segmentation → matching → metrics."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm

from .grounding import detect, load_grounding_dino
from .matching import hungarian_match
from .metrics import aggregate_results, compute_instance_metrics
from .prompts import generate_all_prompts
from .sampling import save_sample, stratified_sample
from .segmentation import load_sam2, segment_boxes


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """Load experiment configuration from YAML."""
    with open(config_path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Numpy-safe JSON encoder
# ---------------------------------------------------------------------------

class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that converts numpy scalars and arrays to native Python types."""

    def default(self, o: object) -> object:  # noqa: ANN401
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.bool_):
            return bool(o)
        return super().default(o)


# ---------------------------------------------------------------------------
# GT mask loading
# ---------------------------------------------------------------------------

def _load_gt_masks(
    coco: COCO,
    image_id: int,
    category_id: int,
    img_height: int,
    img_width: int,
) -> list[np.ndarray]:
    """Extract GT binary masks for a specific category in an image.

    Uses ``coco.annToMask()`` for each annotation.
    """
    ann_ids = coco.getAnnIds(imgIds=[image_id], catIds=[category_id], iscrowd=False)
    anns = coco.loadAnns(ann_ids)
    masks: list[np.ndarray] = []
    for ann in anns:
        mask = coco.annToMask(ann)  # (H, W) uint8 {0, 1}
        masks.append(mask.astype(bool))
    return masks


def _load_gt_boxes(coco: COCO, image_id: int, category_id: int) -> list[list[float]]:
    """Extract GT bounding boxes (xyxy) for a specific category in an image."""
    ann_ids = coco.getAnnIds(imgIds=[image_id], catIds=[category_id], iscrowd=False)
    anns = coco.loadAnns(ann_ids)
    boxes: list[list[float]] = []
    for ann in anns:
        x, y, w, h = ann["bbox"]  # COCO xywh
        boxes.append([x, y, x + w, y + h])
    return boxes


# ---------------------------------------------------------------------------
# Result I/O
# ---------------------------------------------------------------------------

def _save_raw_result(result: dict, output_dir: str) -> None:
    """Save per-image result as JSON with numpy-safe encoding."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    image_id = result.get("image_id", "unknown")
    path = out / f"{image_id}.json"
    path.write_text(
        json.dumps(result, cls=_NumpyEncoder, indent=2) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# E5: failure taxonomy classification
# ---------------------------------------------------------------------------

def _classify_failure_type(record: dict) -> str:
    """Classify a non-'good' result into a failure taxonomy category.

    Categories: grounding_miss, box_incomplete, mask_excessive,
    mask_incomplete, false_positive.
    """
    if record.get("is_false_positive"):
        return "false_positive"
    if not record.get("detected", True):
        return "grounding_miss"
    box_iou = record.get("box_iou", 0.0)
    if box_iou < 0.50:
        return "box_incomplete"
    mask_iou = record.get("mask_iou", 0.0)
    if mask_iou < 0.50:
        pred_area = record.get("pred_mask_area", 0)
        gt_area = record.get("gt_mask_area", 1)
        if pred_area > gt_area:
            return "mask_excessive"
        return "mask_incomplete"
    return "mask_incomplete"


# ---------------------------------------------------------------------------
# Bootstrap CI for language deltas
# ---------------------------------------------------------------------------

def _bootstrap_delta_ci(
    en_values: np.ndarray,
    pt_values: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Return (ci_lower, ci_upper) for the delta mean(EN) - mean(PT)."""
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        en_sample = rng.choice(en_values, size=len(en_values), replace=True)
        pt_sample = rng.choice(pt_values, size=len(pt_values), replace=True)
        deltas[i] = en_sample.mean() - pt_sample.mean()
    alpha = (1 - ci) / 2
    return float(np.percentile(deltas, alpha * 100)), float(np.percentile(deltas, (1 - alpha) * 100))


def _holm_correction(p_values: list[float]) -> list[float]:
    """Apply Holm-Bonferroni correction preserving original order."""
    if not p_values:
        return []

    order = np.argsort(p_values)
    m = len(p_values)
    adjusted_sorted = np.zeros(m, dtype=float)

    for rank, idx in enumerate(order):
        adjusted_sorted[rank] = min((m - rank) * p_values[idx], 1.0)

    # Enforce monotonicity in sorted space.
    for i in range(1, m):
        adjusted_sorted[i] = max(adjusted_sorted[i], adjusted_sorted[i - 1])

    adjusted = np.zeros(m, dtype=float)
    for rank, idx in enumerate(order):
        adjusted[idx] = adjusted_sorted[rank]

    return adjusted.tolist()


def _compute_e3_prompt_statistics(gt_en_df, tables: Path) -> None:
    """Export inferential statistics for E3 prompt formulation analysis.

    Writes:
      - e3_prompt_stats.csv
      - e3_prompt_pairwise_dunn.csv
    """
    from itertools import combinations

    import pandas as pd
    from scipy.stats import kruskal, norm, rankdata

    # No GoF pattern applies - this is a direct statistical transform pipeline.
    valid = gt_en_df.dropna(subset=["prompt_type", "mask_iou"]).copy()
    if valid.empty:
        return

    grouped = {
        prompt: grp["mask_iou"].astype(float).values
        for prompt, grp in valid.groupby("prompt_type")
        if len(grp) > 0
    }
    if len(grouped) < 2:
        return

    ordered_prompts = sorted(grouped.keys())
    samples = [grouped[p] for p in ordered_prompts]

    h_stat, p_value = kruskal(*samples, nan_policy="omit")

    n_total = sum(len(arr) for arr in samples)
    k_groups = len(samples)
    epsilon_sq = (
        float((h_stat - k_groups + 1) / (n_total - k_groups))
        if n_total > k_groups
        else 0.0
    )
    epsilon_sq = float(np.clip(epsilon_sq, 0.0, 1.0))

    stats_df = pd.DataFrame(
        [
            {
                "metric": "mask_iou",
                "test": "kruskal_wallis",
                "h_statistic": float(h_stat),
                "p_value": float(p_value),
                "epsilon_squared": epsilon_sq,
                "n_total": int(n_total),
                "n_groups": int(k_groups),
            },
        ],
    )
    stats_df.to_csv(tables / "e3_prompt_stats.csv", index=False)

    ranked = rankdata(valid["mask_iou"].to_numpy(), method="average")
    rank_df = valid.assign(rank=ranked)
    rank_sum = rank_df.groupby("prompt_type")["rank"].sum()
    n_by_group = rank_df.groupby("prompt_type")["rank"].size()

    # Tie correction for Dunn-style z statistics.
    tie_counts = rank_df["mask_iou"].value_counts()
    tie_term = float((tie_counts.pow(3) - tie_counts).sum())
    denom = float(n_total**3 - n_total)
    tie_correction = 1.0 - (tie_term / denom if denom > 0 else 0.0)
    variance = (n_total * (n_total + 1) / 12.0) * tie_correction

    pair_rows: list[dict] = []
    uncorrected: list[float] = []
    for a, b in combinations(ordered_prompts, 2):
        mean_rank_a = rank_sum[a] / n_by_group[a]
        mean_rank_b = rank_sum[b] / n_by_group[b]
        se = np.sqrt(variance * ((1.0 / n_by_group[a]) + (1.0 / n_by_group[b])))

        z_score = float((mean_rank_a - mean_rank_b) / se) if se > 0 else 0.0
        p_unc = float(2 * norm.sf(abs(z_score)))
        pair_rows.append(
            {
                "prompt_a": a,
                "prompt_b": b,
                "z_score": z_score,
                "p_uncorrected": p_unc,
            },
        )
        uncorrected.append(p_unc)

    adjusted = _holm_correction(uncorrected)
    for row, p_adj in zip(pair_rows, adjusted):
        row["p_holm"] = float(p_adj)
        row["significant_005"] = bool(p_adj < 0.05)

    if pair_rows:
        pd.DataFrame(pair_rows).to_csv(tables / "e3_prompt_pairwise_dunn.csv", index=False)


# ---------------------------------------------------------------------------
# Experiment tables (E1–E4)
# ---------------------------------------------------------------------------

def _generate_experiment_tables(all_results: list[dict], tables_dir: str) -> None:
    """Generate CSV tables for experiments E1–E4."""
    import pandas as pd

    tables = Path(tables_dir)
    tables.mkdir(parents=True, exist_ok=True)

    if not all_results:
        return

    df = pd.DataFrame(all_results)

    # ---- E1: overall baseline (simple EN) ----------------------------------
    baseline = df[(df["language"] == "en") & (df["prompt_type"] == "simple")]
    if not baseline.empty:
        e1_overall = (
            baseline.groupby("class_name")
            .agg(
                mean_mask_iou=("mask_iou", "mean"),
                mean_dice=("dice", "mean"),
                mean_boundary_f1=("boundary_f1", "mean"),
                detection_rate=("detected", "mean"),
                count=("mask_iou", "size"),
            )
            .reset_index()
        )
        e1_overall.to_csv(tables / "e1_overall.csv", index=False)

        # Utility distribution
        e1_util = (
            baseline.groupby("class_name")["utility_class"]
            .value_counts(normalize=True)
            .mul(100.0)
            .unstack(fill_value=0.0)
            .reset_index()
        )
        e1_util.to_csv(tables / "e1_utility_distribution.csv", index=False)

    # ---- E2: language comparison (Appendix A — only when PT data present) ---
    if df["language"].nunique() > 1:
        fp_mask_all = (
            df["is_false_positive"].fillna(False).astype(bool)
            if "is_false_positive" in df.columns
            else pd.Series(False, index=df.index)
        )
        gt_all = df[~fp_mask_all]
        e2_lang = (
            gt_all.groupby(["class_name", "language"])
            .agg(
                mean_mask_iou=("mask_iou", "mean"),
                mean_dice=("dice", "mean"),
                mean_boundary_f1=("boundary_f1", "mean"),
                detection_rate=("detected", "mean"),
                count=("mask_iou", "size"),
            )
            .reset_index()
        )
        e2_lang.to_csv(tables / "e2_lang_comparison.csv", index=False)

        # Language delta (EN − PT) per class with bootstrap CI
        pivot = e2_lang.pivot_table(
            index="class_name",
            columns="language",
            values="mean_mask_iou",
        )
        if "en" in pivot.columns and "pt" in pivot.columns:
            delta_rows: list[dict] = []
            for cls in pivot.index:
                en_vals = gt_all[(gt_all["class_name"] == cls) & (gt_all["language"] == "en")]["mask_iou"].values
                pt_vals = gt_all[(gt_all["class_name"] == cls) & (gt_all["language"] == "pt")]["mask_iou"].values
                ci_lo, ci_hi = _bootstrap_delta_ci(en_vals, pt_vals)
                delta_rows.append({
                    "class_name": cls,
                    "en_miou": float(pivot.loc[cls, "en"]),
                    "pt_miou": float(pivot.loc[cls, "pt"]),
                    "delta_en_minus_pt": float(pivot.loc[cls, "en"] - pivot.loc[cls, "pt"]),
                    "ci_lower": ci_lo,
                    "ci_upper": ci_hi,
                })
            pd.DataFrame(delta_rows).to_csv(tables / "e2_lang_delta.csv", index=False)

    # ---- E3: prompt formulation – primary analysis (EN only) ---------------
    en_df = df[df["language"] == "en"].copy()
    fp_mask_en = (
        en_df["is_false_positive"].fillna(False).astype(bool)
        if "is_false_positive" in en_df.columns
        else pd.Series(False, index=en_df.index)
    )
    gt_en = en_df[~fp_mask_en]  # GT-side: matched + misses (no false positives)

    e3_prompt = (
        gt_en.groupby("prompt_type")
        .agg(
            mean_mask_iou=("mask_iou", "mean"),
            mean_dice=("dice", "mean"),
            mean_boundary_f1=("boundary_f1", "mean"),
            detection_rate=("detected", "mean"),
            count=("mask_iou", "size"),
        )
        .reset_index()
    )
    # Conditional mIoU at box_iou ≥ 0.75 (SAM quality given high-quality box)
    if "box_iou" in gt_en.columns:
        cond_iou = (
            gt_en[gt_en["box_iou"] >= 0.75]
            .groupby("prompt_type")["mask_iou"]
            .mean()
            .rename("conditional_miou_box75")
        )
        e3_prompt = e3_prompt.merge(cond_iou, on="prompt_type", how="left")
    e3_prompt.to_csv(tables / "e3_prompt_type.csv", index=False)

    # E3 per-class breakdown (for heatmap figure)
    e3_per_class = (
        gt_en.groupby(["prompt_type", "class_name"])
        .agg(
            mean_mask_iou=("mask_iou", "mean"),
            detection_rate=("detected", "mean"),
            count=("mask_iou", "size"),
        )
        .reset_index()
    )
    e3_per_class.to_csv(tables / "e3_prompt_per_class.csv", index=False)

    _compute_e3_prompt_statistics(gt_en, tables)

    # E3-appendix: prompt × language comparison (written only when PT data present)
    if df["language"].nunique() > 1:
        e3_lang = (
            gt_all.groupby(["prompt_type", "language"])
            .agg(
                mean_mask_iou=("mask_iou", "mean"),
                detection_rate=("detected", "mean"),
                count=("mask_iou", "size"),
            )
            .reset_index()
        )
        e3_lang.to_csv(tables / "e3_prompt_lang_appendix.csv", index=False)

    # ---- E4: conditional analysis ------------------------------------------
    if "box_iou" in df.columns:
        rows: list[dict] = []
        for threshold in (0.50, 0.75):
            subset = df[df["box_iou"] >= threshold]
            if not subset.empty:
                rows.append({
                    "box_iou_threshold": threshold,
                    "n": len(subset),
                    "mean_mask_iou": subset["mask_iou"].mean(),
                    "mean_dice": subset["dice"].mean(),
                    "mean_boundary_f1": subset["boundary_f1"].mean(),
                })
        if rows:
            pd.DataFrame(rows).to_csv(tables / "e4_conditional.csv", index=False)

        # Error source: grounding vs segmentation
        # A detection is a "grounding error" when box_iou < 0.50
        # and a "segmentation error" when box_iou ≥ 0.50 but mask_iou < 0.50
        matched = df[df["detected"] == True]  # noqa: E712
        if not matched.empty:
            total = len(matched)
            grounding_err = (matched["box_iou"] < 0.50).sum()
            seg_err = ((matched["box_iou"] >= 0.50) & (matched["mask_iou"] < 0.50)).sum()
            error_df = pd.DataFrame([{
                "total_matched": total,
                "grounding_errors": int(grounding_err),
                "grounding_error_pct": round(grounding_err / total * 100, 2),
                "segmentation_errors": int(seg_err),
                "segmentation_error_pct": round(seg_err / total * 100, 2),
            }])
            error_df.to_csv(tables / "e4_error_source.csv", index=False)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_experiment(config_path: str = "configs/experiment.yaml") -> None:
    """Full pipeline: sample → prompts → grounding → segmentation → matching → metrics."""
    cfg = load_config(config_path)

    # --- paths --------------------------------------------------------------
    annotations_path: str = cfg["data"]["annotations"]
    images_dir: str = cfg["data"]["images_dir"]
    sample_file: str = cfg["data"]["sample_file"]
    raw_dir: str = cfg["output"]["raw_dir"]
    tables_dir: str = cfg["output"]["tables_dir"]
    figures_dir: str = cfg["output"]["figures_dir"]

    for d in (raw_dir, tables_dir, figures_dir):
        os.makedirs(d, exist_ok=True)

    # --- 1. Sample ----------------------------------------------------------
    if Path(sample_file).exists():
        image_ids: list[int] = json.loads(Path(sample_file).read_text(encoding="utf-8"))
    else:
        image_ids = stratified_sample(
            annotations_path,
            target_classes=[c["id"] for c in cfg["classes"]],
            min_instances=cfg["data"]["min_instances_per_class"],
            sample_size=cfg["data"]["sample_size"],
            seed=cfg["experiment"]["seed"],
        )
        save_sample(image_ids, sample_file)

    # --- 2. Prompts ---------------------------------------------------------
    all_prompts = generate_all_prompts(cfg["classes"])
    languages: list[str] = cfg["prompts"]["languages"]
    prompt_types: list[str] = cfg["prompts"]["types"]

    # --- 3. Load models -----------------------------------------------------
    gd_cfg = cfg["models"]["grounding_dino"]
    gd_model = load_grounding_dino(
        weights_path=gd_cfg["weights"],
        config_path=gd_cfg["config"],
    )

    sam_cfg = cfg["models"]["sam2"]
    sam_predictor = load_sam2(
        weights_path=sam_cfg["weights"],
        model_cfg=sam_cfg["model_cfg"],
    )

    # --- 4. Load COCO -------------------------------------------------------
    coco = COCO(annotations_path)

    # Build class look-ups
    class_id_to_name: dict[int, str] = {c["id"]: c["en"] for c in cfg["classes"]}
    target_cat_ids: list[int] = [c["id"] for c in cfg["classes"]]

    # Matching config
    iou_threshold: float = cfg["matching"]["iou_threshold"]

    # --- 5. Process images --------------------------------------------------
    all_instance_results: list[dict] = []
    failure_examples: list[dict] = []
    n_failures_seen = 0

    for image_id in tqdm(image_ids, desc="Processing images"):
        # Load image
        img_info = coco.loadImgs([image_id])[0]
        img_path = os.path.join(images_dir, img_info["file_name"])
        pil_img = Image.open(img_path).convert("RGB")
        img_np = np.array(pil_img)  # HWC RGB uint8
        img_h, img_w = img_np.shape[:2]

        # GT annotations for this image (all target classes)
        img_anns = coco.getAnnIds(imgIds=[image_id], catIds=target_cat_ids, iscrowd=False)
        present_cats = {ann["category_id"] for ann in coco.loadAnns(img_anns)}

        image_results: list[dict] = []

        for cat_id in target_cat_ids:
            if cat_id not in present_cats:
                continue  # class not present in this image

            class_name = class_id_to_name[cat_id]
            gt_masks = _load_gt_masks(coco, image_id, cat_id, img_h, img_w)
            gt_boxes = _load_gt_boxes(coco, image_id, cat_id)

            if not gt_masks:
                continue

            for lang in languages:
                for ptype in prompt_types:
                    prompt_text = all_prompts[class_name][lang][ptype]

                    # Grounding
                    detections = detect(
                        gd_model,
                        img_np,
                        prompt_text,
                        box_threshold=gd_cfg["box_threshold"],
                        text_threshold=gd_cfg["text_threshold"],
                    )

                    pred_boxes = [d["box"] for d in detections]

                    # Segmentation
                    pred_masks = segment_boxes(sam_predictor, img_np, pred_boxes)

                    # Hungarian matching
                    match_result = hungarian_match(
                        pred_masks, gt_masks, iou_threshold=iou_threshold,
                    )

                    # Per-matched-pair metrics
                    for pred_idx, gt_idx, _ in match_result["matched"]:
                        metrics = compute_instance_metrics(
                            mask_pred=pred_masks[pred_idx],
                            mask_gt=gt_masks[gt_idx],
                            box_pred=pred_boxes[pred_idx],
                            box_gt=gt_boxes[gt_idx],
                        )
                        record = {
                            "image_id": image_id,
                            "class_name": class_name,
                            "category_id": cat_id,
                            "language": lang,
                            "prompt_type": ptype,
                            "prompt_text": prompt_text,
                            "detected": True,
                            "pred_mask_area": int(pred_masks[pred_idx].sum()),
                            "gt_mask_area": int(gt_masks[gt_idx].sum()),
                            **metrics,
                        }
                        image_results.append(record)
                        all_instance_results.append(record)

                        # E5: classify and collect failure examples
                        if record.get("utility_class") != "good":
                            record["failure_type"] = _classify_failure_type(record)
                            n_failures_seen += 1
                            _example = {
                                "image": img_np,
                                "gt_mask": gt_masks[gt_idx],
                                "pred_mask": pred_masks[pred_idx],
                                "pred_box": pred_boxes[pred_idx],
                                "failure_type": record["failure_type"],
                                "label": f"{class_name}/{lang}/{ptype}",
                            }
                            if len(failure_examples) < 8:
                                failure_examples.append(_example)
                            elif random.random() < 8 / n_failures_seen:
                                failure_examples[random.randint(0, 7)] = _example

                    # False positives (no GT match)
                    for fp_idx in match_result["false_positives"]:
                        record = {
                            "image_id": image_id,
                            "class_name": class_name,
                            "category_id": cat_id,
                            "language": lang,
                            "prompt_type": ptype,
                            "prompt_text": prompt_text,
                            "detected": True,
                            "mask_iou": 0.0,
                            "dice": 0.0,
                            "boundary_f1": 0.0,
                            "utility_class": "bad",
                            "box_iou": 0.0,
                            "is_false_positive": True,
                        }
                        record["failure_type"] = _classify_failure_type(record)
                        image_results.append(record)
                        all_instance_results.append(record)

                        # E5: collect failure example
                        n_failures_seen += 1
                        _example = {
                            "image": img_np,
                            "gt_mask": np.zeros((img_h, img_w), dtype=bool),
                            "pred_mask": pred_masks[fp_idx],
                            "pred_box": pred_boxes[fp_idx],
                            "failure_type": record["failure_type"],
                            "label": f"{class_name}/{lang}/{ptype}",
                        }
                        if len(failure_examples) < 8:
                            failure_examples.append(_example)
                        elif random.random() < 8 / n_failures_seen:
                            failure_examples[random.randint(0, 7)] = _example

                    # Misses (GT without prediction)
                    for miss_idx in match_result["misses"]:
                        record = {
                            "image_id": image_id,
                            "class_name": class_name,
                            "category_id": cat_id,
                            "language": lang,
                            "prompt_type": ptype,
                            "prompt_text": prompt_text,
                            "detected": False,
                            "mask_iou": 0.0,
                            "dice": 0.0,
                            "boundary_f1": 0.0,
                            "utility_class": "bad",
                            "box_iou": 0.0,
                            "is_miss": True,
                        }
                        record["failure_type"] = _classify_failure_type(record)
                        image_results.append(record)
                        all_instance_results.append(record)

                        # E5: collect failure example
                        n_failures_seen += 1
                        _example = {
                            "image": img_np,
                            "gt_mask": gt_masks[miss_idx],
                            "pred_mask": None,
                            "pred_box": None,
                            "failure_type": record["failure_type"],
                            "label": f"{class_name}/{lang}/{ptype}",
                        }
                        if len(failure_examples) < 8:
                            failure_examples.append(_example)
                        elif random.random() < 8 / n_failures_seen:
                            failure_examples[random.randint(0, 7)] = _example

        # Save raw per-image result
        _save_raw_result(
            {"image_id": image_id, "results": image_results},
            raw_dir,
        )

    # --- 6. Aggregate tables ------------------------------------------------
    _generate_experiment_tables(all_instance_results, tables_dir)

    # --- 6b. E5 failure taxonomy table --------------------------------------
    import pandas as pd

    failure_records = [r for r in all_instance_results if "failure_type" in r]
    if failure_records:
        ft_series = pd.Series([r["failure_type"] for r in failure_records])
        ft_counts = ft_series.value_counts()
        total = ft_counts.sum()
        taxonomy_df = pd.DataFrame({
            "failure_type": ft_counts.index,
            "count": ft_counts.values,
            "percentage": (ft_counts.values / total * 100).round(1),
        })
        taxonomy_df.to_csv(Path(tables_dir) / "e5_failure_taxonomy.csv", index=False)

    # --- 7. Generate figures ------------------------------------------------
    from .visualization import generate_all_figures, plot_failure_examples

    generate_all_figures(tables_dir, figures_dir)

    if failure_examples:
        plot_failure_examples(failure_examples, str(Path(figures_dir) / "e5_failures"))

    print(
        f"\nExperiment complete. "
        f"{len(all_instance_results)} instance results from {len(image_ids)} images."
    )
