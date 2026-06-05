"""Visualization: experiment figures E1–E5 with colorblind-friendly palettes."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Global plot defaults
sns.set_theme(style="whitegrid", context="paper")
_DPI = 150
_PALETTE = "colorblind"


# ---------------------------------------------------------------------------
# E1: class comparison (simple EN baseline)
# ---------------------------------------------------------------------------

def plot_class_comparison(results_df: pd.DataFrame, output_path: str) -> None:
    """E1: Bar chart of mIoU per class (simple EN baseline)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ordered = results_df.sort_values("mean_mask_iou", ascending=False)
    sns.barplot(
        data=ordered,
        x="class_name",
        y="mean_mask_iou",
        palette=_PALETTE,
        ax=ax,
    )
    ax.set_xlabel("Class")
    ax.set_ylabel("Mean Mask IoU")
    ax.set_title("E1: Per-Class mIoU (Simple EN Baseline)")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(output_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# E2: language heatmap
# ---------------------------------------------------------------------------

def plot_lang_heatmap(results_df: pd.DataFrame, output_path: str) -> None:
    """E2: Heatmap of class × language × mIoU."""
    pivot = results_df.pivot_table(
        index="class_name",
        columns="language",
        values="mean_mask_iou",
    )
    fig, ax = plt.subplots(figsize=(6, 8))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("E2: mIoU by Class × Language")
    ax.set_ylabel("Class")
    ax.set_xlabel("Language")
    plt.tight_layout()
    fig.savefig(output_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# E3: prompt type × language interaction
# ---------------------------------------------------------------------------

def plot_prompt_interaction(results_df: pd.DataFrame, output_path: str) -> None:
    """E3: Grouped bar chart of prompt_type (× language if multilingual)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    multilingual = (
        "language" in results_df.columns
        and results_df["language"].nunique() > 1
    )
    if multilingual:
        sns.barplot(
            data=results_df,
            x="prompt_type",
            y="mean_mask_iou",
            hue="language",
            palette=_PALETTE,
            ax=ax,
        )
        ax.legend(title="Language")
        ax.set_title("E3: Prompt Type × Language Interaction")
    else:
        sns.barplot(
            data=results_df,
            x="prompt_type",
            y="mean_mask_iou",
            palette=_PALETTE,
            ax=ax,
        )
        ax.set_title("E3: Prompt Type Comparison (EN)")
    ax.set_xlabel("Prompt Type")
    ax.set_ylabel("Mean Mask IoU")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    fig.savefig(output_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# E4: box IoU vs mask IoU scatter
# ---------------------------------------------------------------------------

def plot_box_vs_mask_scatter(results_df: pd.DataFrame, output_path: str) -> None:
    """E4: Scatter plot Box IoU (x) vs Mask IoU (y) colored by language."""
    fig, ax = plt.subplots(figsize=(7, 7))
    sns.scatterplot(
        data=results_df,
        x="box_iou",
        y="mask_iou",
        hue="language",
        alpha=0.4,
        s=15,
        palette=_PALETTE,
        ax=ax,
    )
    # Diagonal reference line
    ax.plot([0, 1], [0, 1], ls="--", color="grey", linewidth=0.8, label="y=x")
    ax.set_xlabel("Box IoU")
    ax.set_ylabel("Mask IoU")
    ax.set_title("E4: Box IoU vs Mask IoU")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(title="Language")
    plt.tight_layout()
    fig.savefig(output_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# E5: failure example overlays
# ---------------------------------------------------------------------------

def plot_failure_examples(
    failures: list[dict],
    output_dir: str,
) -> None:
    """E5: Save failure example images with GT and predicted mask overlays.

    Each ``failure`` dict is expected to contain:
    - ``image``: np.ndarray HWC RGB uint8
    - ``gt_mask``: np.ndarray (H, W) bool
    - ``pred_mask``: np.ndarray (H, W) bool (may be None if miss)
    - ``pred_box``: [x1, y1, x2, y2] or None
    - ``failure_type``: str label (e.g. "FP", "miss", "low IoU")
    - ``label``: str description (class, prompt, etc.)
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    n = min(len(failures), 8)
    for i, fail in enumerate(failures[:n]):
        image = fail["image"]
        gt_mask = fail["gt_mask"]
        pred_mask = fail.get("pred_mask")
        pred_box = fail.get("pred_box")
        failure_type = fail.get("failure_type", "unknown")
        label = fail.get("label", "")

        n_cols = 3 if pred_mask is not None else 2
        fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

        # Original image
        axes[0].imshow(image)
        axes[0].set_title("Original")
        axes[0].axis("off")

        # GT overlay
        gt_overlay = image.copy()
        gt_overlay[gt_mask] = (
            gt_overlay[gt_mask] * 0.5 + np.array([0, 255, 0], dtype=np.uint8) * 0.5
        ).astype(np.uint8)
        axes[1].imshow(gt_overlay)
        axes[1].set_title("GT Mask")
        axes[1].axis("off")

        # Pred overlay
        if pred_mask is not None:
            pred_overlay = image.copy()
            pred_overlay[pred_mask] = (
                pred_overlay[pred_mask] * 0.5
                + np.array([255, 0, 0], dtype=np.uint8) * 0.5
            ).astype(np.uint8)
            if pred_box is not None:
                import matplotlib.patches as patches

                x1, y1, x2, y2 = pred_box
                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor="yellow", facecolor="none",
                )
                axes[2].add_patch(rect)
            axes[2].imshow(pred_overlay)
            axes[2].set_title(f"Pred Mask [{failure_type}]")
            axes[2].axis("off")

        fig.suptitle(label, fontsize=10)
        plt.tight_layout()
        fig.savefig(out / f"failure_{i:02d}.png", dpi=_DPI, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# E5: failure taxonomy distribution
# ---------------------------------------------------------------------------

def plot_failure_taxonomy(taxonomy_df: pd.DataFrame, output_path: str) -> None:
    """E5: Horizontal bar chart of failure count per taxonomy category."""
    ordered = taxonomy_df.sort_values("count", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(
        data=ordered,
        x="count",
        y="failure_type",
        orient="h",
        palette=_PALETTE,
        ax=ax,
    )
    ax.set_xlabel("Count")
    ax.set_ylabel("Failure Type")
    ax.set_title("E5: Failure Taxonomy Distribution")
    plt.tight_layout()
    fig.savefig(output_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Generate all figures from CSV tables
# ---------------------------------------------------------------------------

def generate_all_figures(tables_dir: str, figures_dir: str) -> None:
    """Load CSVs from *tables_dir* and generate all experiment figures."""
    tables = Path(tables_dir)
    figures = Path(figures_dir)
    figures.mkdir(parents=True, exist_ok=True)

    # E1
    e1_path = tables / "e1_overall.csv"
    if e1_path.exists():
        e1_df = pd.read_csv(e1_path)
        plot_class_comparison(e1_df, str(figures / "e1_class_comparison.png"))

    # E2
    e2_path = tables / "e2_lang_comparison.csv"
    if e2_path.exists():
        e2_df = pd.read_csv(e2_path)
        plot_lang_heatmap(e2_df, str(figures / "e2_lang_heatmap.png"))

    # E3
    e3_path = tables / "e3_prompt_type.csv"
    if e3_path.exists():
        e3_df = pd.read_csv(e3_path)
        plot_prompt_interaction(e3_df, str(figures / "e3_prompt_interaction.png"))

    # E4 — needs raw instance-level data with box_iou and mask_iou columns
    e4_cond = tables / "e4_conditional.csv"
    e2_lang = tables / "e2_lang_comparison.csv"
    if e4_cond.exists() and e2_lang.exists():
        # Build a scatter-compatible df from the lang comparison
        lang_df = pd.read_csv(e2_lang)
        if {"mean_mask_iou"}.issubset(lang_df.columns):
            # If we have raw results dir alongside tables, load them
            raw_dir = tables.parent / "raw"
            if raw_dir.exists():
                _scatter_from_raw(raw_dir, str(figures / "e4_box_vs_mask.png"))

    # E5 — failure taxonomy
    e5_path = tables / "e5_failure_taxonomy.csv"
    if e5_path.exists():
        e5_df = pd.read_csv(e5_path)
        plot_failure_taxonomy(e5_df, str(figures / "e5_failure_taxonomy.png"))


def _scatter_from_raw(raw_dir: Path, output_path: str) -> None:
    """Build scatter plot from raw per-image JSON files."""
    records: list[dict] = []
    for jf in sorted(raw_dir.glob("*.json")):
        data = json.loads(jf.read_text(encoding="utf-8"))
        for r in data.get("results", []):
            if "box_iou" in r and "mask_iou" in r:
                records.append({
                    "box_iou": r["box_iou"],
                    "mask_iou": r["mask_iou"],
                    "language": r.get("language", "unknown"),
                })
    if records:
        df = pd.DataFrame(records)
        plot_box_vs_mask_scatter(df, output_path)
