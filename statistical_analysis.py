"""
Statistical analysis for VLM pipeline experiment.
Computes bootstrap 95% CIs, Wilcoxon signed-rank tests, and Cliff's delta
effect sizes for prompt formulation comparison.

Usage:
    python statistical_analysis.py --raw-dir results/run-20260515T011220Z/raw --output-dir results/tables
"""
import argparse
import json
import pathlib
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# 1. Load raw per-instance data
# ---------------------------------------------------------------------------

def load_raw(raw_dir: pathlib.Path) -> pd.DataFrame:
    records = []
    skipped = 0
    for p in sorted(raw_dir.glob("[!.]*.json")):
        try:
            with open(p) as f:
                data = json.load(f)
            for r in data["results"]:
                records.append(r)
        except (json.JSONDecodeError, KeyError):
            skipped += 1
    if skipped:
        print(f"  (skipped {skipped} unreadable files)")
    df = pd.DataFrame(records)
    # Keep only English prompts (EN-only experiment after pivot)
    df = df[df["language"] == "en"].copy()
    # Exclude false-positive predictions (keep GT instances only)
    if "is_false_positive" in df.columns:
        n_fp = (df["is_false_positive"] == True).sum()
        df = df[df["is_false_positive"] != True].copy()
        print(f"  (excluded {n_fp} false-positive records, keeping GT instances only)")
    # For non-detected GT instances, mask_iou = 0
    df["mask_iou"] = df["mask_iou"].fillna(0.0)
    return df


# ---------------------------------------------------------------------------
# 2. Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_ci(values: np.ndarray, n_boot: int = 10_000, ci: float = 0.95,
                 seed: int = 42) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        means[i] = sample.mean()
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(means, [alpha, 1 - alpha])
    return float(values.mean()), float(lo), float(hi)


def bootstrap_macro_ci(
    df: pd.DataFrame,
    value_col: str,
    class_col: str = "class_name",
    n_boot: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap CI for macro-average with stratified resampling by class."""
    rng = np.random.default_rng(seed)
    class_to_vals = {
        cls: grp[value_col].dropna().to_numpy(dtype=float)
        for cls, grp in df.groupby(class_col)
    }
    class_to_vals = {k: v for k, v in class_to_vals.items() if len(v) > 0}
    if not class_to_vals:
        return 0.0, 0.0, 0.0

    means = np.empty(n_boot)
    classes = sorted(class_to_vals)

    for i in range(n_boot):
        per_class = []
        for cls in classes:
            vals = class_to_vals[cls]
            sample = rng.choice(vals, size=len(vals), replace=True)
            per_class.append(sample.mean())
        means[i] = float(np.mean(per_class))

    observed = float(np.mean([class_to_vals[cls].mean() for cls in classes]))
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(means, [alpha, 1 - alpha])
    return observed, float(lo), float(hi)


def build_instance_balanced_df(
    df: pd.DataFrame,
    prompt_types: list[str],
    class_col: str = "class_name",
    seed: int = 42,
) -> tuple[pd.DataFrame, int]:
    """Create a deterministic class-balanced subset with equal instances per class.

    Balancing is done independently for each prompt_type to avoid leakage across
    prompt comparisons while preserving class parity.
    """
    balanced_parts: list[pd.DataFrame] = []
    min_count_global: int | None = None

    for idx, pt in enumerate(prompt_types):
        sub = df[df["prompt_type"] == pt].copy()
        if sub.empty:
            continue

        counts = sub[class_col].value_counts()
        min_count = int(counts.min())
        if min_count_global is None:
            min_count_global = min_count
        else:
            min_count_global = min(min_count_global, min_count)

        sampled = (
            sub.groupby(class_col, group_keys=False)
            .apply(
                lambda g: g.sample(
                    n=min_count,
                    random_state=seed + idx,
                    replace=False,
                ),
            )
            .reset_index(drop=True)
        )
        balanced_parts.append(sampled)

    if not balanced_parts:
        return pd.DataFrame(columns=df.columns), 0

    out = pd.concat(balanced_parts, ignore_index=True)
    return out, int(min_count_global or 0)


# ---------------------------------------------------------------------------
# 3. Cliff's delta (non-parametric effect size)
# ---------------------------------------------------------------------------

def cliffs_delta(x: np.ndarray, y: np.ndarray) -> tuple[float, str]:
    """Cliff's delta between x and y. Returns (delta, magnitude)."""
    n_x, n_y = len(x), len(y)
    more = np.sum(x[:, None] > y[None, :])
    less = np.sum(x[:, None] < y[None, :])
    delta = (more - less) / (n_x * n_y)
    abs_d = abs(delta)
    if abs_d < 0.147:
        mag = "negligible"
    elif abs_d < 0.33:
        mag = "small"
    elif abs_d < 0.474:
        mag = "medium"
    else:
        mag = "large"
    return float(delta), mag


# ---------------------------------------------------------------------------
# 4. Per-image aggregation for paired tests
# ---------------------------------------------------------------------------

def per_image_miou(df: pd.DataFrame, prompt_type: str) -> pd.Series:
    """Mean mask_iou per image for a given prompt type."""
    sub = df[df["prompt_type"] == prompt_type]
    return sub.groupby("image_id")["mask_iou"].mean()


# ---------------------------------------------------------------------------
# 5. Main analysis
# ---------------------------------------------------------------------------

def run_analysis(raw_dir: pathlib.Path, output_dir: pathlib.Path):
    print("Loading raw data (EN-only)...")
    df = load_raw(raw_dir)
    print(f"  {len(df):,} GT instance records (EN)")

    prompt_types = ["simple", "direct", "contextual", "object"]
    classes = sorted(df["class_name"].unique())

    # ----- A0. Class + instance balanced dataset ----------------------------
    balanced_df, min_per_class = build_instance_balanced_df(df, prompt_types)
    if min_per_class > 0:
        print(
            "\n=== A0. Balanced evaluation dataset ===\n"
            f"  Balanced to {min_per_class} instances/class/prompt_type "
            f"across {len(classes)} classes.\n"
            f"  Balanced records: {len(balanced_df):,}"
        )

        bal_rows = []
        for pt in prompt_types:
            pt_df = balanced_df[balanced_df["prompt_type"] == pt]
            vals = pt_df["mask_iou"].to_numpy(dtype=float)
            mean, lo, hi = bootstrap_ci(vals)
            det_vals = pt_df["detected"].astype(float).to_numpy()
            det_mean, det_lo, det_hi = bootstrap_ci(det_vals)
            bal_rows.append(
                {
                    "prompt_type": pt,
                    "instances_per_class": min_per_class,
                    "n_total": int(len(pt_df)),
                    "miou_balanced": round(mean, 4),
                    "miou_ci_lower": round(lo, 4),
                    "miou_ci_upper": round(hi, 4),
                    "detection_rate_balanced": round(det_mean, 4),
                    "det_ci_lower": round(det_lo, 4),
                    "det_ci_upper": round(det_hi, 4),
                },
            )
            print(
                f"  {pt:12s}: "
                f"mIoU={mean:.4f} [{lo:.4f}, {hi:.4f}] | "
                f"det={det_mean:.4f} [{det_lo:.4f}, {det_hi:.4f}]"
            )

        bal_df = pd.DataFrame(bal_rows)
        bal_df.to_csv(output_dir / "stat_instance_balanced_prompt_type.csv", index=False)

    # ----- A. Bootstrap CIs by prompt type (micro + macro) -----
    print("\n=== A. Bootstrap 95% CI by prompt type (micro + macro) ===")
    ci_rows = []
    for pt in prompt_types:
        pt_df = df[df["prompt_type"] == pt]
        vals = pt_df["mask_iou"].values
        mean_micro, lo_micro, hi_micro = bootstrap_ci(vals)
        mean_macro, lo_macro, hi_macro = bootstrap_macro_ci(pt_df, "mask_iou")
        ci_rows.append({
            "prompt_type": pt,
            "n": len(vals),
            "mean_miou_micro": round(mean_micro, 4),
            "ci_lower_micro": round(lo_micro, 4),
            "ci_upper_micro": round(hi_micro, 4),
            "mean_miou_macro": round(mean_macro, 4),
            "ci_lower_macro": round(lo_macro, 4),
            "ci_upper_macro": round(hi_macro, 4),
        })
        print(
            f"  {pt:12s}: "
            f"micro mIoU = {mean_micro:.4f} [{lo_micro:.4f}, {hi_micro:.4f}] | "
            f"macro mIoU = {mean_macro:.4f} [{lo_macro:.4f}, {hi_macro:.4f}] "
            f"(n={len(vals)})"
        )
    ci_df = pd.DataFrame(ci_rows)
    ci_df.to_csv(output_dir / "stat_bootstrap_prompt_type.csv", index=False)

    # ----- B. Bootstrap CIs by prompt type × class -----
    print("\n=== B. Bootstrap 95% CI by prompt type × class ===")
    ci_class_rows = []
    for pt in prompt_types:
        for cls in classes:
            vals = df[(df["prompt_type"] == pt) & (df["class_name"] == cls)]["mask_iou"].values
            if len(vals) < 5:
                continue
            mean, lo, hi = bootstrap_ci(vals)
            ci_class_rows.append({
                "prompt_type": pt,
                "class_name": cls,
                "n": len(vals),
                "mean_miou": round(mean, 4),
                "ci_lower": round(lo, 4),
                "ci_upper": round(hi, 4),
            })
    ci_class_df = pd.DataFrame(ci_class_rows)
    ci_class_df.to_csv(output_dir / "stat_bootstrap_prompt_class.csv", index=False)
    print(f"  Saved {len(ci_class_df)} rows")

    # ----- C. Wilcoxon signed-rank test (paired by image) -----
    print("\n=== C. Wilcoxon signed-rank test (simple vs. each) ===")
    simple_img = per_image_miou(df, "simple")
    wilcoxon_rows = []
    for pt in ["direct", "contextual", "object"]:
        other_img = per_image_miou(df, pt)
        common = simple_img.index.intersection(other_img.index)
        x = simple_img.loc[common].values
        y = other_img.loc[common].values
        diff = x - y
        # Remove zeros for Wilcoxon
        nonzero = diff[diff != 0]
        if len(nonzero) < 10:
            continue
        stat, p = stats.wilcoxon(nonzero, alternative="greater")
        delta, mag = cliffs_delta(x, y)
        wilcoxon_rows.append({
            "comparison": f"simple > {pt}",
            "n_images": len(common),
            "n_nonzero": len(nonzero),
            "wilcoxon_stat": round(float(stat), 2),
            "p_value": f"{p:.2e}",
            "p_significant_005": p < 0.05,
            "cliffs_delta": round(delta, 4),
            "effect_magnitude": mag,
            "mean_diff": round(float(diff.mean()), 4),
        })
        print(f"  simple > {pt:12s}: W={stat:.0f}, p={p:.2e}, Cliff's δ={delta:.4f} ({mag}), mean Δ={diff.mean():.4f}")
    wilcoxon_df = pd.DataFrame(wilcoxon_rows)
    wilcoxon_df.to_csv(output_dir / "stat_wilcoxon_tests.csv", index=False)

    # ----- D. Detection rate CIs (micro + macro) -----
    print("\n=== D. Detection rate bootstrap CIs (micro + macro) ===")
    det_rows = []
    for pt in prompt_types:
        pt_df = df[df["prompt_type"] == pt].copy()
        vals = pt_df["detected"].astype(float).values
        mean_micro, lo_micro, hi_micro = bootstrap_ci(vals)
        mean_macro, lo_macro, hi_macro = bootstrap_macro_ci(
            pt_df.assign(detected_float=pt_df["detected"].astype(float)),
            "detected_float",
        )
        det_rows.append({
            "prompt_type": pt,
            "n": len(vals),
            "detection_rate_micro": round(mean_micro, 4),
            "ci_lower_micro": round(lo_micro, 4),
            "ci_upper_micro": round(hi_micro, 4),
            "detection_rate_macro": round(mean_macro, 4),
            "ci_lower_macro": round(lo_macro, 4),
            "ci_upper_macro": round(hi_macro, 4),
        })
        print(
            f"  {pt:12s}: "
            f"micro det = {mean_micro:.4f} [{lo_micro:.4f}, {hi_micro:.4f}] | "
            f"macro det = {mean_macro:.4f} [{lo_macro:.4f}, {hi_macro:.4f}]"
        )
    det_df = pd.DataFrame(det_rows)
    det_df.to_csv(output_dir / "stat_bootstrap_detection_rate.csv", index=False)

    # Convenience export focused on class-normalized (macro) metrics
    macro_export = ci_df[[
        "prompt_type",
        "mean_miou_macro",
        "ci_lower_macro",
        "ci_upper_macro",
    ]].merge(
        det_df[[
            "prompt_type",
            "detection_rate_macro",
            "ci_lower_macro",
            "ci_upper_macro",
        ]].rename(
            columns={
                "ci_lower_macro": "det_ci_lower_macro",
                "ci_upper_macro": "det_ci_upper_macro",
            },
        ),
        on="prompt_type",
        how="left",
    )
    macro_export.to_csv(output_dir / "stat_macro_prompt_type.csv", index=False)

    # ----- E. Summary table for paper -----
    print("\n=== E. Paper-ready summary table ===")
    summary_rows = []
    for pt in prompt_types:
        ci_row = ci_df[ci_df["prompt_type"] == pt].iloc[0]
        det_row = det_df[det_df["prompt_type"] == pt].iloc[0]
        row = {
            "prompt_type": pt,
            "miou_micro": f"{ci_row['mean_miou_micro']:.3f}",
            "miou_micro_ci": f"[{ci_row['ci_lower_micro']:.3f}, {ci_row['ci_upper_micro']:.3f}]",
            "miou_macro": f"{ci_row['mean_miou_macro']:.3f}",
            "miou_macro_ci": f"[{ci_row['ci_lower_macro']:.3f}, {ci_row['ci_upper_macro']:.3f}]",
            "detection_rate_micro": f"{det_row['detection_rate_micro']:.3f}",
            "det_micro_ci": f"[{det_row['ci_lower_micro']:.3f}, {det_row['ci_upper_micro']:.3f}]",
            "detection_rate_macro": f"{det_row['detection_rate_macro']:.3f}",
            "det_macro_ci": f"[{det_row['ci_lower_macro']:.3f}, {det_row['ci_upper_macro']:.3f}]",
        }
        if pt != "simple":
            w_row = wilcoxon_df[wilcoxon_df["comparison"] == f"simple > {pt}"].iloc[0]
            row["p_value"] = w_row["p_value"]
            row["cliffs_delta"] = w_row["cliffs_delta"]
            row["effect"] = w_row["effect_magnitude"]
        else:
            row["p_value"] = "—"
            row["cliffs_delta"] = "—"
            row["effect"] = "—"
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "stat_paper_summary.csv", index=False)
    print(summary_df.to_string(index=False))

    print(f"\n✓ All outputs saved to {output_dir}/stat_*.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=pathlib.Path,
                        default=pathlib.Path("results/run-20260515T011220Z/raw"))
    parser.add_argument("--output-dir", type=pathlib.Path,
                        default=pathlib.Path("results/tables"))
    args = parser.parse_args()
    run_analysis(args.raw_dir, args.output_dir)
