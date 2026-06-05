# %% [markdown]
# # 02 — Run Experiments
#
# Execute the full VLM pre-labeling pipeline: Grounding DINO + SAM 2.1 across all
# prompt × language combinations.
#
# **⏱ Estimated runtime**: 2–4 hours on a single GPU (500 images × 80 prompts).
# Ensure GPU is available before running the experiment cell.
#
# **Paper sections**: §3 Methodology — Pipeline Architecture, §4 Experiments.

# %%
import json
from pathlib import Path

import pandas as pd
import yaml

from vlm_pipeline.pipeline import load_config, run_experiment

%matplotlib inline

CONFIG_PATH = "../configs/experiment.yaml"

# %% [markdown]
# ## Configuration overview
#
# Display the full experiment configuration used for this run.

# %%
config = load_config(CONFIG_PATH)
print(yaml.dump(config, default_flow_style=False, sort_keys=False, allow_unicode=True))

# %% [markdown]
# ## Validate prerequisites
#
# Before launching the experiment, verify that all required files and hardware are
# in place. All checks must pass (✅) before proceeding.

# %%
import torch

checks = {
    "Sample file": Path("../data/sample/sample_500.json").exists(),
    "COCO annotations": Path("../data/coco/annotations/instances_val2017.json").exists(),
    "COCO images dir": Path("../data/coco/val2017").is_dir(),
    "GDINO weights": Path("../weights/groundingdino_swinb_cogcoor.pth").exists(),
    "SAM 2.1 weights": Path("../weights/sam2.1_hiera_large.pt").exists(),
    "CUDA available": torch.cuda.is_available(),
}

all_pass = True
for name, ok in checks.items():
    status = "✅" if ok else "❌"
    print(f"  {status}  {name}")
    if not ok:
        all_pass = False

print()
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
else:
    print("  ⚠️  No GPU detected — experiment will be extremely slow on CPU.")

if not all_pass:
    print("\n  ⛔ Some prerequisites are missing. Fix them before running the experiment.")
else:
    print("\n  ✅ All prerequisites satisfied — ready to run.")

# %% [markdown]
# ## Run experiment
#
# This cell launches the full pipeline. Progress is reported via `tqdm`.
# Raw results are saved per-image as JSON files under `results/raw/`.
# Aggregated tables are written to `results/tables/`.

# %%
run_experiment(CONFIG_PATH)

# %% [markdown]
# ## Quick inspection
#
# Load a few raw result files and inspect their structure and key metrics.

# %%
raw_dir = Path("../results/raw")
if raw_dir.exists():
    result_files = sorted(raw_dir.glob("*.json"))
    print(f"Total raw result files: {len(result_files)}")
    print()

    for rf in result_files[:3]:
        data = json.loads(rf.read_text(encoding="utf-8"))
        image_id = data.get("image_id", rf.stem)
        num_results = len(data.get("results", []))
        print(f"── {rf.name} ──")
        print(f"   Image ID:  {image_id}")
        print(f"   Results:   {num_results} entries")
        if num_results > 0:
            first = data["results"][0]
            keys = list(first.keys())
            print(f"   Keys:      {keys}")
            if "mask_iou" in first:
                print(f"   First entry mask_iou: {first['mask_iou']:.4f}")
        print()
else:
    print("⚠️  results/raw/ not found — run the experiment first.")

# %% [markdown]
# ## Summary statistics
#
# Quick count of total results, broken down by language and prompt type.

# %%
tables_dir = Path("../results/tables")
if tables_dir.exists():
    csv_files = sorted(tables_dir.glob("*.csv"))
    print(f"Generated table files: {len(csv_files)}")
    for f in csv_files:
        df = pd.read_csv(f)
        print(f"  {f.name:<35} {len(df):>5} rows × {len(df.columns)} cols")
    print()

    # Show breakdown from E3 (has language + prompt_type)
    e3_path = tables_dir / "e3_prompt_type.csv"
    if e3_path.exists():
        e3 = pd.read_csv(e3_path)
        if "language" in e3.columns and "prompt_type" in e3.columns:
            print("Results by language:")
            for lang, group in e3.groupby("language"):
                total = int(group["count"].sum()) if "count" in group.columns else len(group)
                print(f"  {lang}: {total:,} entries")
            print()
            print("Results by prompt type:")
            for pt, group in e3.groupby("prompt_type"):
                total = int(group["count"].sum()) if "count" in group.columns else len(group)
                print(f"  {pt}: {total:,} entries")
else:
    print("⚠️  results/tables/ not found — run the experiment first.")
