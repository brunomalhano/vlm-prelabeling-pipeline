# %% [markdown]
# # 01 — Data Exploration
#
# Explore the COCO val2017 dataset and validate the stratified sample used in the
# VLM pre-labeling experiments.
#
# **Paper sections**: §3 Methodology — Dataset & Sampling, §4.1 Dataset Description.
#
# This notebook is self-contained: run all cells top-to-bottom to reproduce the
# exploration artifacts.

# %%
import json
import random
from collections import Counter
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from pycocotools.coco import COCO

from vlm_pipeline.prompts import generate_all_prompts, CLASS_MAP, PROMPT_TEMPLATES
from vlm_pipeline.sampling import (
    TARGET_CLASSES,
    print_coverage_report,
    save_sample,
    stratified_sample,
)

%matplotlib inline
plt.style.use("seaborn-v0_8-whitegrid")

ANNOTATIONS_PATH = "../data/coco/annotations/instances_val2017.json"
IMAGES_DIR = Path("../data/coco/val2017")
SAMPLE_PATH = Path("../data/sample/sample_500.json")

# %% [markdown]
# ## Load COCO annotations

# %%
coco = COCO(ANNOTATIONS_PATH)

num_images = len(coco.getImgIds())
num_categories = len(coco.getCatIds())
num_annotations = len(coco.getAnnIds())

print(f"Images:       {num_images:,}")
print(f"Categories:   {num_categories}")
print(f"Annotations:  {num_annotations:,}")

# %% [markdown]
# ## Target classes overview
#
# The experiment uses 10 COCO categories selected to span a range of visual
# complexity (persons, animals, vehicles, furniture, food).

# %%
target_cats = coco.loadCats(TARGET_CLASSES)
rows = []
for cat in target_cats:
    ann_ids = coco.getAnnIds(catIds=[cat["id"]])
    img_ids = coco.getImgIds(catIds=[cat["id"]])
    rows.append({
        "COCO ID": cat["id"],
        "Category": cat["name"],
        "Instances": len(ann_ids),
        "Images": len(img_ids),
    })

target_df = pd.DataFrame(rows)
target_df = target_df.sort_values("Instances", ascending=False).reset_index(drop=True)
display(target_df)

# %% [markdown]
# ## Generate stratified sample
#
# Either load the existing sample or generate a new one with seed=42.
# The sampling guarantees ≥30 instances per target class and prioritizes
# multi-class images to maximize coverage within 500 images.

# %%
if SAMPLE_PATH.exists():
    sample_ids = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
    print(f"Loaded existing sample: {len(sample_ids)} images from {SAMPLE_PATH}")
else:
    sample_ids = stratified_sample(ANNOTATIONS_PATH)
    save_sample(sample_ids, str(SAMPLE_PATH))
    print(f"Generated and saved new sample: {len(sample_ids)} images")

print()
print_coverage_report(coco, sample_ids)

# %% [markdown]
# ## Class distribution comparison
#
# Side-by-side bar chart comparing instance counts in the full dataset versus the
# stratified sample for each target class.

# %%
sample_set = set(sample_ids)
cats_map = {c["id"]: c["name"] for c in coco.loadCats(TARGET_CLASSES)}

full_counts = []
sample_counts = []
class_names = []

for cat_id in TARGET_CLASSES:
    name = cats_map[cat_id]
    class_names.append(name)
    full_counts.append(len(coco.getAnnIds(catIds=[cat_id])))
    sample_counts.append(len(coco.getAnnIds(imgIds=list(sample_set), catIds=[cat_id])))

x = np.arange(len(class_names))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 5))
bars1 = ax.bar(x - width / 2, full_counts, width, label="Full val2017", color="#4c72b0")
bars2 = ax.bar(x + width / 2, sample_counts, width, label="Sample (500)", color="#dd8452")

ax.set_xlabel("Class")
ax.set_ylabel("Instance Count")
ax.set_title("Instance Count per Class: Full Dataset vs. Stratified Sample")
ax.set_xticks(x)
ax.set_xticklabels(class_names, rotation=45, ha="right")
ax.legend()
ax.set_yscale("log")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Sample images preview
#
# Display 6 random sample images (2×3 grid) with ground-truth bounding boxes and
# category labels for the 10 target classes overlaid.

# %%
rng = random.Random(42)
preview_ids = rng.sample(sample_ids, min(6, len(sample_ids)))

# Distinct colors for target classes
cmap = plt.cm.get_cmap("tab10", len(TARGET_CLASSES))
color_map = {cat_id: cmap(i) for i, cat_id in enumerate(TARGET_CLASSES)}

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for idx, img_id in enumerate(preview_ids):
    ax = axes[idx]
    img_info = coco.loadImgs([img_id])[0]
    img_path = IMAGES_DIR / img_info["file_name"]

    if img_path.exists():
        img = Image.open(img_path).convert("RGB")
        ax.imshow(img)
    else:
        ax.text(0.5, 0.5, f"Image not found:\n{img_info['file_name']}",
                ha="center", va="center", transform=ax.transAxes, fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    # Overlay bounding boxes for target classes
    ann_ids = coco.getAnnIds(imgIds=[img_id], catIds=TARGET_CLASSES, iscrowd=False)
    anns = coco.loadAnns(ann_ids)
    for ann in anns:
        x, y, w, h = ann["bbox"]
        cat_id = ann["category_id"]
        color = color_map.get(cat_id, "red")
        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=2, edgecolor=color, facecolor="none",
        )
        ax.add_patch(rect)
        cat_name = cats_map.get(cat_id, str(cat_id))
        ax.text(x, y - 4, cat_name, fontsize=7, color="white",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.8))

    ax.set_title(f"ID: {img_id}", fontsize=9)
    ax.axis("off")

plt.suptitle("Sample Images with Ground-Truth Annotations", fontsize=13)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Prompt preview
#
# All 80 prompts (10 classes × 4 types × 2 languages) generated by the prompt
# engine. These are the exact text prompts fed to Grounding DINO.

# %%
all_prompts = generate_all_prompts()
rows = []
for class_en, langs in all_prompts.items():
    for lang, types in langs.items():
        for ptype, text in types.items():
            rows.append({
                "Class": class_en,
                "Language": lang.upper(),
                "Prompt Type": ptype,
                "Prompt Text": text,
            })

prompt_df = pd.DataFrame(rows)
print(f"Total prompts: {len(prompt_df)}")
print(f"  Classes: {prompt_df['Class'].nunique()}")
print(f"  Types:   {prompt_df['Prompt Type'].nunique()}")
print(f"  Languages: {prompt_df['Language'].nunique()}")
print()
display(prompt_df)
