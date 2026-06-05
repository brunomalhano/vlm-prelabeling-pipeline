# VLM Pre-Labeling Pipeline

Reproducible pipeline for evaluating prompt formulation effects in modular pre-labeling for instance segmentation using Grounding DINO + SAM 2.1 on COCO val2017.

This repository is aligned with the final manuscript:

- **Prompt Formulation Effects in Modular Vision-Language Pre-labeling Pipelines for Instance Segmentation**
- Authors: Bruno Malhano de Oliveira Jordao, Giovane Quadrelli
- Public repo URL: https://github.com/brunomalhano/vlm-prelabeling-pipeline

## Scope

The default configuration reproduces the **primary EN-only experiment** used in the paper:

- 500 sampled COCO val2017 images
- 10 target classes
- 4 prompt formulations: `simple`, `direct`, `contextual`, `object`
- Matching: Hungarian assignment per class per image
- Models: Grounding DINO (Swin-B) + SAM 2.1 (Hiera-L)

Portuguese (`pt`) can be enabled as an additional analysis mode by editing `configs/experiment.yaml` (`prompts.languages`).

## Repository Structure

```text
.
├── configs/
│   └── experiment.yaml
├── data/
│   └── sample/
├── notebooks/
├── results/
│   ├── figures/
│   └── tables/
├── scripts/
│   ├── prepare_public_release.sh
│   └── setup_data.sh
├── src/
│   └── vlm_pipeline/
│       ├── cli.py
│       ├── grounding.py
│       ├── matching.py
│       ├── metrics.py
│       ├── pipeline.py
│       ├── prompts.py
│       ├── sampling.py
│       ├── segmentation.py
│       └── visualization.py
├── tests/
├── dashboard.py
├── statistical_analysis.py
├── requirements.txt
└── setup.py
```

## Environment Requirements

- Python 3.10+
- CUDA-compatible GPU (the manuscript run used NVIDIA T4 16 GB)
- Linux/macOS shell environment with `curl` and `unzip`

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Data and Weights Setup

```bash
bash scripts/setup_data.sh
```

The setup script prepares:

- `data/coco/val2017` and `data/coco/annotations/instances_val2017.json`
- `weights/groundingdino_swinb_cogcoor.pth`
- `weights/sam2.1_hiera_large.pt`
- output folders under `results/`

## CLI Usage

### 1. Generate stratified sample

```bash
vlm-pipeline sample --config configs/experiment.yaml
```

### 2. Validate existing sample

```bash
vlm-pipeline sample --config configs/experiment.yaml --validate
```

### 3. Run full experiment

```bash
vlm-pipeline run --config configs/experiment.yaml
```

### 4. Regenerate figures from tables

```bash
vlm-pipeline visualize --tables-dir results/tables --figures-dir results/figures
```

## Key Outputs

Core artifacts produced by the pipeline include:

- Paper summary metrics: `results/tables/stat_paper_summary.csv`
- Paired inferential tests: `results/tables/stat_wilcoxon_tests.csv`
- Bootstrap CIs: `results/tables/stat_bootstrap_*.csv`
- Balanced analysis: `results/tables/stat_instance_balanced_prompt_type.csv`
- Main figures: `results/figures/e1_*.png` to `results/figures/e5_*.png`

## Reference Results (from current artifacts)

From `results/tables/stat_paper_summary.csv`:

- `simple`: mIoU micro `0.527` (95% CI `[0.515, 0.539]`)
- `direct`: mIoU micro `0.210`
- `contextual`: mIoU micro `0.250`
- `object`: mIoU micro `0.204`

This matches the manuscript conclusion that **simple class-name prompts outperform longer formulations** in this pipeline setup.

## Reproducibility Notes

- Keep `seed: 42` in `configs/experiment.yaml` for direct comparability.
- Keep model thresholds as configured unless running sensitivity analyses.
- The pipeline supports both natural-distribution and class-balanced reporting via generated statistics tables.

## Public Release Workflow

Use `PUBLIC_REPO_PREP.md` for release governance and manuscript declaration linkage.
