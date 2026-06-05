# VLM Pre-Labeling Pipeline

Pipeline for evaluating PT/EN prompts for pre-labeling with Grounding DINO + SAM 2.1 on COCO val2017.

## Overview

This pipeline evaluates how language (English vs Portuguese) and prompt formulation affect the quality of Vision-Language Model (VLM) based pre-labeling for instance segmentation tasks.

### Architecture

```
vlm-pipeline/
├── configs/
│   └── experiment.yaml          # Experiment configuration
├── data/
│   └── coco/
│       ├── annotations/         # COCO val2017 annotations
│       └── val2017/             # COCO val2017 images
├── notebooks/
│   ├── 01_data_exploration.py   # Dataset exploration & sample validation
│   ├── 02_run_experiments.py    # Execute the full pipeline
│   └── 03_analysis.py          # Results analysis & figure generation
├── results/
│   ├── figures/                 # Publication-ready figures
│   ├── raw/                     # Raw inference outputs
│   └── tables/                  # Aggregated CSV tables
├── scripts/
│   └── setup_data.sh            # Download data and model weights
├── src/
│   └── vlm_pipeline/
│       ├── __init__.py
│       ├── cli.py               # CLI entry point
│       ├── grounding.py         # Grounding DINO inference
│       ├── matching.py          # Prediction-GT matching (IoU)
│       ├── metrics.py           # mAP, IoU, boundary metrics
│       ├── pipeline.py          # Orchestration: sample → infer → evaluate
│       ├── prompts.py           # Prompt templates (EN/PT)
│       ├── sampling.py          # Stratified sampling from COCO
│       ├── segmentation.py      # SAM 2.1 mask refinement
│       └── visualization.py    # Figures and tables generation
├── tests/
├── requirements.txt
└── setup.py
```

### Models

| Model | Variant | Size |
|-------|---------|------|
| Grounding DINO | Swin-B | ~938 MB |
| SAM 2.1 | Hiera Large | ~2.4 GB |

## Setup

### Prerequisites

- Python 3.10+
- NVIDIA GPU with ≥30 GB VRAM (A100 recommended)
- CUDA 12.1+

### Installation

```bash
cd platform/vlm-pipeline

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install the package in editable mode
pip install -e .
```

### Download Data and Weights

```bash
bash scripts/setup_data.sh
```

This script downloads:

1. COCO val2017 images and annotations into `data/coco/`
2. Grounding DINO Swin-B weights into `weights/grounding-dino/`
3. SAM 2.1 Hiera Large weights into `weights/sam2.1/`

## Usage

### Generate Sample

```bash
vlm-pipeline sample --config configs/experiment.yaml
```

Generates a stratified sample from COCO val2017 ensuring all 80 categories are represented.

### Run Experiment

```bash
vlm-pipeline run --config configs/experiment.yaml
```

Executes the full pipeline: stratified sampling → prompt generation → Grounding DINO inference → SAM 2.1 refinement → metric computation.

### Generate Visualizations

```bash
vlm-pipeline visualize --tables-dir results/tables --figures-dir results/figures
```

Produces publication-ready figures from aggregated result tables.

## Experiments

| ID | Description |
|----|-------------|
| **E1** | Overall segmentation quality per class |
| **E2** | English vs Portuguese comparison |
| **E3** | Prompt formulation impact (class name, description, contextual) |
| **E4** | Grounding vs segmentation error diagnosis (conditional metrics) |
| **E5** | Qualitative failure analysis (visual examples) |

## Pre-experiment Validation

Before running the full experiment, verify:

- [ ] COCO val2017 images present in `data/coco/val2017/` (5,000 images)
- [ ] COCO annotations present in `data/coco/annotations/instances_val2017.json`
- [ ] Grounding DINO weights present in `weights/grounding-dino/`
- [ ] SAM 2.1 weights present in `weights/sam2.1/`
- [ ] GPU available and CUDA version ≥12.1 (`nvidia-smi`)
- [ ] Stratified sample generated and validated via `notebooks/01_data_exploration.py`
- [ ] Single-image smoke test passes before full run

## Estimated Runtime

~3 hours on NVIDIA A100 40GB for 500 images × 80 conditions.

## Publishing as a Standalone Public Repository

For manuscript submissions, prefer a dedicated public repository for this pipeline
instead of pointing to local workspace paths.

1. Generate a clean publishable bundle:

```bash
cd platform/vlm-pipeline
bash scripts/prepare_public_release.sh
```

2. Publish the generated bundle to a public GitHub repository
	(recommended name: `vlm-prelabeling-pipeline`).

3. Use the public repository URL (and optional Zenodo DOI) in the paper
	`Declarations` section.

See `PUBLIC_REPO_PREP.md` for the complete workflow.
