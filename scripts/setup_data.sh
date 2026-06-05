#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "=== VLM Pipeline: Data & Weights Setup ==="

# ── Create directory structure ──────────────────────────────────────
echo "[1/4] Creating directories..."
mkdir -p data/coco/annotations
mkdir -p data/coco/val2017
mkdir -p data/sample
mkdir -p weights
mkdir -p results/raw
mkdir -p results/tables
mkdir -p results/figures

# ── Download COCO val2017 images ────────────────────────────────────
if [ ! -d "data/coco/val2017" ] || [ -z "$(ls -A data/coco/val2017 2>/dev/null)" ]; then
    echo "[2/4] Downloading COCO val2017 images..."
    curl -L -o data/coco/val2017.zip \
        http://images.cocodataset.org/zips/val2017.zip
    unzip -q -o data/coco/val2017.zip -d data/coco/
    rm -f data/coco/val2017.zip
else
    echo "[2/4] COCO val2017 images already present, skipping."
fi

# ── Download COCO annotations ──────────────────────────────────────
if [ ! -f "data/coco/annotations/instances_val2017.json" ]; then
    echo "[3/4] Downloading COCO val2017 annotations..."
    curl -L -o data/coco/annotations_trainval2017.zip \
        http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    unzip -q -o data/coco/annotations_trainval2017.zip -d data/coco/
    rm -f data/coco/annotations_trainval2017.zip
else
    echo "[3/4] COCO annotations already present, skipping."
fi

# ── Download model weights ─────────────────────────────────────────
echo "[4/4] Downloading model weights..."

# Grounding DINO SwinB
if [ ! -f "weights/groundingdino_swinb_cogcoor.pth" ]; then
    echo "  -> Grounding DINO SwinB weights..."
    curl -L -o weights/groundingdino_swinb_cogcoor.pth \
        https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth
else
    echo "  -> Grounding DINO weights already present, skipping."
fi

# SAM 2.1 Hiera Large
if [ ! -f "weights/sam2.1_hiera_large.pt" ]; then
    echo "  -> SAM 2.1 Hiera Large weights..."
    curl -L -o weights/sam2.1_hiera_large.pt \
        https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
else
    echo "  -> SAM 2.1 weights already present, skipping."
fi

echo ""
echo "=== Setup complete ==="
echo "Project root: $PROJECT_ROOT"
echo "  data/coco/val2017/        - COCO validation images"
echo "  data/coco/annotations/    - COCO annotations"
echo "  weights/                  - Model weights"
echo "  results/{raw,tables,figures} - Output directories"
