#!/usr/bin/env bash
# ── Migrate data from Azure Files to Blob Storage ──
# Azure Files mounts fail due to subscription policy blocking shared key access.
# This script uploads data/weights to blob containers for download via managed identity.
set -euo pipefail

STORAGE_ACCOUNT="stvlmpipe702226"
RG="rg-trainning-models"
DATA_DIR="/Users/brunomalhano/brmalhan-advisory/platform/vlm-pipeline/data"
WEIGHTS_DIR="/Users/brunomalhano/brmalhan-advisory/platform/vlm-pipeline/weights"

echo "=== Step 1: Create blob containers ==="
az storage container create --name vlm-data --account-name "$STORAGE_ACCOUNT" --auth-mode login 2>/dev/null && echo "  Created vlm-data" || echo "  vlm-data already exists"
az storage container create --name vlm-weights --account-name "$STORAGE_ACCOUNT" --auth-mode login 2>/dev/null && echo "  Created vlm-weights" || echo "  vlm-weights already exists"

echo ""
echo "=== Step 2: Upload COCO annotations to blob ==="
azcopy copy "$DATA_DIR/coco/annotations/instances_val2017.json" \
  "https://${STORAGE_ACCOUNT}.blob.core.windows.net/vlm-data/coco/annotations/instances_val2017.json" \
  --overwrite=ifSourceNewer 2>&1 | tail -5

echo ""
echo "=== Step 3: Upload COCO images to blob ==="
azcopy copy "$DATA_DIR/coco/val2017/" \
  "https://${STORAGE_ACCOUNT}.blob.core.windows.net/vlm-data/coco/val2017/" \
  --recursive --overwrite=ifSourceNewer 2>&1 | tail -10

echo ""
echo "=== Step 4: Upload GroundingDINO weights ==="
azcopy copy "$WEIGHTS_DIR/groundingdino_swinb_cogcoor.pth" \
  "https://${STORAGE_ACCOUNT}.blob.core.windows.net/vlm-weights/groundingdino_swinb_cogcoor.pth" \
  --overwrite=ifSourceNewer 2>&1 | tail -5

echo ""
echo "=== Step 5: Upload SAM 2.1 weights ==="
azcopy copy "$WEIGHTS_DIR/sam2.1_hiera_large.pt" \
  "https://${STORAGE_ACCOUNT}.blob.core.windows.net/vlm-weights/sam2.1_hiera_large.pt" \
  --overwrite=ifSourceNewer 2>&1 | tail -5

echo ""
echo "=== Upload complete ==="
echo "Blob containers ready for managed identity access."
