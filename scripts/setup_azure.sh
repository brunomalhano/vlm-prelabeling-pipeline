#!/usr/bin/env bash
# --------------------------------------------------------------------------
# Setup Azure infra for VLM Pipeline using EXISTING ACA environment.
# --------------------------------------------------------------------------
set -euo pipefail

# ── Existing resources ─────────────────────────────────────────────
RG="rg-trainning-models"
LOCATION="westus"
ACA_ENV="managedEnvironment-rgtrainningmode-8b63"
ACR_NAME="acrkvbqwbbjjkoy6"
ACR_RG="rg-demortc"

# ── New resources ──────────────────────────────────────────────────
STORAGE_ACCOUNT="stvlmpipe$(date +%s | tail -c 7)"
DATA_SHARE="vlm-data"
WEIGHTS_SHARE="vlm-weights"
RESULTS_CONTAINER="vlm-results"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== VLM Pipeline Azure Setup ==="
echo "RG       : $RG"
echo "ACA Env  : $ACA_ENV"
echo "ACR      : $ACR_NAME ($ACR_RG)"
echo "Storage  : $STORAGE_ACCOUNT"
echo "Project  : $PROJECT_ROOT"
echo ""

# ── Step 1: Storage Account ───────────────────────────────────────
echo "[1/5] Creating Storage Account..."
az storage account create \
    --resource-group "$RG" \
    --name "$STORAGE_ACCOUNT" \
    --location "$LOCATION" \
    --sku Standard_LRS \
    --kind StorageV2 \
    --output none
echo "  Done: $STORAGE_ACCOUNT"

STORAGE_KEY=$(az storage account keys list \
    --resource-group "$RG" \
    --account-name "$STORAGE_ACCOUNT" \
    --query '[0].value' -o tsv)

# ── Step 2: File Shares + Blob Container ──────────────────────────
echo "[2/5] Creating file shares and blob container..."
az storage share create --account-name "$STORAGE_ACCOUNT" --account-key "$STORAGE_KEY" --name "$DATA_SHARE" --quota 50 --output none
az storage share create --account-name "$STORAGE_ACCOUNT" --account-key "$STORAGE_KEY" --name "$WEIGHTS_SHARE" --quota 10 --output none
az storage container create --account-name "$STORAGE_ACCOUNT" --account-key "$STORAGE_KEY" --name "$RESULTS_CONTAINER" --output none
echo "  Done: $DATA_SHARE, $WEIGHTS_SHARE, $RESULTS_CONTAINER"

# ── Step 3: Upload data ──────────────────────────────────────────
echo "[3/5] Uploading COCO annotations..."
az storage file upload \
    --account-name "$STORAGE_ACCOUNT" --account-key "$STORAGE_KEY" \
    --share-name "$DATA_SHARE" \
    --source "$PROJECT_ROOT/data/coco/annotations/instances_val2017.json" \
    --path "coco/annotations/instances_val2017.json" \
    --output none
echo "  Annotations uploaded."

echo "  Uploading COCO images (5000 files, ~1GB)..."
az storage file upload-batch \
    --account-name "$STORAGE_ACCOUNT" --account-key "$STORAGE_KEY" \
    --destination "$DATA_SHARE" \
    --source "$PROJECT_ROOT/data/coco/val2017" \
    --destination-path "coco/val2017" \
    --output none \
    --max-connections 16
echo "  Images uploaded."

echo "  Uploading model weights..."
az storage file upload \
    --account-name "$STORAGE_ACCOUNT" --account-key "$STORAGE_KEY" \
    --share-name "$WEIGHTS_SHARE" \
    --source "$PROJECT_ROOT/weights/groundingdino_swinb_cogcoor.pth" \
    --path "groundingdino_swinb_cogcoor.pth" \
    --output none

az storage file upload \
    --account-name "$STORAGE_ACCOUNT" --account-key "$STORAGE_KEY" \
    --share-name "$WEIGHTS_SHARE" \
    --source "$PROJECT_ROOT/weights/sam2.1_hiera_large.pt" \
    --path "sam2.1_hiera_large.pt" \
    --output none
echo "  Weights uploaded."

# ── Step 4: Mount storage on ACA environment ──────────────────────
echo "[4/5] Mounting file shares on ACA environment..."
az containerapp env storage set \
    --resource-group "$RG" --name "$ACA_ENV" \
    --storage-name "vlm-data-storage" \
    --azure-file-account-name "$STORAGE_ACCOUNT" \
    --azure-file-account-key "$STORAGE_KEY" \
    --azure-file-share-name "$DATA_SHARE" \
    --access-mode ReadOnly \
    --output none

az containerapp env storage set \
    --resource-group "$RG" --name "$ACA_ENV" \
    --storage-name "vlm-weights-storage" \
    --azure-file-account-name "$STORAGE_ACCOUNT" \
    --azure-file-account-key "$STORAGE_KEY" \
    --azure-file-share-name "$WEIGHTS_SHARE" \
    --access-mode ReadOnly \
    --output none
echo "  Storage mounted on ACA environment."

# ── Step 5: Enable ACR admin ─────────────────────────────────────
echo "[5/5] Enabling ACR admin access..."
az acr update --name "$ACR_NAME" --resource-group "$ACR_RG" --admin-enabled true --output none 2>/dev/null || true
echo "  ACR admin enabled."

# ── Save vars ─────────────────────────────────────────────────────
cat > /tmp/vlm-deploy-vars.sh <<EOF
export STORAGE_ACCOUNT="$STORAGE_ACCOUNT"
export STORAGE_KEY="$STORAGE_KEY"
export RG="$RG"
export ACA_ENV="$ACA_ENV"
export ACR_NAME="$ACR_NAME"
export ACR_RG="$ACR_RG"
EOF

echo ""
echo "=== Setup Complete ==="
echo "Variables saved to /tmp/vlm-deploy-vars.sh"
echo "Next: Build image and create job."
