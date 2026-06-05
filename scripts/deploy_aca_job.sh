#!/usr/bin/env bash
# --------------------------------------------------------------------------
# Deploy VLM Pipeline as an Azure Container Apps Job with GPU (A100).
#
# Prerequisites:
#   - Azure CLI logged in (az login)
#   - Docker installed and running
#   - Local data already downloaded via scripts/setup_data.sh
#
# Usage:
#   bash scripts/deploy_aca_job.sh [--location eastus2]
# --------------------------------------------------------------------------
set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────
LOCATION="${1:-eastus2}"
RESOURCE_GROUP="rg-vlm-pipeline"
ACR_NAME="acrvlmpipeline$(openssl rand -hex 3)"
STORAGE_ACCOUNT="stvlmpipeline$(openssl rand -hex 3)"
ACA_ENV="vlm-pipeline-env"
JOB_NAME="vlm-pipeline-job"
IMAGE_NAME="vlm-pipeline"
IMAGE_TAG="latest"
DATA_CONTAINER="vlm-data"
RESULTS_CONTAINER="vlm-results"
DATA_SHARE="vlm-data-share"
WEIGHTS_SHARE="vlm-weights-share"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== VLM Pipeline — Azure Container Apps Job Deployment ==="
echo "Location       : $LOCATION"
echo "Resource Group : $RESOURCE_GROUP"
echo "Project Root   : $PROJECT_ROOT"
echo ""

# ── Step 1: Resource Group ─────────────────────────────────────────
echo "[1/8] Creating resource group..."
az group create \
    --name "$RESOURCE_GROUP" \
    --location "$LOCATION" \
    --output none

# ── Step 2: Azure Container Registry ──────────────────────────────
echo "[2/8] Creating Azure Container Registry ($ACR_NAME)..."
az acr create \
    --resource-group "$RESOURCE_GROUP" \
    --name "$ACR_NAME" \
    --sku Basic \
    --admin-enabled true \
    --output none

ACR_LOGIN_SERVER=$(az acr show --name "$ACR_NAME" --query loginServer -o tsv)
echo "  ACR: $ACR_LOGIN_SERVER"

# ── Step 3: Storage Account + File Shares ──────────────────────────
echo "[3/8] Creating Storage Account and file shares..."
az storage account create \
    --resource-group "$RESOURCE_GROUP" \
    --name "$STORAGE_ACCOUNT" \
    --location "$LOCATION" \
    --sku Standard_LRS \
    --kind StorageV2 \
    --output none

STORAGE_KEY=$(az storage account keys list \
    --resource-group "$RESOURCE_GROUP" \
    --account-name "$STORAGE_ACCOUNT" \
    --query '[0].value' -o tsv)

# Create file shares for data and weights
az storage share create \
    --account-name "$STORAGE_ACCOUNT" \
    --account-key "$STORAGE_KEY" \
    --name "$DATA_SHARE" \
    --quota 50 \
    --output none

az storage share create \
    --account-name "$STORAGE_ACCOUNT" \
    --account-key "$STORAGE_KEY" \
    --name "$WEIGHTS_SHARE" \
    --quota 10 \
    --output none

# Create results blob container
az storage container create \
    --account-name "$STORAGE_ACCOUNT" \
    --account-key "$STORAGE_KEY" \
    --name "$RESULTS_CONTAINER" \
    --output none

echo "  Storage: $STORAGE_ACCOUNT"

# ── Step 4: Upload data and weights to file shares ─────────────────
echo "[4/8] Uploading COCO data to file share (this may take a while)..."

# Upload annotations
az storage file upload \
    --account-name "$STORAGE_ACCOUNT" \
    --account-key "$STORAGE_KEY" \
    --share-name "$DATA_SHARE" \
    --source "$PROJECT_ROOT/data/coco/annotations/instances_val2017.json" \
    --path "coco/annotations/instances_val2017.json" \
    --output none

# Upload images in batch
az storage file upload-batch \
    --account-name "$STORAGE_ACCOUNT" \
    --account-key "$STORAGE_KEY" \
    --destination "$DATA_SHARE" \
    --source "$PROJECT_ROOT/data/coco/val2017" \
    --destination-path "coco/val2017" \
    --output none \
    --max-connections 16

echo "  Uploading model weights..."
az storage file upload \
    --account-name "$STORAGE_ACCOUNT" \
    --account-key "$STORAGE_KEY" \
    --share-name "$WEIGHTS_SHARE" \
    --source "$PROJECT_ROOT/weights/groundingdino_swinb_cogcoor.pth" \
    --path "groundingdino_swinb_cogcoor.pth" \
    --output none

az storage file upload \
    --account-name "$STORAGE_ACCOUNT" \
    --account-key "$STORAGE_KEY" \
    --share-name "$WEIGHTS_SHARE" \
    --source "$PROJECT_ROOT/weights/sam2.1_hiera_large.pt" \
    --path "sam2.1_hiera_large.pt" \
    --output none

echo "  Data and weights uploaded."

# ── Step 5: Build and push Docker image ────────────────────────────
echo "[5/8] Building and pushing Docker image..."
az acr build \
    --registry "$ACR_NAME" \
    --image "$IMAGE_NAME:$IMAGE_TAG" \
    --file "$PROJECT_ROOT/Dockerfile" \
    "$PROJECT_ROOT"

FULL_IMAGE="$ACR_LOGIN_SERVER/$IMAGE_NAME:$IMAGE_TAG"
echo "  Image: $FULL_IMAGE"

# ── Step 6: Container Apps Environment with GPU ────────────────────
echo "[6/8] Creating Container Apps Environment with GPU workload profile..."
az containerapp env create \
    --resource-group "$RESOURCE_GROUP" \
    --name "$ACA_ENV" \
    --location "$LOCATION" \
    --enable-workload-profiles \
    --output none

# Add GPU workload profile (NC24-A100 = 24 vCPU, 220 GiB memory, 1x A100 80GB)
az containerapp env workload-profile add \
    --resource-group "$RESOURCE_GROUP" \
    --name "$ACA_ENV" \
    --workload-profile-name "gpu-a100" \
    --workload-profile-type "NC24-A100" \
    --min-nodes 0 \
    --max-nodes 1 \
    --output none

# Add storage mounts to the environment
az containerapp env storage set \
    --resource-group "$RESOURCE_GROUP" \
    --name "$ACA_ENV" \
    --storage-name "data-storage" \
    --azure-file-account-name "$STORAGE_ACCOUNT" \
    --azure-file-account-key "$STORAGE_KEY" \
    --azure-file-share-name "$DATA_SHARE" \
    --access-mode ReadOnly \
    --output none

az containerapp env storage set \
    --resource-group "$RESOURCE_GROUP" \
    --name "$ACA_ENV" \
    --storage-name "weights-storage" \
    --azure-file-account-name "$STORAGE_ACCOUNT" \
    --azure-file-account-key "$STORAGE_KEY" \
    --azure-file-share-name "$WEIGHTS_SHARE" \
    --access-mode ReadOnly \
    --output none

echo "  Environment created with GPU workload profile."

# ── Step 7: Get ACR credentials ───────────────────────────────────
ACR_USERNAME=$(az acr credential show --name "$ACR_NAME" --query username -o tsv)
ACR_PASSWORD=$(az acr credential show --name "$ACR_NAME" --query 'passwords[0].value' -o tsv)

# ── Step 8: Create and start the Container Apps Job ────────────────
echo "[7/8] Creating Container Apps Job..."
az containerapp job create \
    --resource-group "$RESOURCE_GROUP" \
    --name "$JOB_NAME" \
    --environment "$ACA_ENV" \
    --image "$FULL_IMAGE" \
    --registry-server "$ACR_LOGIN_SERVER" \
    --registry-username "$ACR_USERNAME" \
    --registry-password "$ACR_PASSWORD" \
    --workload-profile-name "gpu-a100" \
    --cpu 24 \
    --memory 220Gi \
    --trigger-type "Manual" \
    --replica-timeout 14400 \
    --replica-retry-limit 0 \
    --env-vars \
        "AZURE_STORAGE_ACCOUNT=$STORAGE_ACCOUNT" \
        "RESULTS_CONTAINER=$RESULTS_CONTAINER" \
    --output none

# Configure volume mounts via YAML patch (CLI doesn't support --volume-mounts directly in job create)
cat > /tmp/vlm-job-patch.yaml <<EOF
properties:
  template:
    containers:
    - name: $JOB_NAME
      image: $FULL_IMAGE
      resources:
        cpu: 24
        memory: 220Gi
      volumeMounts:
      - volumeName: data-vol
        mountPath: /app/data
      - volumeName: weights-vol
        mountPath: /app/weights
      env:
      - name: AZURE_STORAGE_ACCOUNT
        value: $STORAGE_ACCOUNT
      - name: RESULTS_CONTAINER
        value: $RESULTS_CONTAINER
    volumes:
    - name: data-vol
      storageName: data-storage
      storageType: AzureFile
    - name: weights-vol
      storageName: weights-storage
      storageType: AzureFile
EOF

az containerapp job update \
    --resource-group "$RESOURCE_GROUP" \
    --name "$JOB_NAME" \
    --yaml /tmp/vlm-job-patch.yaml \
    --output none 2>/dev/null || echo "  (YAML patch skipped — volume mounts may need manual configuration)"

rm -f /tmp/vlm-job-patch.yaml

echo "[8/8] Starting the job..."
EXECUTION_NAME=$(az containerapp job start \
    --resource-group "$RESOURCE_GROUP" \
    --name "$JOB_NAME" \
    --query name -o tsv)

echo ""
echo "=========================================="
echo "  Job started successfully!"
echo "=========================================="
echo ""
echo "Execution   : $EXECUTION_NAME"
echo "Resource Grp: $RESOURCE_GROUP"
echo "Location    : $LOCATION"
echo "GPU         : NVIDIA A100 80GB"
echo "Image       : $FULL_IMAGE"
echo ""
echo "Monitor with:"
echo "  az containerapp job execution show \\"
echo "    --resource-group $RESOURCE_GROUP \\"
echo "    --name $JOB_NAME \\"
echo "    --job-execution-name $EXECUTION_NAME"
echo ""
echo "Stream logs:"
echo "  az containerapp job logs show \\"
echo "    --resource-group $RESOURCE_GROUP \\"
echo "    --name $JOB_NAME \\"
echo "    --execution $EXECUTION_NAME \\"
echo "    --follow"
echo ""
echo "Results will be at:"
echo "  Storage: $STORAGE_ACCOUNT / $RESULTS_CONTAINER"
echo ""
echo "Clean up when done:"
echo "  az group delete --name $RESOURCE_GROUP --yes --no-wait"
