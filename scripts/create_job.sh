#!/usr/bin/env bash
set -euo pipefail

RG="rg-trainning-models"
ACA_ENV="managedEnvironment-rgtrainningmode-8b63"
ACR_NAME="acrkvbqwbbjjkoy6"
ACR_RG="rg-demortc"
STORAGE_ACCOUNT="stvlmpipe702226"
JOB_NAME="vlm-pipeline-job"
IMAGE="acrkvbqwbbjjkoy6.azurecr.io/vlm-pipeline:latest"

echo "=== Creating ACA Job ==="

# Step 1: Get storage key
echo "[1/5] Getting storage key..."
STORAGE_KEY=$(az storage account keys list -g "$RG" --account-name "$STORAGE_ACCOUNT" --query '[0].value' -o tsv)
echo "  Key retrieved."

# Step 2: Mount storage on ACA environment
echo "[2/5] Mounting storage on ACA environment..."
az containerapp env storage set \
    --resource-group "$RG" --name "$ACA_ENV" \
    --storage-name vlm-data-storage \
    --azure-file-account-name "$STORAGE_ACCOUNT" \
    --azure-file-account-key "$STORAGE_KEY" \
    --azure-file-share-name vlm-data \
    --access-mode ReadOnly --output none
echo "  Data storage mounted."

az containerapp env storage set \
    --resource-group "$RG" --name "$ACA_ENV" \
    --storage-name vlm-weights-storage \
    --azure-file-account-name "$STORAGE_ACCOUNT" \
    --azure-file-account-key "$STORAGE_KEY" \
    --azure-file-share-name vlm-weights \
    --access-mode ReadOnly --output none
echo "  Weights storage mounted."

# Step 3: Get ACR credentials
echo "[3/5] Getting ACR credentials..."
ACR_USERNAME=$(az acr credential show --name "$ACR_NAME" --resource-group "$ACR_RG" --query username -o tsv)
ACR_PASSWORD=$(az acr credential show --name "$ACR_NAME" --resource-group "$ACR_RG" --query 'passwords[0].value' -o tsv)
echo "  ACR credentials retrieved."

# Step 4: Create the job via YAML (supports volume mounts)
echo "[4/5] Creating Container Apps Job..."
cat > /tmp/vlm-job.yaml <<EOF
properties:
  environmentId: $(az containerapp env show -g $RG -n $ACA_ENV --query id -o tsv)
  configuration:
    triggerType: Manual
    replicaTimeout: 14400
    replicaRetryLimit: 0
    manualTriggerConfig:
      parallelism: 1
      replicaCompletionCount: 1
    registries:
    - server: acrkvbqwbbjjkoy6.azurecr.io
      username: ${ACR_USERNAME}
      passwordSecretRef: acr-password
    secrets:
    - name: acr-password
      value: ${ACR_PASSWORD}
  template:
    containers:
    - name: vlm-pipeline
      image: ${IMAGE}
      resources:
        cpu: 8
        memory: 56Gi
      volumeMounts:
      - volumeName: data-vol
        mountPath: /app/data
      - volumeName: weights-vol
        mountPath: /app/weights
      env:
      - name: AZURE_STORAGE_ACCOUNT
        value: ${STORAGE_ACCOUNT}
      - name: RESULTS_CONTAINER
        value: vlm-results
    volumes:
    - name: data-vol
      storageName: vlm-data-storage
      storageType: AzureFile
    - name: weights-vol
      storageName: vlm-weights-storage
      storageType: AzureFile
  workloadProfileName: NC8as-T4
EOF

az containerapp job create \
    --resource-group "$RG" \
    --name "$JOB_NAME" \
    --yaml /tmp/vlm-job.yaml \
    --output none
echo "  Job created."

# Step 5: Start the job
echo "[5/5] Starting the job..."
EXECUTION=$(az containerapp job start \
    --resource-group "$RG" \
    --name "$JOB_NAME" \
    --query name -o tsv)

echo ""
echo "=========================================="
echo "  Job started!"
echo "=========================================="
echo "Execution   : $EXECUTION"
echo "Resource Grp: $RG"
echo "GPU         : NVIDIA T4 16GB"
echo "Image       : $IMAGE"
echo ""
echo "Monitor:"
echo "  az containerapp job execution show -g $RG -n $JOB_NAME --job-execution-name $EXECUTION -o table"
echo ""
echo "Logs:"
echo "  az containerapp job logs show -g $RG -n $JOB_NAME --execution $EXECUTION --follow"

rm -f /tmp/vlm-job.yaml
