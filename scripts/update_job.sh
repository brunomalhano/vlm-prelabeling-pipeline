#!/usr/bin/env bash
# Update the ACA job: remove volume mounts, add blob env vars, use new image
set -euo pipefail

RG="rg-trainning-models"
JOB="vlm-pipeline-job"
ENV_NAME="managedEnvironment-rgtrainningmode-8b63"
ACR="acrkvbqwbbjjkoy6"

cat > /tmp/vlm-job-update.yaml << 'EOF'
properties:
  configuration:
    triggerType: Manual
    replicaTimeout: 14400
    replicaRetryLimit: 0
    manualTriggerConfig:
      parallelism: 1
      replicaCompletionCount: 1
    registries:
      - server: acrkvbqwbbjjkoy6.azurecr.io
        username: acrkvbqwbbjjkoy6
        passwordSecretRef: acr-password
  template:
    containers:
      - name: vlm-pipeline
        image: acrkvbqwbbjjkoy6.azurecr.io/vlm-pipeline:latest
        resources:
          cpu: 8.0
          memory: 56Gi
        env:
          - name: AZURE_STORAGE_ACCOUNT
            value: stvlmpipe702226
          - name: DATA_CONTAINER
            value: vlm-data
          - name: WEIGHTS_CONTAINER
            value: vlm-weights
          - name: RESULTS_CONTAINER
            value: vlm-results
EOF

echo "=== Updating job via ARM REST API ==="
# Get ACR password for the secret
ACR_PASS=$(az acr credential show -n "$ACR" --query "passwords[0].value" -o tsv 2>/dev/null)

# Update job using CLI - remove volumes and update env vars
az containerapp job update \
  -g "$RG" \
  -n "$JOB" \
  --image "acrkvbqwbbjjkoy6.azurecr.io/vlm-pipeline:latest" \
  --set-env-vars \
    "AZURE_STORAGE_ACCOUNT=stvlmpipe702226" \
    "DATA_CONTAINER=vlm-data" \
    "WEIGHTS_CONTAINER=vlm-weights" \
    "RESULTS_CONTAINER=vlm-results" \
  2>&1 | tail -5

echo ""
echo "=== Removing volume mounts via REST API ==="
# Get current job JSON, remove volumes/volumeMounts, PATCH back
JOB_ID="/subscriptions/b0497e33-d40e-4a93-b317-711a39113eca/resourceGroups/$RG/providers/Microsoft.App/jobs/$JOB"

az rest --method patch \
  --url "https://management.azure.com${JOB_ID}?api-version=2024-03-01" \
  --headers "Content-Type=application/json" \
  --body '{
    "properties": {
      "template": {
        "containers": [{
          "name": "vlm-pipeline",
          "image": "acrkvbqwbbjjkoy6.azurecr.io/vlm-pipeline:latest",
          "resources": {
            "cpu": 8.0,
            "memory": "56Gi"
          },
          "env": [
            {"name": "AZURE_STORAGE_ACCOUNT", "value": "stvlmpipe702226"},
            {"name": "DATA_CONTAINER", "value": "vlm-data"},
            {"name": "WEIGHTS_CONTAINER", "value": "vlm-weights"},
            {"name": "RESULTS_CONTAINER", "value": "vlm-results"}
          ]
        }],
        "volumes": []
      }
    }
  }' --query "properties.template.{containers: containers[0].{name:name,env:env[].name,volumeMounts:volumeMounts}, volumes: volumes}" -o json 2>&1

echo ""
echo "=== Done ==="
