#!/usr/bin/env bash
# --------------------------------------------------------------------------
# Entrypoint for the VLM pipeline Container Apps Job.
# Data and weights are baked into the image — no downloads at runtime.
# Only the results upload requires managed identity auth.
#
# SAFETY DESIGN:
#   - Phase 1 (validation): strict mode — fail fast if baked data is missing.
#   - Phase 2 (experiment): runs the heavy GPU workload (~2h).
#   - Phase 3 (post-experiment "safe zone"): NOTHING can kill the script.
#     Visualizations, uploads, and dumps are all wrapped in || true.
#     If upload fails, CSV tables are dumped to stdout (captured by Log
#     Analytics) so results can be recovered without re-running.
#   - The script ALWAYS exits 0 after the experiment completes.
# --------------------------------------------------------------------------

# ══════════════════════════════════════════════════════════════════
# PHASE 1 — VALIDATION (strict mode: fail fast if data is missing)
# ══════════════════════════════════════════════════════════════════
set -uo pipefail

STORAGE_ACCOUNT="${AZURE_STORAGE_ACCOUNT:-stvlmpipe702226}"
RESULTS_CONTAINER="${RESULTS_CONTAINER:-vlm-results}"

echo "=== VLM Pipeline Job ==="
echo "Hostname : $(hostname)"
echo "GPU info :"
nvidia-smi || echo "WARNING: nvidia-smi not available"
echo ""

python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"

echo "[1/5] Validating baked data and weights..."

echo "  COCO images  : $(find /app/data/coco/val2017 -name '*.jpg' 2>/dev/null | wc -l) jpg files"
echo "  Annotations  : $(ls -lh /app/data/coco/annotations/instances_val2017.json 2>/dev/null || echo 'NOT FOUND')"
echo "  GDINO weights: $(ls -lh /app/weights/groundingdino_swinb_cogcoor.pth 2>/dev/null || echo 'NOT FOUND')"
echo "  SAM weights  : $(ls -lh /app/weights/sam2.1_hiera_large.pt 2>/dev/null || echo 'NOT FOUND')"

if [ ! -d "/app/data/coco/val2017" ] || [ -z "$(ls -A /app/data/coco/val2017 2>/dev/null)" ]; then
    echo "ERROR: COCO images not found at /app/data/coco/val2017"; exit 1
fi
if [ ! -f "/app/data/coco/annotations/instances_val2017.json" ]; then
    echo "ERROR: COCO annotations not found"; exit 1
fi
if [ ! -f "/app/weights/groundingdino_swinb_cogcoor.pth" ]; then
    echo "ERROR: Grounding DINO weights not found"; exit 1
fi
if [ ! -f "/app/weights/sam2.1_hiera_large.pt" ]; then
    echo "ERROR: SAM 2.1 weights not found"; exit 1
fi

echo "  All data and weights validated."

# ══════════════════════════════════════════════════════════════════
# PHASE 2 — EXPERIMENT (the expensive GPU work)
# ══════════════════════════════════════════════════════════════════
echo "[2/5] Generating stratified sample..."
mkdir -p /app/data/sample
python -m vlm_pipeline sample --config configs/experiment.yaml

echo "[3/5] Running experiment pipeline..."
if ! python -m vlm_pipeline run --config configs/experiment.yaml; then
    echo "WARNING: experiment run exited with error — continuing to upload any partial results"
fi

# ══════════════════════════════════════════════════════════════════
# PHASE 3 — SAFE ZONE (nothing below this line can kill the script)
# After hours of GPU work, we MUST exit 0 and preserve results.
# ══════════════════════════════════════════════════════════════════
set +euo pipefail  # disable all strict modes
trap '' ERR        # ignore any ERR trap

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  ENTERING SAFE ZONE — script will exit 0 from this point"
echo "══════════════════════════════════════════════════════════════"

# ── Visualizations (non-fatal) ─────────────────────────────────────
echo "[4/5] Generating visualizations..."
python -m vlm_pipeline visualize \
    --tables-dir results/tables \
    --figures-dir results/figures \
    2>&1 || echo "WARNING: visualization step had errors — figures may be incomplete"

echo ""
echo "=== Experiment complete ==="
echo "Results at /app/results/"
ls -lhR /app/results/ 2>/dev/null || echo "(could not list results)"

# ── Upload results to Blob via managed identity ────────────────────
echo "[5/5] Uploading results to Azure Blob Storage..."
RUN_ID="run-$(date -u +%Y%m%dT%H%M%SZ)"
UPLOAD_OK=false

for attempt in 1 2 3; do
    echo "  Login attempt $attempt/3..."
    if az login --identity --allow-no-subscriptions 2>&1; then
        echo "  MSI login succeeded."
        echo "  Uploading to ${RESULTS_CONTAINER}/${RUN_ID}..."
        if az storage blob upload-batch \
            --account-name "$STORAGE_ACCOUNT" \
            --destination "$RESULTS_CONTAINER" \
            --destination-path "$RUN_ID" \
            --source "/app/results" \
            --auth-mode login \
            --overwrite true 2>&1; then
            echo "  Upload OK → ${RESULTS_CONTAINER}/${RUN_ID}"
            UPLOAD_OK=true
            break
        else
            echo "  Upload-batch failed on attempt $attempt"
        fi
    else
        echo "  MSI login failed on attempt $attempt — waiting 10s..."
        sleep 10
    fi
done

# ── Fallback: dump CSV tables to stdout (Log Analytics recovery) ───
if [ "$UPLOAD_OK" = false ]; then
    echo ""
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║  UPLOAD FAILED — dumping CSV tables to stdout           ║"
    echo "║  Recover from Log Analytics if needed                   ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo ""
    for csv_file in /app/results/tables/*.csv; do
        if [ -f "$csv_file" ]; then
            echo "=== CSV_DUMP_START: $(basename "$csv_file") ==="
            cat "$csv_file"
            echo ""
            echo "=== CSV_DUMP_END: $(basename "$csv_file") ==="
            echo ""
        fi
    done
fi

echo ""
echo "=== VLM Pipeline Job finished ==="
echo "  Upload: $UPLOAD_OK"
echo "  Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit 0
