# --------------------------------------------------------------------------
# VLM Pre-Labeling Pipeline — GPU container for Azure Container Apps Jobs
# --------------------------------------------------------------------------
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps + az CLI (for managed identity auth) + azcopy
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3-pip \
        libgl1 libglib2.0-0 curl unzip wget ca-certificates gnupg lsb-release && \
    # Azure CLI
    curl -sL https://aka.ms/InstallAzureCLIDeb | bash && \
    # azcopy
    curl -sL https://aka.ms/downloadazcopy-v10-linux -o /tmp/azcopy.tar.gz && \
    tar -xzf /tmp/azcopy.tar.gz -C /tmp && \
    cp /tmp/azcopy_linux_amd64_*/azcopy /usr/local/bin/ && \
    chmod +x /usr/local/bin/azcopy && \
    rm -rf /tmp/azcopy* && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps (cached layer) — use cu121 index so PyTorch matches CUDA 12.x driver
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 && \
    pip install -r requirements.txt

# Model weights — baked into image (static, publicly available, ~1.6 GB).
# Cached as its own layer so source-code changes do not re-download.
RUN mkdir -p /app/weights && \
    wget -q --show-progress \
        -O /app/weights/groundingdino_swinb_cogcoor.pth \
        "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth" && \
    wget -q --show-progress \
        -O /app/weights/sam2.1_hiera_large.pt \
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt" && \
    echo "Weights baked: $(du -sh /app/weights/)"

# COCO val2017 — baked from official public source (~1 GB).
# Eliminates runtime download and any dependency on blob auth for reads.
RUN mkdir -p /app/data/coco/val2017 /app/data/coco/annotations && \
    for attempt in 1 2 3; do \
        wget -q --show-progress --tries=3 --retry-connrefused \
            -O /tmp/val2017.zip \
            "http://images.cocodataset.org/zips/val2017.zip" && break; \
        echo "RETRY $attempt for val2017.zip"; rm -f /tmp/val2017.zip; \
    done && \
    unzip -q /tmp/val2017.zip -d /app/data/coco/ && \
    rm /tmp/val2017.zip && \
    for attempt in 1 2 3; do \
        wget -q --show-progress --tries=3 --retry-connrefused \
            -O /tmp/annotations.zip \
            "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" && break; \
        echo "RETRY $attempt for annotations.zip"; rm -f /tmp/annotations.zip; \
    done && \
    mkdir -p /tmp/ann && \
    unzip -q /tmp/annotations.zip "annotations/instances_val2017.json" -d /tmp/ann/ && \
    mv /tmp/ann/annotations/instances_val2017.json /app/data/coco/annotations/ && \
    rm -rf /tmp/ann /tmp/annotations.zip && \
    echo "COCO: $(find /app/data/coco/val2017 -name '*.jpg' | wc -l) images baked"

# Pipeline source
COPY src/ src/
COPY setup.py .
RUN pip install -e .

# Config
COPY configs/ configs/

# Entrypoint script
COPY scripts/run_job.sh .
RUN chmod +x run_job.sh

# Data and weights:
#   - Model weights (GDINO + SAM) and COCO val2017 are baked at build time.
#   - Results are uploaded to Azure Blob at runtime via managed identity.
# Environment vars required:
#   AZURE_STORAGE_ACCOUNT  — storage account name
#   RESULTS_CONTAINER      — blob container for results upload (default: vlm-results)

CMD ["./run_job.sh"]
