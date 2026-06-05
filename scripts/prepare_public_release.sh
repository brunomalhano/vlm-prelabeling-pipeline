#!/usr/bin/env bash
set -euo pipefail

# Prepare a clean, reusable standalone bundle for publishing this pipeline
# to a dedicated public GitHub repository.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-${ROOT_DIR}/.tmp/public-release/vlm-prelabeling-pipeline}"

mkdir -p "${OUT_DIR}"

rsync -a --delete \
  --exclude ".git" \
  --exclude ".venv" \
  --exclude "__pycache__" \
  --exclude ".pytest_cache" \
  --exclude "data/coco" \
  --exclude "weights" \
  --exclude "results/raw" \
  --exclude "results/run-*" \
  "${ROOT_DIR}/" "${OUT_DIR}/"

cat <<EOF
Public release bundle prepared at:
  ${OUT_DIR}

Next steps:
  1) cd "${OUT_DIR}"
  2) git init
  3) git add . && git commit -m "Initial public release"
  4) git remote add origin <YOUR_PUBLIC_REPO_URL>
  5) git push -u origin main
EOF
