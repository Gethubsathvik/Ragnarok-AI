#!/usr/bin/env bash
# Linux / macOS / WSL setup. Usage:  bash setup_env.sh
# Override CUDA wheel tag if needed:  CUDA_TAG=cu124 bash setup_env.sh
set -e
CUDA_TAG="${CUDA_TAG:-cu128}"   # cu128 covers RTX 20/30/40/50-series; use cu124/cu121 for older drivers

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

echo ">>> Installing PyTorch (CUDA wheel tag: ${CUDA_TAG})"
pip install torch --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

echo ">>> Installing remaining libraries"
pip install -r requirements.txt

echo ">>> Verifying GPU"
python scripts/00_verify_env.py
