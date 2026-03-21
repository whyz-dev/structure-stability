#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_DATA_ROOT="/Users/mgo/Downloads/open (7) 2"

if [ -z "${PHYSICS_DATA_ROOT:-}" ] && [ -d "$DEFAULT_DATA_ROOT" ]; then
  export PHYSICS_DATA_ROOT="$DEFAULT_DATA_ROOT"
fi

export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTHONDONTWRITEBYTECODE=1

cd "$SCRIPT_DIR"

if [ ! -x "$SCRIPT_DIR/.venv/bin/python" ]; then
  bash "$SCRIPT_DIR/bootstrap_mac_env.sh"
fi

if ! "$SCRIPT_DIR/.venv/bin/python" - <<'PY'
import cv2
import numpy
import pandas
import sklearn
import torch
import torchvision
from PIL import Image
PY
then
  bash "$SCRIPT_DIR/bootstrap_mac_env.sh"
fi

# shellcheck disable=SC1091
source "$SCRIPT_DIR/.venv/bin/activate"

PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"

"$PYTHON_BIN" full_physics_solution.py full-run \
  --out-dir "$SCRIPT_DIR/runs/final" \
  --backbone efficientnet_v2_s \
  --pretrained \
  --image-size 288 \
  --batch-size 4 \
  --epochs 12 \
  --num-folds 5 \
  --num-workers 0 \
  --tta-passes 4
