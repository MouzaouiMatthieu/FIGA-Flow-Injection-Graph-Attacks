#!/usr/bin/env bash
set -euo pipefail

python main.py \
  --dataset CICIDS2017-Undersampled \
  --attack targeted_evasion \
  --target_model SAGE \
  --target_model_path "$1" \
  --surrogate_models SAGE GCN gat_skip \
  --surrogate_model_paths "$2" "$3" "$4" \
  --budget 100 \
  --output_dir results/evasion
