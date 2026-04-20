# FIGA: Flow Injection Graph Attacks

This repository contains a reduced implementation for heterogeneous-graph evasion attacks, using FIGA. 


- Datasets: `CICIDS-2017`, `X-IIoTID`
- Models: `SAGE`, `GCN`, `GAT`
- Attacks: `targeted_evasion`, `random_evasion`

## Setup

This project is developed for Python `3.11.6`.

### Recommended (conda)

```bash
conda env create -f environment.yml
conda activate ADV_ATK_GNN_SURR
python --version
```

### Alternative (pip)

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run Evasion Experiments

### Targeted Evasion

```bash
python main.py \
  --dataset CICIDS2017 \
  --attack targeted_evasion \
  --target_model SAGE \
  --target_model_path /path/to/victim_model.pt \
  --surrogate_models GCN \
  --surrogate_model_paths /path/to/surr_sage.pt \
  --budget 100 \
  --output_dir results/evasion
```

### Random Baseline

```bash
python main.py \
  --dataset X-IIoTID \
  --attack random_evasion \
  --target_model GCN \
  --target_model_path /path/to/victim_model.pt \
  --budget 100 \
  --output_dir results/evasion
```


## Reproduce Results

1. Fix seed and budget in CLI options (`--seed`, `--budget`).
2. Run the same command for each dataset/model combination.
3. Collect outputs from:
   - `results/evasion/run_summary.json`
   - `results/evasion/*_metrics.json`

# Technical Report

The technical report (proofs of Theorem 1, Corrolary 1 and Proposition 1) can be found in `report/technical_report.pdf`.
