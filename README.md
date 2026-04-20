# FIGA: Flow Injection Graph Attacks

This repository contains a reduced implementation for heterogeneous-graph evasion attacks, using FIGA. 


- Datasets: `CICIDS-2017`, `X-IIoTID`
- Models: `SAGE`, `GCN`, `GAT`
- Attacks: `targeted_evasion`, `random_evasion`

## Setup

1. Create and activate a Python environment.
2. Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

3. Place dataset files under `data/raw`


## Run Evasion Experiments

### Targeted Evasion

```bash
python main.py \
  --dataset CICIDS2017-Undersampled \
  --attack targeted_evasion \
  --target_model SAGE \
  --target_model_path /path/to/victim_model.pt \
  --surrogate_models SAGE GCN gat_skip \
  --surrogate_model_paths /path/to/surr_sage.pt /path/to/surr_gcn.pt /path/to/surr_gat_skip.pt \
  --budget 20 \
  --output_dir results/evasion
```

### Random Baseline

```bash
python main.py \
  --dataset X-IIoTID \
  --attack random_evasion \
  --target_model GCN \
  --target_model_path /path/to/victim_model.pt \
  --budget 20 \
  --output_dir results/evasion
```


## Reproduce Results

1. Fix seed and budget in CLI options (`--seed`, `--budget`).
2. Run the same command for each dataset/model combination.
3. Collect outputs from:
   - `results/evasion/run_summary.json`
   - `results/evasion/*_metrics.json`

