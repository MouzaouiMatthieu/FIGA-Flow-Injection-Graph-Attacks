"""Dataset loading utilities.

All supported datasets use a three-way temporal split (G_train / G_surr / G_test):
  - NIDS is trained on G_train.
  - Surrogate is trained on G_surr (using NIDS hard pseudo-labels).
  - Attack is computed on G_surr, then replayed on G_test for evaluation.
"""
import os
from src.utils.logger import setup_logging
setup_logging()
import logging
logger = logging.getLogger(__name__)
import torch
import dgl
from typing import Any
from pathlib import Path

_STORAGE = os.environ.get("AA_GNN_DATA_ROOT", "data/raw")

# Dataset name → (hardcoded data path, module, class name)
_DATASET_CONFIG = {
    "X-IIoTID": (
        f"{_STORAGE}/X-IIoTID",
        "src.data.graph_creation.datasets.xiiotid",
        "XIIoTIDHeterogeneousGraph",
    ),
    "CICIDS2017": (
        f"{_STORAGE}/CIC-IDS-2017/Traffic_Labelling",
        "src.data.graph_creation.datasets.cicids2017",
        "CICIDS2017HeterogeneousGraphThreeWay",
    ),
}




def merge_graphs_for_nids(g_train: dgl.DGLGraph, g_test: dgl.DGLGraph) -> dgl.DGLGraph:
    """Merge train and test graphs for NIDS evaluation."""
    import copy
    
    g_merged = copy.deepcopy(g_test)
    
    # Get train flow nodes and features
    train_flows = torch.where(g_train.nodes['flow'].data.get('train_mask', 
        torch.ones(g_train.num_nodes('flow'), dtype=torch.bool)))[0]
    train_features = g_train.nodes['flow'].data['h'][train_flows]
    train_labels = g_train.nodes['flow'].data['label'][train_flows]
    
    n_train_flows = len(train_flows)
    g_merged.add_nodes(n_train_flows, ntype='flow')
    start_idx = g_merged.num_nodes('flow') - n_train_flows
    
    g_merged.nodes['flow'].data['h'][start_idx:] = train_features
    g_merged.nodes['flow'].data['label'][start_idx:] = train_labels
    
    # Add merged_mask to identify flows from train
    if 'merged_mask' not in g_merged.nodes['flow'].data:
        g_merged.nodes['flow'].data['merged_mask'] = torch.zeros(
            g_merged.num_nodes('flow'), dtype=torch.bool, device=g_merged.device
        )
    g_merged.nodes['flow'].data['merged_mask'][start_idx:] = True
    
    if 'test_mask' in g_merged.nodes['flow'].data:
        g_merged.nodes['flow'].data['test_mask'][start_idx:] = False
    
    # Merge endpoint connections
    for i, flow_idx in enumerate(train_flows):
        flow_out_edges = g_train.out_edges(flow_idx, etype='depends_on')
        if len(flow_out_edges[0]) > 0:
            target_eps = flow_out_edges[1]
            new_flow_id = start_idx + i
            g_merged.add_edges(
                torch.full((len(target_eps),), new_flow_id, device=g_merged.device),
                target_eps,
                etype='depends_on'
            )
            g_merged.add_edges(
                target_eps,
                torch.full((len(target_eps),), new_flow_id, device=g_merged.device),
                etype='links_to'
            )
    
    return g_merged


def load_dataset(
    dataset_name: str,
    classes: str = "binary",
    train_frac: float = 0.7,
    surr_frac: float = 0.1,
    seed: int = 42,
    return_merged: bool = False,
    **kwargs
) -> Any:
    """
    Load dataset with three-way split.
    
    Args:
        dataset_name: Name of the dataset
        classes: "binary" or "category"
        train_frac: Fraction for training
        surr_frac: Fraction for surrogate
        seed: Random seed
        return_merged: If True, also return merged graph (train+test)
        **kwargs: Additional arguments
        
    Returns:
        If return_merged=False: (dataset, g_train, g_surr, g_test)
        If return_merged=True: (dataset, g_train, g_surr, g_test, g_merged)
    """
    from src.data.graph_creation.datasets.cicids2017 import CICIDS2017HeterogeneousGraphThreeWay
    from src.data.graph_creation.datasets.xiiotid import XIIoTIDHeterogeneousGraph
    
    # Dataset root: prefer explicit `path` kwarg; otherwise use the
    # repository-wide `_DATASET_CONFIG` entry. Fall back to legacy
    # local raw/data layout if the dataset is unknown.
    dataset_root = kwargs.get('path', None)
    if dataset_root is None:
        dataset_entry = _DATASET_CONFIG.get(dataset_name)
        if dataset_entry is not None:
            dataset_root = dataset_entry[0]
        else:
            dataset_root = os.path.join(
                Path(__file__).parent.parent, "data", "raw", dataset_name
            )
    
    if dataset_name == "CICIDS2017":
        dataset = CICIDS2017HeterogeneousGraphThreeWay(
            root=dataset_root,
            classes_def=classes,
            apply_undersampling=True,
            seed=seed,
            train_frac=train_frac,
            surr_frac=surr_frac,
            stratify_attack=True,
            gaussian_init=True,
        )
    elif dataset_name == "X-IIoTID":
        dataset = XIIoTIDHeterogeneousGraph(
            root=dataset_root,
            classes_def=classes,
            seed=seed,
            train_frac=train_frac,
            surr_frac=surr_frac,
            stratify_attack=True,
            gaussian_init=True,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    g_train = dataset.get_dgl(0)
    g_surr = dataset.get_dgl(1)
    g_test = dataset.get_dgl(2)
    
    if return_merged:
        # Try to get pre-merged graph if dataset supports it (len >= 4)
        try:
            if dataset.len() >= 4:
                g_merged = dataset.get_dgl(3)
                logger.info("Loaded pre-merged graph from dataset")
            else:
                raise ValueError("Dataset does not have pre-merged graph")
        except Exception as e:
            logger.warning(f"Pre-merged graph not available: {e}, creating manually")
            g_merged = merge_graphs_for_nids(g_train, g_test)
        
        return dataset, g_train, g_surr, g_test, g_merged
    
    return dataset, g_train, g_surr, g_test

def build_bipartite_graph(raw_data, **kwargs):
    raise NotImplementedError
