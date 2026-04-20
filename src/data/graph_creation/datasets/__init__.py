
"""
Dataset registry and factory for heterogeneous graph creation.

This module provides a unified interface for creating heterogeneous graphs
from different network security datasets. Each dataset has its own module
with specialized preprocessing and graph construction logic.
"""

from typing import Dict, Type
from torch_geometric.data import Dataset

# Import datasets from modular structure
from .cicids2017 import CICIDS2017HeterogeneousGraph, CICIDS2017HeterogeneousGraphThreeWay
from .xiiotid import XIIoTIDHeterogeneousGraph
# Dataset registry mapping dataset names to their classes
DATASET_REGISTRY: Dict[str, Type[Dataset]] = {
    "cicids2017": CICIDS2017HeterogeneousGraph,
    "cicids2017_3way": CICIDS2017HeterogeneousGraphThreeWay,
    "xiiotid": XIIoTIDHeterogeneousGraph,
}

def get_dataset_class(dataset_name: str) -> Type[Dataset]:
    """
    Retrieve dataset class by name.
    
    Args:
        dataset_name: Name of the dataset (case-insensitive)
        
    Returns:
        Dataset class corresponding to the name
        
    Raises:
        ValueError: If dataset name is not recognized
        
    Example:
        dataset_cls = get_dataset_class("cicids2017")
        dataset = dataset_cls(root="/path/to/data", classes_def="binary")
    """
    name_lower = dataset_name.lower()
    if name_lower not in DATASET_REGISTRY:
        available = ", ".join(DATASET_REGISTRY.keys())
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available datasets: {available}"
        )
    return DATASET_REGISTRY[name_lower]

__all__ = [
    "DATASET_REGISTRY",
    "get_dataset_class",
    "CICIDS2017HeterogeneousGraph",
    "CICIDS2017HeterogeneousGraphThreeWay",
    "XIIoTIDHeterogeneousGraph",
]
