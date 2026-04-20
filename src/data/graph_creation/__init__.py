# Graph Creation module
import torch
import random
import numpy as np
import os

# Set a default seed for reproducibility
DEFAULT_SEED = 42

def set_seed(seed=DEFAULT_SEED):
    """
    Set seeds for all random number generators used in graph creation.
    
    Args:
        seed: Integer seed value (default: 42)
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# Initialize with default seed
set_seed(DEFAULT_SEED)

from src.data.graph_creation.datasets import (
    DATASET_REGISTRY,
    get_dataset_class,
    CICIDS2017HeterogeneousGraph,
    XIIoTIDHeterogeneousGraph,
)

from src.data.graph_creation.datasets.cicids2017 import pipeline_create_heterogeneous_cicids2017
from src.data.graph_creation.datasets.xiiotid import pipeline_create_heterogeneous_xiiotid

from src.data.graph_creation.heterogeneous import log_memory
from src.data.graph_creation.utils import temporal_stratified_split

pipeline_create_heterogeneous_graph = pipeline_create_heterogeneous_cicids2017

__all__ = [
    "set_seed",
    "DEFAULT_SEED",
    "DATASET_REGISTRY",
    "get_dataset_class",
    "CICIDS2017HeterogeneousGraph",
    "XIIoTIDHeterogeneousGraph",
    "pipeline_create_heterogeneous_graph",
    "pipeline_create_heterogeneous_cicids2017",
    "pipeline_create_heterogeneous_xiiotid",
    "temporal_stratified_split",
    "log_memory",
]
