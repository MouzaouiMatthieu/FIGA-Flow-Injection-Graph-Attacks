# Heterogeneous graph models package

from importlib import import_module

from .base import BaseHeterogeneous
from .sage import HeteroGraphSAGE, HeteroSAGELayer
from .gcn import HeteroGCN, HeteroGCNLayer

_gat_module = import_module("." + "gat" + "_skip", __name__)
HeteroGAT = _gat_module.HeteroGATSkip
HeteroGATSkipLayer = _gat_module.HeteroGATSkipLayer

from ..training import train_worker_heterogeneous

__all__ = [
    'BaseHeterogeneous',
    'HeteroGraphSAGE',
    'HeteroSAGELayer',
    'HeteroGAT',
    'HeteroGCN',
    'HeteroGCNLayer',
    'train_worker_heterogeneous'
]
