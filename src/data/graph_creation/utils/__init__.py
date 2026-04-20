"""
Shared utilities for graph creation.

This module contains helper functions used across multiple dataset
implementations, including splitting strategies, graph conversions,
and feature scaling utilities.
"""

from .splitting import temporal_stratified_split
from .feature_scaling import inverse_transform_features, get_feature_perturbation_stats
from ..graph_converters import convert_networkx_to_dgl_heterogeneous

def create_clones(edge_df):
    """Return edge_df unchanged (placeholder for clone generation)."""
    return edge_df

__all__ = [
    "temporal_stratified_split",
    "inverse_transform_features",
    "get_feature_perturbation_stats",
    "create_clones",
    "convert_networkx_to_dgl_heterogeneous",
]
