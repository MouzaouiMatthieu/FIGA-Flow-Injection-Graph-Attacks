# Heterogeneous GraphSAGE implementation

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import logging
import dgl
import dgl.function as fn

from .base import BaseHeterogeneous

class HeteroSAGELayer(nn.Module):
    """Heterogeneous GraphSAGE layer implementation with configurable aggregation type."""
    def __init__(self, in_feats, out_feats, activation, aggr_type='mean'):
        """Initialize HeteroSAGELayer.
        Args:
            in_feats: Input feature dimensions
            out_feats: Output feature dimensions
            activation: Activation function
            aggr_type: Aggregation type for SAGEConv ('mean', 'max', 'pool', 'lstm')
        """
        super(HeteroSAGELayer, self).__init__()
        self.activation = activation
        self.aggr_type = aggr_type
        # If in_feats is a dict, extract correct dims for each edge type
        if isinstance(in_feats, dict):
            in_feats_ep_to_flow = in_feats.get('endpoint', in_feats[list(in_feats.keys())[0]])
            in_feats_flow_to_ep = in_feats.get('flow', in_feats[list(in_feats.keys())[0]])
        else:
            in_feats_ep_to_flow = in_feats
            in_feats_flow_to_ep = in_feats

        self.conv = dgl.nn.HeteroGraphConv({
            ('endpoint', 'links_to', 'flow'): dgl.nn.SAGEConv(in_feats_ep_to_flow, out_feats, aggr_type),
            ('flow', 'depends_on', 'endpoint'): dgl.nn.SAGEConv(in_feats_flow_to_ep, out_feats, aggr_type)
        }, aggregate=aggr_type)

    def forward(self, g, inputs, return_pre_activation=False, edge_weight=None, mod_args=None):
        """Forward pass of HeteroSAGELayer.
        
        Args:
            g: Input heterogeneous graph
            inputs: Node features for each node type
            return_pre_activation: Whether to return output before activation
            edge_weight: Dictionary of edge weights for each etype (optional)
            mod_args: Dictionary of arguments to pass to underlying conv layers (optional)
            
        Returns:
            Updated node features for each node type
        """
        if mod_args is None:
            mod_args = {}
        if edge_weight is not None:
            mod_args['edge_weight'] = edge_weight

        agg = self.conv(g, inputs, mod_args=mod_args)
        if return_pre_activation:
            return agg
        out = {ntype: self.activation(h) for ntype, h in agg.items()}
        return out


class HeteroGraphSAGE(BaseHeterogeneous):
    """Heterogeneous GraphSAGE model with configurable aggregation type."""
    def __init__(self, in_feats, hidden_feats, out_feats, num_classes, 
                 n_layers=4, activation=F.relu, dropout=0.2, aggr_type='mean'):
        """Initialize HeteroGraphSAGE model.
        Args:
            in_feats: Input feature dimensions
            hidden_feats: Hidden layer dimensions
            out_feats: Output feature dimensions
            num_classes: Number of output classes
            n_layers: Number of GraphSAGE layers
            activation: Activation function
            dropout: Dropout rate
            aggr_type: Aggregation type for SAGEConv ('mean', 'max', 'pool', 'lstm')
        """
        super(HeteroGraphSAGE, self).__init__()
        assert n_layers >= 2, "Use at least 2 layers"
        self.aggr_type = aggr_type
        self.name = f"Heterogeneous GraphSAGE (aggr={aggr_type})" if aggr_type != 'mean' else "Heterogeneous GraphSAGE"
        self.dropout = nn.Dropout(dropout)
        self.n_layers = n_layers
        self.activation = activation
        self.layers = nn.ModuleList()
        # First layer
        self.layers.append(HeteroSAGELayer(in_feats, hidden_feats, activation, aggr_type=aggr_type))
        # Intermediate layers (hidden -> hidden)
        for _ in range(n_layers - 2):
            self.layers.append(HeteroSAGELayer(hidden_feats, hidden_feats, activation, aggr_type=aggr_type))
        # Last layer
        self.layers.append(HeteroSAGELayer(hidden_feats, out_feats, activation, aggr_type=aggr_type))
        # MLP classifier
        self.classifier = nn.Linear(out_feats, num_classes)
        
        # Optimal batch size for inference (computed once, stored for reuse)
        self.optimal_batch_size = None
        # Flag to avoid recomputing batch size multiple times
        self._optimal_batch_size_initialized = False
    
    def set_optimal_batch_size(self, graph_or_num_nodes):
        """Set optimal batch size for inference based on graph size.
        
        Uses neighbor sampling (same as training) for memory-efficient inference.
        Batch sizes can be larger than full-neighborhood sampling.
        
        Args:
            graph_or_num_nodes: DGLHeteroGraph or int (number of flow nodes)
        
        Returns:
            Optimal batch size or None for small graphs
        """
        if hasattr(graph_or_num_nodes, 'num_nodes'):
            num_flow_nodes = graph_or_num_nodes.num_nodes('flow')
        else:
            num_flow_nodes = graph_or_num_nodes

        if num_flow_nodes > 10_000_000:
            self.optimal_batch_size = 32768  
        else:
            self.optimal_batch_size = None
        print(f"\n\nSet optimal batch size to: {self.optimal_batch_size}\n")
        return self.optimal_batch_size

    def initialize_optimal_batch_size(self, graph_or_num_nodes):
        """Initialize `optimal_batch_size` once. Safe to call multiple times; the
        computation runs only the first time.

        This allows callers (or a loader) to set batch sizing explicitly after
        loading a saved model, avoiding any per-forward checks.
        """
        if getattr(self, '_optimal_batch_size_initialized', False):
            return self.optimal_batch_size

        # Delegate to set_optimal_batch_size which contains the sizing logic
        self.set_optimal_batch_size(graph_or_num_nodes)
        self._optimal_batch_size_initialized = True
        return self.optimal_batch_size

# In HeteroGraphSAGE class, modify forward method:

    def forward(self, g, feats=None, edge_weight=None, mod_args=None, return_embeddings=False):
        """Forward pass with option to return node embeddings."""
        if isinstance(g, list):
            return self._forward_minibatch(g, feats, return_embeddings)
        return self._forward_full_graph(g, feats, edge_weight, mod_args, return_embeddings)

    def _forward_full_graph(self, g, feats=None, edge_weight=None, mod_args=None, return_embeddings=False):
        """Full-batch forward with embedding extraction."""
        if feats is None:
            h = {
                'endpoint': g.nodes['endpoint'].data['h'],
                'flow': g.nodes['flow'].data['h']
            }
        else:
            h = feats

        for i, layer in enumerate(self.layers):
            h = layer(g, h, edge_weight=edge_weight, mod_args=mod_args)
            if i < self.n_layers - 1:
                h = {k: self.dropout(v) for k, v in h.items()}
        
        # Classifier
        logits = self.classifier(h['flow'])
        
        if return_embeddings:
            return logits, h
        
        return logits

    def _forward_minibatch(self, blocks, feats=None, return_embeddings=False):
        """Minibatch forward with embedding extraction."""
        if feats is None:
            h = {
                'endpoint': blocks[0].srcnodes['endpoint'].data['h'],
                'flow': blocks[0].srcnodes['flow'].data['h']
            }
        else:
            h = feats

        for i, layer in enumerate(self.layers):
            h = layer(blocks[i], h)
            if i < self.n_layers - 1:
                h = {k: self.dropout(v) for k, v in h.items()}
        
        logits = self.classifier(h['flow'])
        
        if return_embeddings:
            return logits, h
        
        return logits
    def _forward_auto_minibatch(self, g, feats=None, batch_size=131072):
        """Automatically create and process minibatches for large graphs using neighbor sampling.
        
        Uses neighbor sampling (same fanout as training) for memory-efficient inference.
        For deterministic results in greedy search, set random seed before calling model.forward().
        """
        import contextlib
        import logging
        logger = logging.getLogger(__name__)
        
        device = next(self.parameters()).device
        

        
        # NEIGHBOR sampling matching training configuration
        # Rationale: Full-neighborhood sampling causes OOM on large graphs
        # Use same fanout as training for consistency and memory efficiency
        # For determinism in greedy search, set random seed before calling forward()
        if self.n_layers <= 3:
            fanout = [15] * self.n_layers
        else:
            fanout = [20, 15, 10] + [10] * (self.n_layers - 3)
        

        sampler = dgl.dataloading.MultiLayerNeighborSampler(fanout)
        
        # Create flow indices on CPU (DGL will handle GPU transfer when device=device is set)
        # Rationale: Match training pattern - graph and indices both on CPU, DataLoader transfers to GPU
        flow_indices = torch.arange(g.num_nodes('flow'))
        
        # Ensure graph is on CPU for DataLoader (it will transfer to GPU internally)
        # This matches training setup where G is on CPU and device parameter handles transfer
        g_cpu = g.cpu() if g.device.type != 'cpu' else g
        
        # CRITICAL: Use device=device to keep everything on GPU (same as training)
        # This avoids slow CPU-GPU transfers and matches training performance
        dataloader = dgl.dataloading.DataLoader(
            g_cpu,  # Graph on CPU (will be transferred to device internally)
            {'flow': flow_indices},  # Indices on CPU
            sampler,
            batch_size=batch_size,
            shuffle=False,  # CRITICAL: maintain order for correct reconstruction
            drop_last=False,
            num_workers=0,
            device=device  # DataLoader handles GPU transfer internally (same as training)
        )
        
        all_logits = []
        context = torch.no_grad() if not self.training else contextlib.nullcontext()
        
        # Only show the detailed minibatch tqdm progress when logger is DEBUG
        disable_progress = not logging.getLogger(__name__).isEnabledFor(logging.DEBUG)
        with context:
            for input_nodes, output_nodes, blocks in tqdm(dataloader, desc="EXACT Minibatch Inference", unit="batch", leave=False, disable=disable_progress):
                # Blocks already on GPU since device=device in DataLoader
                # Get input features for both node types from blocks
                h = {
                    'flow': blocks[0].srcnodes['flow'].data['h'],
                    'endpoint': blocks[0].srcnodes['endpoint'].data['h']
                }
                
                # Forward pass through layers
                batch_logits = self._forward_minibatch(blocks, feats=h)
                
                # Keep on GPU - no unnecessary transfers
                all_logits.append(batch_logits)
        
        # Concatenate on GPU (fast)
        result = torch.cat(all_logits, dim=0)
        return result
