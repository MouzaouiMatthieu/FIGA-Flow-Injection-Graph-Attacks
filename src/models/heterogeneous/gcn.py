# Heterogeneous GCN implementation

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

from .base import BaseHeterogeneous

class HeteroGCNLayer(nn.Module):
    """Heterogeneous Graph Convolutional Network layer."""
    
    def __init__(self, in_feats, out_feats, activation, dropout=0.2):
        """Initialize HeteroGCNLayer.
        
        Args:
            in_feats: Input feature dimensions
            out_feats: Output feature dimensions
            activation: Activation function
            dropout: Dropout rate
        """
        super(HeteroGCNLayer, self).__init__()
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.conv = dgl.nn.HeteroGraphConv({
            ('endpoint', 'links_to', 'flow'): dgl.nn.GraphConv(
                in_feats, out_feats, activation=None, allow_zero_in_degree=True, norm='both'),
            ('flow', 'depends_on', 'endpoint'): dgl.nn.GraphConv(
                in_feats, out_feats, activation=None, allow_zero_in_degree=True, norm='both')
        }, aggregate='mean')
        # Learnable transform for implicit self-loops
        self.self_loop_weight = nn.Parameter(torch.randn(in_feats, out_feats) * 0.01)

    def forward(self, g, inputs, edge_weight=None, mod_args=None):
        """Forward pass of HeteroGCNLayer.
        
        Args:
            g: Input heterogeneous graph
            inputs: Node features for each node type
            
        Returns:
            Updated node features for each node type
        """
        if mod_args is None:
            mod_args = {}
        if edge_weight is not None:
             mod_args['edge_weight'] = edge_weight

        h = self.conv(g, inputs, mod_args=mod_args)
        # Add implicit self-loop contribution
        for ntype in h:
            if ntype in inputs:
                h[ntype] = h[ntype] + inputs[ntype] @ self.self_loop_weight
        h = {ntype: self.dropout(h[ntype]) for ntype in h}
        return h

class HeteroGCN(BaseHeterogeneous):
    """Heterogeneous Graph Convolutional Network model."""
    
    def __init__(self, in_feats: int, hidden_feats: int, out_feats: int, 
                 num_classes: int, n_layers: int = 4, activation=F.relu, dropout=0.2):
        """Initialize HeteroGCN model.
        
        Args:
            in_feats: Input feature dimensions
            hidden_feats: Hidden layer dimensions
            out_feats: Output feature dimensions
            num_classes: Number of output classes
            n_layers: Number of GCN layers
            activation: Activation function
            dropout: Dropout rate
        """
        super(HeteroGCN, self).__init__()
        assert n_layers >= 2, "Use at least 2 layers"
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        self.name = "Heterogeneous GCN"
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(HeteroGCNLayer(in_feats, hidden_feats, activation, dropout))
        
        # Intermediate layers
        for _ in range(n_layers - 2):
            self.layers.append(HeteroGCNLayer(hidden_feats, hidden_feats, activation, dropout))
            
        # Last layer
        self.layers.append(HeteroGCNLayer(hidden_feats, out_feats, activation, dropout))
        
        # Classifier
        self.classifier = nn.Linear(out_feats, num_classes)
        
        # Optimal batch size for inference (computed once, stored for reuse)
        self.optimal_batch_size = None
        # Flag to avoid recomputing batch size multiple times
        self._optimal_batch_size_initialized = False
    
    def set_optimal_batch_size(self, graph_or_num_nodes):
        """Set optimal batch size for inference based on graph size.
        
        Uses FULL-NEIGHBORHOOD sampling for exact deterministic inference.
        Small batch sizes required because full-neighborhood blocks are large.
        
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
        
        print(f"Set optimal batch size to: {self.optimal_batch_size}")
        self._optimal_batch_size_initialized = True
        
        return self.optimal_batch_size

    def forward(self, g, feats=None, edge_weight=None, mod_args=None, return_embeddings=False):
        """Forward pass with option to return node embeddings."""
        if isinstance(g, list):
            return self._forward_minibatch(g, feats, return_embeddings)
        return self._forward_full_graph(g, feats, edge_weight, mod_args, return_embeddings)

    def _forward_full_graph(self, g, feats=None, edge_weight=None, mod_args=None, return_embeddings=False):
        """Full-batch forward with embedding extraction (no self-loops)."""
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
        """Minibatch forward with embedding extraction. (Self-loops not needed, handled in full graph.)"""
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
    
    def _forward_auto_minibatch(self, g, inputs=None, batch_size=131072, return_embeddings=False):
        """Automatically create and process minibatches for large graphs using EXACT full-neighborhood sampling."""
        import contextlib
        import logging
        logger = logging.getLogger(__name__)
        
        device = next(self.parameters()).device
        g_gpu = g.to(device)
        
        logger.info(f"GCN: Starting EXACT minibatch inference (batch_size={batch_size})")
        
        # FULL-NEIGHBORHOOD sampling for exact deterministic inference
        # Parameter is number of GNN layers - must match model architecture
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(self.n_layers)
        
        # Create flow indices on same device as graph to avoid device mismatch
        flow_indices = torch.arange(g_gpu.num_nodes('flow'), device=device)
        dataloader = dgl.dataloading.DataLoader(
            g_gpu,
            {'flow': flow_indices},
            sampler,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0
        )
        
        all_logits = []
        all_embeddings = {'flow': [], 'endpoint': []} if return_embeddings else None
        
        context = torch.no_grad() if not self.training else contextlib.nullcontext()
        
        with context:
            for input_nodes, output_nodes, blocks in dataloader:
                blocks = [block.to(device) for block in blocks]
                
                # Get input features for this batch
                if inputs is None:
                    h = {
                        'flow': blocks[0].srcnodes['flow'].data['h'],
                        'endpoint': blocks[0].srcnodes['endpoint'].data['h']
                    }
                else:
                    raise ValueError("Auto minibatch does not support external inputs")
                
                # Forward pass
                if return_embeddings:
                    batch_logits, batch_embeddings = self._forward_minibatch(blocks, h, return_embeddings=True)
                    # Store embeddings for this batch
                    for ntype in ['flow', 'endpoint']:
                        if ntype in batch_embeddings:
                            all_embeddings[ntype].append(batch_embeddings[ntype].cpu())
                else:
                    batch_logits = self._forward_minibatch(blocks, h, return_embeddings=False)
                
                all_logits.append(batch_logits.cpu())
                
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        logits = torch.cat(all_logits, dim=0).to(device)
        
        if return_embeddings:
            # Concatenate embeddings for each node type
            embeddings_dict = {}
            for ntype in ['flow', 'endpoint']:
                if all_embeddings[ntype]:
                    embeddings_dict[ntype] = torch.cat(all_embeddings[ntype], dim=0).to(device)
            logger.info(f"GCN: Minibatch inference complete, output shape {logits.shape}")
            return logits, embeddings_dict
        
        logger.info(f"GCN: Minibatch inference complete, output shape {logits.shape}")
        return logits