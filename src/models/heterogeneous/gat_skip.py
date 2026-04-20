import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
from tqdm import tqdm

from .base import BaseHeterogeneous


# ------------------------------------------------------------------
# Layer
# ------------------------------------------------------------------

class HeteroGATSkipLayer(nn.Module):
    def __init__(self, in_feats, out_feats, num_heads=4, dropout=0.0):
        super().__init__()

        self.out_dim = out_feats * num_heads

        # GAT convolution
        self.conv = dglnn.HeteroGraphConv({
            ('endpoint', 'links_to', 'flow'):
                dglnn.GATConv(
                    in_feats, out_feats, num_heads,
                    feat_drop=dropout,
                    attn_drop=dropout,
                    allow_zero_in_degree=True
                ),
            ('flow', 'depends_on', 'endpoint'):
                dglnn.GATConv(
                    in_feats, out_feats, num_heads,
                    feat_drop=dropout,
                    attn_drop=dropout,
                    allow_zero_in_degree=True
                ),
        }, aggregate='mean')

        # Residual projection
        self.res_proj = nn.ModuleDict({
            'flow': nn.Linear(in_feats, self.out_dim) if in_feats != self.out_dim else nn.Identity(),
            'endpoint': nn.Linear(in_feats, self.out_dim) if in_feats != self.out_dim else nn.Identity(),
        })

    def forward(self, g, inputs):
        """
        g: full graph or block
        inputs: dict of node_type -> features
        """
        # GAT forward
        h_new = self.conv(g, inputs)

        # Flatten multi-head outputs
        h_new = {k: v.flatten(1) for k, v in h_new.items()}

        # For blocks, keep only destination nodes
        if hasattr(g, 'dsttypes'):
            h_res = {}
            for k in h_new:
                if k in inputs and k in g.dsttypes:
                    num_dst = g.num_dst_nodes(k)
                    h_res[k] = self.res_proj[k](inputs[k][:num_dst])
        else:
            h_res = {k: self.res_proj[k](inputs[k]) for k in inputs if k in h_new}

        # Skip connection
        h = {}
        for k in h_new:
            if k in h_res:
                h[k] = h_new[k] + h_res[k]
            else:
                h[k] = h_new[k]

        return h


# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------

class HeteroGATSkip(BaseHeterogeneous):
    def __init__(
        self,
        in_feats,
        hidden_feats,
        out_feats,
        num_heads=4,
        num_classes=2,
        n_layers=2,
        dropout=0.0,
        activation=F.elu,
    ):
        super().__init__()

        assert n_layers >= 2, "Use at least 2 layers"

        self.n_layers = n_layers
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.name = "Heterogeneous GAT with Skip Connections"
        self.num_heads = num_heads

        self.layers = nn.ModuleList()

        # First layer
        self.layers.append(
            HeteroGATSkipLayer(in_feats, hidden_feats, num_heads, dropout)
        )

        # Hidden layers
        for _ in range(n_layers - 2):
            self.layers.append(
                HeteroGATSkipLayer(
                    hidden_feats * num_heads,
                    hidden_feats,
                    num_heads,
                    dropout
                )
            )

        # Last layer
        self.layers.append(
            HeteroGATSkipLayer(
                hidden_feats * num_heads,
                out_feats,
                num_heads,
                dropout
            )
        )

        # Classifier
        self.classifier = nn.Linear(out_feats * num_heads, num_classes)

        # Optimal batch size for inference
        self.optimal_batch_size = None
        self._optimal_batch_size_initialized = False

    def set_optimal_batch_size(self, graph_or_num_nodes):
        if hasattr(graph_or_num_nodes, 'num_nodes'):
            num_flow_nodes = graph_or_num_nodes.num_nodes('flow')
        else:
            num_flow_nodes = graph_or_num_nodes

        if num_flow_nodes > 1_000_000:
            self.optimal_batch_size = 32768
        elif num_flow_nodes > 600_000:
            self.optimal_batch_size = 2048
        else:
            self.optimal_batch_size = None

        print(f"Set optimal batch size to: {self.optimal_batch_size}")
        return self.optimal_batch_size

    def initialize_optimal_batch_size(self, graph_or_num_nodes):
        if getattr(self, '_optimal_batch_size_initialized', False):
            return self.optimal_batch_size
        self.set_optimal_batch_size(graph_or_num_nodes)
        self._optimal_batch_size_initialized = True
        return self.optimal_batch_size

    def forward(
        self,
        g,
        feats=None,
        edge_weight=None,
        mod_args=None,
        return_embeddings=False
    ):
        # If g is a list, it's a minibatch (blocks); delegate to minibatch forward
        if isinstance(g, list):
            return self._forward_minibatch(g, feats, return_embeddings=return_embeddings)

        # Input features
        if feats is None:
            h = {
                'endpoint': g.nodes['endpoint'].data['h'],
                'flow': g.nodes['flow'].data['h']
            }
        else:
            h = feats

        # Forward through layers
        for i, layer in enumerate(self.layers):
            h = layer(g, h)

            # Apply activation + dropout except last layer
            if i < self.n_layers - 1:
                h = {k: self.activation(v) for k, v in h.items()}
                h = {k: self.dropout(v) for k, v in h.items()}

        # Classification on flow nodes
        logits = self.classifier(h['flow'])

        if return_embeddings:
            return logits, h

        return logits

    def _forward_full_graph(self, g, feats=None, edge_weight=None, mod_args=None, return_embeddings=False):
        return self.forward(g, feats, edge_weight, mod_args, return_embeddings)

    def _forward_minibatch(self, blocks, feats=None, return_embeddings=False):
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
                h = {k: self.activation(v) for k, v in h.items()}
                h = {k: self.dropout(v) for k, v in h.items()}

        logits = self.classifier(h['flow'])

        if return_embeddings:
            return logits, h

        return logits