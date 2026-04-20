# Flow Graph E-GraphSAGE implementation
# Adapted from models/egraphsage.py for flow graph representation (edge classification)

import torch
import torch.nn as nn
import dgl
import dgl.function as fn
import matplotlib.pyplot as plt
import numpy as np
import logging
from typing import Tuple, Optional
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve
from sklearn.utils import class_weight
from src.models.training import train_worker

logger = logging.getLogger(__name__)


class SAGELayer(torch.nn.Module):
    """
    GraphSAGE layer with edge feature integration for flow graphs.
    
    Academic justification:
        This layer implements the E-GraphSAGE message passing scheme where
        edge features are incorporated during aggregation. For flow graphs,
        edges represent network flows, so edge features carry critical
        classification information.
    """
    def __init__(self, ndim_in, edims, ndim_out, activation):
        """
        Initialize SAGELayer with edge feature integration.
        
        Args:
            ndim_in: Input node feature dimension
            edims: Edge feature dimension
            ndim_out: Output node feature dimension
            activation: Activation function (e.g., F.relu)
        """
        super(SAGELayer, self).__init__()
        # Message function: concatenate source node features with edge features
        self.W_msg = nn.Linear(ndim_in + edims, ndim_out)
        # Apply function: combine node features with aggregated messages
        self.W_apply = nn.Linear(ndim_in + ndim_out, ndim_out)
        self.activation = activation

    def message_func(self, edges):
        """
        Message passing function incorporating edge features.
        
        Args:
            edges: DGL EdgeBatch with src, dst, and data attributes
            
        Returns:
            Dictionary with messages for aggregation
            
        Academic justification:
            Concatenating edge features with source node features allows
            the model to learn flow-aware representations, critical for
            network intrusion detection where flow characteristics
            (packet sizes, timing, protocols) carry attack signatures.
        """
        # Handle both 2D and 3D tensors
        src_h = edges.src["h"]
        edge_h = edges.data["h"]
        
        # If 2D, add batch dimension
        if src_h.dim() == 2:
            src_h = src_h.unsqueeze(1)
        if edge_h.dim() == 2:
            edge_h = edge_h.unsqueeze(1)
        
        return {"m": self.W_msg(torch.cat([src_h, edge_h], 2))}

    def update_edge_features(self, g):
        """
        Update edge features by averaging connected node embeddings.
        
        Academic justification:
            Edge features evolve through layers by incorporating structural
            context from their endpoints. This creates richer edge
            representations for classification.
        """
        src_emb = g.ndata["h"][g.edges()[0]]
        dst_emb = g.ndata["h"][g.edges()[1]]
        g.edata["h"] = (src_emb + dst_emb) / 2

    def forward(self, g_dgl, nfeats, efeats):
        """
        Forward pass of SAGELayer.
        
        Args:
            g_dgl: DGL graph
            nfeats: Node features [num_nodes, feature_dim] or [num_nodes, batch_size, feature_dim]
            efeats: Edge features [num_edges, feature_dim] or [num_edges, batch_size, feature_dim]
            
        Returns:
            Updated node and edge features
        """
        with g_dgl.local_scope():
            g = g_dgl
            
            # Add batch dimension if not present
            if nfeats.dim() == 2:
                nfeats = nfeats.unsqueeze(1)
            if efeats.dim() == 2:
                efeats = efeats.unsqueeze(1)
            
            g.ndata["h"] = nfeats
            g.edata["h"] = efeats

            # Message passing with edge feature integration
            g.update_all(self.message_func, fn.mean("m", "h_neigh"))

            # Update node features
            new_node_feats = self.activation(
                self.W_apply(torch.cat([g.ndata["h"], g.ndata["h_neigh"]], dim=2))
            )
            g.ndata["h"] = new_node_feats

            # Update edge features based on latest node embeddings
            self.update_edge_features(g)
            return g.ndata["h"], g.edata["h"]


class SAGE(torch.nn.Module):
    """
    Multi-layer E-GraphSAGE encoder for flow graphs.
    
    Academic justification:
        Two-layer architecture aggregates information from 2-hop neighborhoods,
        capturing local network topology patterns critical for intrusion detection.
    """
    def __init__(self, ndim_in, ndim_out, edim, activation, dropout):
        """
        Initialize SAGE encoder.
        
        Args:
            ndim_in: Input node feature dimension
            ndim_out: Output node feature dimension
            edim: Edge feature dimension (consistent across layers)
            activation: Activation function
            dropout: Dropout probability
        """
        super(SAGE, self).__init__()
        self.layers = nn.ModuleList()
        # Layer 1: Input dimension → edge dimension
        self.layers.append(SAGELayer(ndim_in, edim, edim, activation))
        # Layer 2: Edge dimension → output dimension
        self.layers.append(SAGELayer(edim, edim, ndim_out, activation))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, nfeats, efeats):
        """
        Forward pass through SAGE layers.
        
        Args:
            g: DGL graph
            nfeats: Initial node features
            efeats: Initial edge features
            
        Returns:
            Final node and edge embeddings
        """
        for i, layer in enumerate(self.layers):
            if i != 0:
                nfeats = self.dropout(nfeats)
            nfeats, efeats = layer(g, nfeats, efeats)
        return nfeats.sum(1), efeats


class MLPPredictor(torch.nn.Module):
    """
    MLP-based edge classifier for flow graphs.
    
    Academic justification:
        Edge classification is performed by combining embeddings of source
        and destination endpoints with the edge's own features. This captures
        both structural context (who communicates) and content (what is exchanged).
    """
    def __init__(self, in_features, out_classes):
        """
        Initialize MLP predictor.
        
        Args:
            in_features: Node embedding dimension
            out_classes: Number of output classes
        """
        super().__init__()
        self.W = nn.Linear(in_features * 2, out_classes)

    def apply_edges(self, edges):
        """
        Compute edge scores by concatenating endpoint embeddings.
        
        Academic justification:
            Concatenation preserves directional information (src→dst) which
            is critical for distinguishing attack direction in network flows.
        """
        h_u = edges.src["h"]
        h_v = edges.dst["h"]
        score = self.W(torch.cat([h_u, h_v], 1))
        return {"score": score}

    def forward(self, graph, h):
        """
        Forward pass of edge classifier.
        
        Args:
            graph: DGL graph
            h: Node embeddings
            
        Returns:
            Edge classification scores
        """
        with graph.local_scope():
            graph.ndata["h"] = h
            graph.apply_edges(self.apply_edges)
            return graph.edata["score"]


class FlowGraphSAGE(torch.nn.Module):
    """
    Complete E-GraphSAGE model for flow graph edge classification.
    
    This model adapts the E-GraphSAGE architecture for flow graph representations
    where edges represent network flows and nodes represent endpoints (IP:Port).
    
    Academic justification:
        Flow graphs invert the heterogeneous structure: flows become edges rather
        than nodes. This tests whether attacks exploit node-centric vs edge-centric
        vulnerabilities in GNN architectures. E-GraphSAGE is well-suited because
        it already incorporates edge features during message passing.
    
    Architecture:
        1. Two-layer E-GraphSAGE encoder learns node embeddings
        2. Edge features updated at each layer based on connected nodes
        3. MLP classifier combines endpoint embeddings for edge prediction
    
    Usage:
        model = FlowGraphSAGE(ndim_in=128, ndim_out=128, edim=128,
                             activation=F.relu, dropout=0.2, num_classes=2)
        logits = model(graph, node_features, edge_features)
    """
    def __init__(
        self, ndim_in, ndim_out, edim, activation, dropout, num_classes=2
    ):
        """
        Initialize FlowGraphSAGE model.
        
        Args:
            ndim_in: Input node feature dimension
            ndim_out: Output node embedding dimension
            edim: Edge feature dimension
            activation: Activation function (e.g., torch.nn.functional.relu)
            dropout: Dropout probability
            num_classes: Number of output classes (default: 2 for binary classification)
        """
        super().__init__()
        self.gnn = SAGE(ndim_in, ndim_out, edim, activation, dropout)
        self.pred = MLPPredictor(ndim_out, num_classes)
        self.name = "Flow Graph E-GraphSAGE"
        self.criterion = None
        self.num_classes = num_classes

    def forward(self, g, nfeats, efeats):
        """
        Forward pass of FlowGraphSAGE.
        
        Args:
            g: DGL graph (homogeneous flow graph)
            nfeats: Node features [num_nodes, feature_dim]
            efeats: Edge features [num_edges, feature_dim]
            
        Returns:
            Edge classification logits [num_edges, num_classes]
        """
        h, h_edges = self.gnn(g, nfeats, efeats)
        return self.pred(g, h)

    def train_model(
        self,
        G: dgl.DGLGraph,
        epochs: int,
        use_ddp: bool = False,
        use_minibatching: bool = False,
        batch_size: int = 131072,
        micro_batch_size: int = 32768,
    ) -> dgl.DGLGraph:
        """
        Train the FlowGraphSAGE model.
        
        Academic justification:
            Training procedure identical to heterogeneous models ensures fair
            comparison. DDP and minibatching enable scaling to large graphs.
        
        Args:
            G: Flow graph with edge labels and masks
            epochs: Number of training epochs
            use_ddp: Whether to use DistributedDataParallel (multi-GPU)
            use_minibatching: Whether to use mini-batch training
            batch_size: Batch size for mini-batch training
            micro_batch_size: Micro-batch size for gradient accumulation
            
        Returns:
            Trained graph (with updated model state)
        """
        if use_ddp and torch.cuda.device_count() > 1:
            logging.info(f"Training on {torch.cuda.device_count()} GPUs")
            world_size = torch.cuda.device_count()

            # Ensure graph is on CPU before spawning processes
            G = G.to("cpu")

            shared_metrics = {
                "train_losses": torch.zeros(epochs).share_memory_(),
                "val_losses": torch.zeros(epochs).share_memory_(),
                "train_acc": torch.zeros(epochs).share_memory_(),
                "val_acc": torch.zeros(epochs).share_memory_(),
            }
            
            try:
                torch.multiprocessing.spawn(
                    train_worker,
                    args=(
                        world_size,
                        self,
                        G,
                        epochs,
                        shared_metrics,
                        use_minibatching,
                        batch_size,
                        micro_batch_size,
                    ),
                    nprocs=world_size,
                    join=True,
                )
                self.train_losses = shared_metrics["train_losses"].cpu().numpy().tolist()
                self.val_losses = shared_metrics["val_losses"].cpu().numpy().tolist()
                self.train_acc = shared_metrics["train_acc"].cpu().numpy().tolist()
                self.val_acc = shared_metrics["val_acc"].cpu().numpy().tolist()
                
                # Set criterion after DDP training
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                edge_label = (
                    G.edata["label"]
                    if len(G.edata["label"].shape) == 1
                    else G.edata["label"].argmax(1)
                )
                
                labels = edge_label.cpu().numpy()
                unique_label = np.unique(labels)
                class_weights = class_weight.compute_class_weight(
                    class_weight="balanced", classes=unique_label, y=labels
                )
                class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
                self.criterion = torch.nn.CrossEntropyLoss(weight=class_weights).to(device)
                
            except Exception as e:
                logging.error(f"Error in distributed training: {str(e)}")
                raise e

            return G
        else:
            logging.info("Training on single GPU or CPU")
            return train_worker(
                0,
                1,
                self,
                G,
                epochs,
                None,
                use_minibatching,
                batch_size,
                micro_batch_size,
            )

    def plot_losses(self) -> Tuple[Optional[plt.Figure], Optional[plt.Figure]]:
        """
        Plot training and validation losses/accuracy.
        
        Returns:
            Tuple of (loss_figure, accuracy_figure) or (None, None) if not trained
        """
        if hasattr(self, "train_losses"):
            fig1, ax1 = plt.subplots(figsize=(10, 5))
            ax1.plot(self.train_losses, label="Train Loss")
            ax1.plot(self.val_losses, label="Val Loss")
            ax1.legend()
            ax1.set_title("Losses")

            fig2, ax2 = plt.subplots(figsize=(10, 5))
            ax2.plot(self.train_acc, label="Train Accuracy")
            ax2.plot(self.val_acc, label="Val Accuracy")
            ax2.legend()
            ax2.set_title("Accuracy")

            plt.tight_layout()
            return fig1, fig2
        else:
            logging.warning(
                "No training metrics available. Please train the model first."
            )
            return None, None
    
    def evaluate(self, G: dgl.DGLGraph) -> dict:
        """
        Evaluate model performance on test set.
        
        Academic justification:
            Comprehensive evaluation metrics (accuracy, precision, recall, F1, AUC)
            enable fair comparison with heterogeneous models and analysis of
            class-specific performance.
        
        Args:
            G: Flow graph with test set annotations
            
        Returns:
            Dictionary with evaluation metrics:
            - loss: Test loss
            - classification_report: Per-class precision/recall/F1
            - roc_curves: ROC curves and AUC for each class
            - confusion_matrix: Confusion matrix
            - accuracy: Overall accuracy
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        G = G.clone().to(device)
        mask = G.edata["test_mask"].bool().to(device)
        labels = (
            G.edata["label"]
            if len(G.edata["label"].shape) == 1
            else G.edata["label"].argmax(dim=1)
        )
        labels = labels[mask]
        
        self.eval()
        with torch.no_grad():
            outputs = self(G, G.ndata["h"], G.edata["h"])[mask]
            loss = self.criterion(outputs, labels.to(device))
            _, predicted = torch.max(outputs, 1)

            # Classification report (weighted and unweighted)
            sample_weight = np.array(
                [self.criterion.weight[int(label)].item() for label in labels]
            )
            report_weighted = classification_report(
                labels.cpu(), predicted.cpu(), sample_weight=sample_weight
            )
            report = classification_report(labels.cpu(), predicted.cpu())
            
            # ROC curves for each class
            fpr, tpr, roc_auc = {}, {}, {}
            for i in range(self.num_classes):
                fpr[i], tpr[i], _ = roc_curve(
                    labels.cpu() == i, predicted.cpu() == i
                )
                roc_auc[i] = auc(fpr[i], tpr[i])
            
            # Confusion matrix
            conf_matrix = confusion_matrix(labels.cpu(), predicted.cpu())
            
            # Accuracy
            accuracy = (predicted == labels).sum().item() / len(labels)
            
            return {
                "loss": loss.item(),
                "classification_report": report,
                "classification_report_weighted": report_weighted,
                "roc_curves": {
                    i: {"fpr": fpr[i], "tpr": tpr[i], "auc": roc_auc[i]} 
                    for i in range(self.num_classes)
                },
                "confusion_matrix": conf_matrix,
                "accuracy": accuracy,
            }
    
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str, device: Optional[torch.device] = None):
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to saved model checkpoint
            device: Target device (auto-detected if None)
            
        Returns:
            Loaded FlowGraphSAGE model
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Extract model configuration from checkpoint
        config = checkpoint.get('config', {})
        model = cls(**config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        logging.info(f"Loaded FlowGraphSAGE from {checkpoint_path}")
        return model
    
    def save_checkpoint(self, save_path: str, config: dict):
        """
        Save model checkpoint with configuration.
        
        Args:
            save_path: Path to save checkpoint
            config: Model configuration dictionary
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': config,
            'name': self.name
        }, save_path)
        logging.info(f"Saved FlowGraphSAGE checkpoint to {save_path}")
