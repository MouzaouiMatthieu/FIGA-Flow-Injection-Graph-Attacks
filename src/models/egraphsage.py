# E-GraphSAGE implementation
# From https://github.com/waimorris/E-GraphSAGE/tree/master
# Includes mini-batch training, DPP, evaluation, and plotting utilities

import torch
import torch.nn as nn
import dgl
import dgl.function as fn
import matplotlib.pyplot as plt
import numpy as np
import logging
from typing import Tuple
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve
from sklearn.utils import class_weight
from .utils import get_model
from .training import train_worker

logger = logging.getLogger(__name__)


class SAGELayer(torch.nn.Module):
    def __init__(self, ndim_in, edims, ndim_out, activation):
        """
        Modified SAGELayer to include edge features in the message function.
        """
        super(SAGELayer, self).__init__()
        ### force to output fix dimensions
        self.W_msg = nn.Linear(ndim_in + edims, ndim_out)
        ### apply weight
        self.W_apply = nn.Linear(ndim_in + ndim_out, ndim_out)
        self.activation = activation

    def message_func(self, edges):
        """Message passing function

        Args:
            edges (EdgeBatch): instance of EdgeBatch class.
                               During message passing, DGL generates
                               it internally to represent a batch of edges.
                               It has three members src, dst and data to access
                               features of source nodes, destination nodes, and edges, respectively.

            edges.src: features of the source nodes in the batch of edges provided in input
            edges.data: features of the edges in the batch

        Returns:
            _type_: _description_
        """
        return {"m": self.W_msg(torch.cat([edges.src["h"], edges.data["h"]], 2))}

    def update_edge_features(self, g):
        """
        Updates edge features by averaging the embeddings of the source and destination nodes.
        Meaning, we have e_uv^(k) = 0.5 * (W_edge(h_u^(k)) + W_edge(h_v^(k)))
        """
        src_emb = g.ndata["h"][g.edges()[0]]
        dst_emb = g.ndata["h"][g.edges()[1]]
        g.edata["h"] = (src_emb + dst_emb) / 2

    def forward(self, g_dgl, nfeats, efeats):
        with g_dgl.local_scope():
            g = g_dgl
            g.ndata["h"] = nfeats
            g.edata["h"] = efeats

            """update_all() is a high-level API that merges message generation,
            message aggregation and node update in a single call,
            which leaves room for optimization as a whole.
            https://docs.dgl.ai/guide/message-api.html
            https://docs.dgl.ai/generated/dgl.DGLGraph.update_all.html
            """
            g.update_all(self.message_func, fn.mean("m", "h_neigh"))

            # Final node embedding at depth K, now it includes edge features (different from GraphSAGE)
            new_node_feats = self.activation(
                self.W_apply(torch.cat([g.ndata["h"], g.ndata["h_neigh"]], dim=2))
            )
            g.ndata["h"] = new_node_feats

            # Update edge features
            self.update_edge_features(g)
            return g.ndata["h"], g.edata["h"]


class SAGE(torch.nn.Module):
    def __init__(self, ndim_in, ndim_out, edim, activation, dropout):
        """
        Modified SAGE model to include edge features in the forward function. (so the dimensions need to be coherent). 
        """
        super(SAGE, self).__init__()
        self.layers = nn.ModuleList()
        # 2 SAGE Layers
        # K = 2 layers --> neighbour information is aggregated from a two-hop neighbourhood
        self.layers.append(SAGELayer(ndim_in, edim, edim, activation))
        self.layers.append(SAGELayer(edim, edim, ndim_out, activation))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, nfeats, efeats):
        for i, layer in enumerate(self.layers):
            if i != 0:
                nfeats = self.dropout(nfeats)
            # Update nodes and edges at layer k
            nfeats, efeats = layer(g, nfeats, efeats)
            ## debug statistics
            #edge_mean = efeats.mean().item()
            #edge_var = efeats.var().item()
            #print(f"Layer {i}: Mean edge embedding = {edge_mean}, Variance = {edge_var}")
        return nfeats.sum(1), efeats


class MLPPredictor(torch.nn.Module):
    def __init__(self, in_features, out_classes):
        super().__init__()
        self.W = nn.Linear(in_features * 2, out_classes)

    def apply_edges(self, edges):
        h_u = edges.src["h"]
        h_v = edges.dst["h"]
        # Final node embedding
        # Final output of the forward propagatiion stage in E-GraphSAGE
        # embedding of each edge 'uv' as concatenatiion of nodes 'u' and nodes 'v'
        score = self.W(torch.cat([h_u, h_v], 1))
        return {"score": score}

    def forward(self, graph, h):
        with graph.local_scope():
            graph.ndata["h"] = h
            graph.apply_edges(self.apply_edges)
            return graph.edata["score"]


class EGraphSAGE(torch.nn.Module):
    def __init__(
        self, ndim_in, ndim_out, edim, activation, dropout, final_softmax_dim=11
    ):
        super().__init__()
        self.gnn = SAGE(ndim_in, ndim_out, edim, activation, dropout)
        self.pred = MLPPredictor(ndim_out, final_softmax_dim)
        self.name = "E-GraphSAGE-DGL"
        self.criterion = None

    def forward(self, g, nfeats, efeats):
        h, h_edges = self.gnn(g, nfeats, efeats)
        return self.pred(g, h)

    def train_model(
        self,
        G: dgl.DGLGraph,
        epochs: int,
        use_ddp: int = False,
        use_minibatching: bool = False,
        batch_size: int = 131072,
        micro_batch_size: int = 32768,
    ) -> dgl.DGLGraph:
        """Trains the model for `epochs` epochs on the graph `G`. Can use DDP (distributes load over all available GPUs) and minibatching.

        Args:
            G: Graph to train on.
            epochs: Number of epochs to train for.
            use_ddp (optional): Whether or not to use DPP. Defaults to False.
            use_minibatching (optional): Whether or not to use mb. Defaults to False.
            batch_size (optional): Batch size. Defaults to 262_144 = 2**18.
            micro_batch_size (optional): Micro batch size. Defaults to 65536 = 2**16.

        Raises:
            e: Error during training.

        Returns:
            G: Graph.
        """
        if (
            use_ddp and torch.cuda.device_count() > 1
        ):  # If we want to distribute load over all available GPUs, and there are more than 1 GPU available.
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
            # Use spawn method for starting processes
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
                self.train_losses = (
                    shared_metrics["train_losses"].cpu().numpy().tolist()
                )
                self.val_losses = shared_metrics["val_losses"].cpu().numpy().tolist()
                self.train_acc = shared_metrics["train_acc"].cpu().numpy().tolist()
                self.val_acc = shared_metrics["val_acc"].cpu().numpy().tolist()
                
                # Set criterion after DDP training since it's not automatically transferred back
                # This is needed because spawned processes don't transfer model attributes back to main process
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                edge_label = (
                    G.edata["label"]
                    if len(G.edata["label"].shape) == 1
                    else G.edata["label"].argmax(1)
                )
                train_mask = G.edata["train_mask"].bool()
                
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

    def plot_losses(self) -> Tuple[plt.Figure, plt.Figure]:
        if hasattr(self, "train_losses"):  # The model has been trained already.
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
        """Evaluate the attack on the graph against the victim model (loss, classification report, ROC curves, confusion matrix, and accuracy).

        Args:
            G: The graph to use as input for the evaluation.

        Returns:
            dict: {
                loss: Loss of the model on G,
                classification_report: Classification report of the model on G,
                roc_curves: {i: {"fpr": fpr[i], "tpr": tpr[i], "auc": roc_auc[i]} for each class i},
                confusion_matrix: Confusion matrix,
                accuracy: Accuracy of the model on G
            }
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

            # Classification report
            sample_weight = np.array(
                [self.criterion.weight[int(label)].item() for label in labels]
            )
            report_weighted = classification_report(
                labels.cpu(), predicted.cpu(), sample_weight=sample_weight
            )
            report = classification_report(labels.cpu(), predicted.cpu())
            # ROC
            fpr, tpr, roc_auc = {}, {}, {}
            for i in range(11):
                fpr[i], tpr[i], _ = roc_curve(
                    labels.cpu() == i, predicted.cpu() == i
                )
                roc_auc[i] = auc(fpr[i], tpr[i])
            # Confusion matrix
            conf_matrix = confusion_matrix(labels.cpu(), predicted.cpu())
            # Accuracy
            accuracy = (predicted == labels).sum().item() / len(labels)
            return {
                "loss": loss,
                "classification_report": report,
                "roc_curves": {i: {"fpr": fpr[i], "tpr": tpr[i], "auc": roc_auc[i]} for i in range(11)},
                "confusion_matrix": conf_matrix,
            }
