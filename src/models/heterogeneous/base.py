# Base class for heterogeneous graph models

import logging
from typing import Tuple

import dgl
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve
from sklearn.utils import class_weight

from ..utils import get_model

logger = logging.getLogger(__name__)


class BaseHeterogeneous(torch.nn.Module):
    """Base class for heterogeneous graph models with common functionality."""

    def __init__(self):
        super(BaseHeterogeneous, self).__init__()
        self.name = None
        self.train_losses = None
        self.val_losses = None
        self.train_acc = None
        self.val_acc = None
        self.criterion = None
        self.optimal_batch_size = None
        self._optimal_batch_size_initialized = False

    def set_optimal_batch_size(self, graph_or_num_nodes):
        if hasattr(graph_or_num_nodes, "num_nodes"):
            try:
                num_flow_nodes = graph_or_num_nodes.num_nodes("flow")
            except Exception:
                num_flow_nodes = None
        else:
            num_flow_nodes = graph_or_num_nodes

        if num_flow_nodes is None:
            self.optimal_batch_size = None
            return self.optimal_batch_size

        self.optimal_batch_size = 32768 if num_flow_nodes > 10_000_000 else None
        return self.optimal_batch_size

    def initialize_optimal_batch_size(self, graph_or_num_nodes):
        if getattr(self, "_optimal_batch_size_initialized", False):
            return self.optimal_batch_size

        self.set_optimal_batch_size(graph_or_num_nodes)
        self._optimal_batch_size_initialized = True
        return self.optimal_batch_size

    def train_model(
        self,
        G,
        epochs,
        use_ddp=False,
        use_minibatching=False,
        batch_size=131072,
        micro_batch_size: int = 32768,
        lr: float = 0.001,
        optimizer_name: str = "adam",
        weight_decay: float = 1e-4,
        surrogate_fanout: bool = False,
        category_weighting: bool = False,
    ):
        """Train the heterogeneous graph model."""
        from ..training import train_worker_heterogeneous

        if use_ddp and torch.cuda.device_count() > 1:
            logging.info(f"Training on {torch.cuda.device_count()} GPUs")
            world_size = torch.cuda.device_count()
            G = G.to("cpu")

            shared_metrics = {
                "train_losses": torch.zeros(epochs).share_memory_(),
                "val_losses": torch.zeros(epochs).share_memory_(),
                "train_acc": torch.zeros(epochs).share_memory_(),
                "val_acc": torch.zeros(epochs).share_memory_(),
            }

            try:
                torch.multiprocessing.spawn(
                    train_worker_heterogeneous,
                    args=(
                        world_size,
                        self,
                        G,
                        epochs,
                        shared_metrics,
                        use_minibatching,
                        batch_size,
                        micro_batch_size,
                        1e-3,
                        5,
                        lr,
                        optimizer_name,
                        weight_decay,
                        surrogate_fanout,
                        category_weighting,
                    ),
                    nprocs=world_size,
                    join=True,
                )

                self.train_losses = shared_metrics["train_losses"].cpu().numpy().tolist()
                self.val_losses = shared_metrics["val_losses"].cpu().numpy().tolist()
                self.train_acc = shared_metrics["train_acc"].cpu().numpy().tolist()
                self.val_acc = shared_metrics["val_acc"].cpu().numpy().tolist()

                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                node_label = (
                    G.nodes['flow'].data['label']
                    if len(G.nodes['flow'].data['label'].shape) == 1
                    else G.nodes['flow'].data['label'].argmax(dim=1)
                )
                train_mask = G.nodes['flow'].data['train_mask'].bool()
                weights = class_weight.compute_class_weight(
                    'balanced',
                    classes=np.unique(node_label[train_mask].cpu().numpy()),
                    y=node_label[train_mask].cpu().numpy(),
                )
                weights = torch.tensor(weights, dtype=torch.float32).to(device)
                self.criterion = torch.nn.CrossEntropyLoss(weight=weights)

            except Exception as e:
                logging.error(f"Error in distributed training: {str(e)}")
                raise e

            return G

        else:
            logging.info("Training on single GPU or CPU")
            return train_worker_heterogeneous(
                0,
                1,
                self,
                G,
                epochs,
                None,
                use_minibatching,
                batch_size,
                micro_batch_size,
                lr=lr,
                optimizer_name=optimizer_name,
                weight_decay=weight_decay,
                surrogate_fanout=surrogate_fanout,
                category_weighting=category_weighting,
            )

    def plot_losses(self) -> Tuple[plt.Figure, plt.Figure]:
        if hasattr(self, "train_losses"):
            fig1, ax1 = plt.subplots(figsize=(10, 5))
            ax1.plot(self.train_losses, label="Train Loss")
            ax1.plot(self.val_losses, label="Val Loss")
            ax1.legend()
            ax1.set_title("Losses")
            ax1.set_xlabel("Epochs")
            ax1.set_ylabel("Loss")
            ax1.grid()
            ax1.set_title(f"Losses per epochs, model = {self.name}")

            fig2, ax2 = plt.subplots(figsize=(10, 5))
            ax2.plot(self.train_acc, label="Train Accuracy")
            ax2.plot(self.val_acc, label="Val Accuracy")
            ax2.legend()
            ax2.set_title("Accuracy")
            ax2.set_xlabel("Epochs")
            ax2.set_ylabel("Accuracy")
            ax2.grid()
            ax2.set_title(f"Accuracy per epochs, model = {self.name}")
            plt.tight_layout()
            return fig1, fig2
        else:
            logging.warning("No training metrics available. Please train the model first.")
            return None, None

    def evaluate(self, G: dgl.DGLGraph) -> dict:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        G = G.clone().to(device)
        mask = G.nodes['flow'].data["test_mask"].bool().to(device)
        labels = (
            G.nodes['flow'].data['label']
            if len(G.nodes['flow'].data['label'].shape) == 1
            else G.nodes['flow'].data['label'].argmax(dim=1)
        )
        labels = labels[mask]
        self.eval()
        with torch.no_grad():
            outputs = self(G)
            loss = self.criterion(outputs[mask], labels.to(device))
            _, predicted = torch.max(outputs[mask], 1)

            sample_weight = np.array(
                [self.criterion.weight[int(label)].item() for label in labels]
            )
            report_weighted = classification_report(
                labels.cpu(), predicted.cpu(), sample_weight=sample_weight, digits=4
            )
            report = classification_report(labels.cpu(), predicted.cpu(), digits=4)

            fpr, tpr, roc_auc = {}, {}, {}
            for i in range(len(torch.unique(labels))):
                fpr[i], tpr[i], _ = roc_curve(labels.cpu() == i, predicted.cpu() == i)
                roc_auc[i] = auc(fpr[i], tpr[i])

            conf_matrix = confusion_matrix(labels.cpu(), predicted.cpu())
            accuracy = (predicted == labels).sum().item() / len(labels)
            return {
                "loss": loss.item(),
                "classification_report": report,
                "classification_report_weighted": report_weighted,
                "roc_curves": {i: {"fpr": fpr[i], "tpr": tpr[i], "auc": roc_auc[i]} for i in roc_auc},
                "confusion_matrix": conf_matrix,
                "accuracy": accuracy,
            }
