import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl
import numpy as np
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix, det_curve
import matplotlib.pyplot as plt
import os
import logging
from typing import Dict, Tuple, List, Optional
from tqdm import tqdm

from src.models.heterogeneous import HeteroGraphSAGE, HeteroGCN, HeteroGAT

logger = logging.getLogger(__name__)

def get_model_class(model_name: str):
    if model_name == "SAGE": return HeteroGraphSAGE
    if model_name == "GCN": return HeteroGCN
    if model_name == "GAT": return HeteroGAT
    raise ValueError(f"Unknown model name: {model_name}")

class ModelTrainer:
    def __init__(
        self,
        model_name: str,
        graph: dgl.DGLGraph,
        n_layers: int = 2,
        num_heads: int = 4,
        device: Optional[torch.device] = None,
        hidden_feats: int = 128,
        out_feats: int = 128,
        dropout: float = 0.5,
        lr: float = 0.01,
        weight_decay: float = 5e-4,
        epochs: int = 100,
        patience: int = 5,
        class_weight: Optional[torch.Tensor] = None,
    ):
        self.model_name = model_name
        self.graph = graph
        self.n_layers = n_layers
        self.num_heads = num_heads
        self.device = device if device is not None else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.hidden_feats = hidden_feats
        self.out_feats = out_feats
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.patience = patience

        # Infer input feature sizes
        try:
            self.in_feats_flow = self.graph.nodes['flow'].data['h'].shape[1]
        except Exception:
            self.in_feats_flow = None
        try:
            self.in_feats_endpoint = self.graph.nodes['endpoint'].data['h'].shape[1]
        except Exception:
            self.in_feats_endpoint = None

        # Infer number of classes
        if 'label' in self.graph.nodes['flow'].data:
            labels = self.graph.nodes['flow'].data['label']
            if labels.ndim > 1:
                self.num_classes = labels.shape[-1]
            else:
                self.num_classes = int(labels.max().item()) + 1
        else:
            self.num_classes = 2

        # Create model class and instantiate
        ModelClass = get_model_class(self.model_name)
        model_name_lower = self.model_name.lower()

        if model_name_lower == 'gat':
            num_heads = self.num_heads
        else:
            num_heads = None

        # Build model according to type
        if model_name_lower == 'gat':
            self.model = ModelClass(
                in_feats={'flow': self.in_feats_flow, 'endpoint': self.in_feats_endpoint},
                hidden_feats=self.hidden_feats,
                out_feats=self.out_feats,
                num_heads=num_heads,
                num_classes=self.num_classes,
                n_layers=self.n_layers,
            ).to(self.device)
        else:
            # SAGE, GCN, GIN, etc.
            self.model = ModelClass(
                in_feats=self.in_feats_flow,
                hidden_feats=self.hidden_feats,
                out_feats=self.out_feats,
                num_classes=self.num_classes,
                n_layers=self.n_layers,
                dropout=self.dropout,
            ).to(self.device)

        # Set default optimal batch sizes.
        # For this dataset we prefer full-batch training for most models
        # (SAGE/GCN/GIN/GTN). To enforce that, set `optimal_batch_size` to a
        # value larger than the number of flow nodes so the minibatch path is
        # skipped. Keep smaller defaults for GAT-style models which may benefit
        # from minibatching.
        try:
            num_flow_nodes = self.graph.num_nodes('flow')
            if model_name_lower == 'gat':
                self.model.optimal_batch_size = 1024 if torch.cuda.is_available() else max(1024, num_flow_nodes + 1)
            else:
                # Force full-batch by setting batch size larger than dataset
                self.model.optimal_batch_size = num_flow_nodes + 1
            self.model._optimal_batch_size_initialized = True
            try:
                os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
                os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', os.environ.get('PYTORCH_CUDA_ALLOC_CONF') + ',max_split_size_mb:64')
            except Exception:
                pass
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            logger.info(f"Model optimal_batch_size set to: {self.model.optimal_batch_size}")
        except Exception:
            pass
        self.class_weight = class_weight
        # Optimizer and loss
        try:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        except Exception:
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        cw = self.class_weight.to(self.device) if self.class_weight is not None else None
        self.criterion = nn.CrossEntropyLoss(weight=cw)
        
    def train(self):
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0

        train_losses = []
        val_losses = []

        logger.info(f"Starting training for {self.model_name}...")

        # Move graph and all associated tensors to the target device for full-graph training.
        # This must happen before building `inputs` so that feature tensors and model
        # weights reside on the same device.
        self.graph = self.graph.to(self.device)
        train_mask = self.graph.nodes['flow'].data['train_mask'].bool()
        val_mask   = self.graph.nodes['flow'].data['val_mask'].bool()
        labels     = self.graph.nodes['flow'].data['label']
        if labels.ndim > 1:
            labels = labels.argmax(dim=1)

        inputs = {
            'flow': self.graph.nodes['flow'].data['h'],
            'endpoint': self.graph.nodes['endpoint'].data['h']
        }
        # If the model defines an `optimal_batch_size`, perform external minibatch
        # training to avoid the model accumulating all batch outputs on GPU.
        # The decision is controlled solely by `optimal_batch_size` so model
        # implementations can opt-in/out by setting this attribute.
        use_minibatch_training = (
            (getattr(self.model, 'optimal_batch_size', None) is not None)
            and (self.graph.num_nodes('flow') > getattr(self.model, 'optimal_batch_size', 0))
        )

        if use_minibatch_training:
            logger.info(f"Using minibatch training with batch_size={self.model.optimal_batch_size}")
            # Build sampler and dataloader for flow nodes
            # Use neighbor sampling (approximate) to reduce per-batch memory and CPU cost.
            # Full-neighborhood sampling is exact but very expensive for large graphs.
            try:
                fanouts = [10] * max(1, getattr(self.model, 'n_layers', 2))
                sampler = dgl.dataloading.MultiLayerNeighborSampler(fanouts)
            except Exception:
                sampler = dgl.dataloading.MultiLayerFullNeighborSampler(self.model.n_layers)

            flow_indices = torch.arange(self.graph.num_nodes('flow'), device=self.device)
            # Use multiple workers to parallelize block construction (configurable)
            try:
                num_workers = min(8, (os.cpu_count() or 2))
                # DGL requires num_workers=0 when graph and indices are on CUDA
                try:
                    device_for_workers = next(self.model.parameters()).device
                    if getattr(device_for_workers, 'type', None) == 'cuda':
                        num_workers = 0
                except Exception:
                    # conservative fallback
                    num_workers = 0
            except Exception:
                num_workers = 0

            dataloader = dgl.dataloading.DataLoader(
                self.graph,
                {'flow': flow_indices},
                sampler,
                batch_size=self.model.optimal_batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=num_workers,
                use_uva=False,
            )

            # AMP disabled by user request: always use full precision training here
            use_amp = False
            scaler = None

            # Single global progress bar for all epochs to avoid many short bars
            try:
                batches_per_epoch = len(dataloader)
            except Exception:
                # Fallback if dataloader length is not available
                batches_per_epoch = (self.graph.num_nodes('flow') + max(1, self.model.optimal_batch_size - 1)) // max(1, self.model.optimal_batch_size)

            total_batches = batches_per_epoch * self.epochs
            pbar = tqdm(total=total_batches, desc=f"Training ({self.model.optimal_batch_size}b)", unit='batch')

            for epoch in range(self.epochs):
                self.model.train()
                epoch_loss_sum = 0.0
                epoch_count = 0

                # Iterate batches for this epoch
                for batch_idx, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
                    # Move blocks to device
                    blocks = [b.to(self.device) for b in blocks]

                    # Forward on this minibatch (FP32 only — AMP disabled)
                    batch_logits = self.model._forward_minibatch(blocks, inputs=None, return_attention=False)
                    if isinstance(batch_logits, dict):
                        batch_logits = batch_logits['flow']
                    out_flow_ids = output_nodes['flow'].to(self.device)
                    mask_local = train_mask[out_flow_ids]
                    if mask_local.sum() == 0:
                        continue
                    batch_labels = labels[out_flow_ids].to(self.device)
                    if batch_labels.ndim > 1:
                        batch_labels = batch_labels.argmax(dim=1)
                    loss = self.criterion(batch_logits[mask_local], batch_labels[mask_local])

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    epoch_loss_sum += loss.item() * mask_local.sum().item()
                    epoch_count += mask_local.sum().item()

                # Aggregate epoch loss
                avg_train_loss = epoch_loss_sum / max(1, epoch_count)
                train_losses.append(avg_train_loss)

                # Validation (minibatch eval without extra tqdm)
                self.model.eval()
                val_loss_sum = 0.0
                val_count = 0
                with torch.no_grad():
                    for batch_idx, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
                        blocks = [b.to(self.device) for b in blocks]
                        batch_logits = self.model._forward_minibatch(blocks, inputs=None, return_attention=False)
                        if isinstance(batch_logits, dict):
                            batch_logits = batch_logits['flow']

                        out_flow_ids = output_nodes['flow'].to(self.device)
                        mask_local = val_mask[out_flow_ids]
                        if mask_local.sum() == 0:
                            continue

                        batch_labels = labels[out_flow_ids].to(self.device)
                        if batch_labels.ndim > 1:
                            batch_labels = batch_labels.argmax(dim=1)

                        vloss = self.criterion(batch_logits[mask_local], batch_labels[mask_local])
                        val_loss_sum += vloss.item() * mask_local.sum().item()
                        val_count += mask_local.sum().item()

                avg_val_loss = val_loss_sum / max(1, val_count)
                val_losses.append(avg_val_loss)

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model_state = self.model.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

                if epoch % 10 == 0:
                    logger.debug(f"Epoch {epoch}: Train Loss {avg_train_loss:.4f}, Val Loss {avg_val_loss:.4f}")

                # Update global progress bar by number of batches processed this epoch
                try:
                    pbar.update(batches_per_epoch)
                except Exception:
                    # Best-effort: increment by 1 per batch if update failed earlier
                    pbar.update(batches_per_epoch)

            pbar.close()

        else:
            # Full-graph training (legacy behavior)
            for epoch in range(self.epochs):
                self.model.train()
                logits = self.model(self.graph, inputs)

                # SAGE/GCN return dict usually?
                if isinstance(logits, dict):
                    logits = logits['flow']

                loss = self.criterion(logits[train_mask], labels[train_mask])

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_losses.append(loss.item())

                # Validation
                self.model.eval()
                with torch.no_grad():
                    val_logits = self.model(self.graph, inputs)
                    if isinstance(val_logits, dict):
                        val_logits = val_logits['flow']
                    val_loss = self.criterion(val_logits[val_mask], labels[val_mask])
                    val_losses.append(val_loss.item())

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model_state = self.model.state_dict()
                        patience_counter = 0
                    else:
                        patience_counter += 1

                if patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

                if epoch % 10 == 0:
                    logger.debug(f"Epoch {epoch}: Train Loss {loss.item():.4f}, Val Loss {val_loss.item():.4f}")
                
        # Restore best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
            # If test mask exists, evaluate and log classification report + rates
            try:
                if 'test_mask' in self.graph.nodes['flow'].data:
                    try:
                        self.evaluate('post_train', '.')
                    except Exception as e:
                        logger.warning(f"Post-train evaluation failed: {e}")
            except Exception:
                pass

        return train_losses, val_losses

    def train_surrogate(
        self,
        teacher: "ModelTrainer",
    ) -> Tuple[List[float], List[float]]:
        """Train this model (surrogate) on ``self.graph`` using NIDS hard pseudo-labels.

        The *correct threat-model* for a surrogate attacker:

        1. The attacker has G_surr *without* true labels.
        2. It queries the trained NIDS (``teacher``) on every flow in G_surr to
           obtain **hard predicted labels** (argmax of NIDS logits).
        3. It trains its own model with standard cross-entropy on those pseudo-labels.

        True labels stored in ``self.graph`` under ``"label"`` are **never used**
        during training — they remain as ground-truth for post-hoc evaluation only.

        Args:
            teacher: A trained ``ModelTrainer`` whose model is the NIDS.

        Returns:
            ``(train_losses, val_losses)`` — lists of per-epoch losses.
        """
        # ── Step 1: query teacher on G_surr to obtain hard pseudo-labels ────
        teacher.model.eval()
        g_dev = self.graph.to(self.device)
        inputs_surr = {
            "flow":     g_dev.nodes["flow"].data["h"],
            "endpoint": g_dev.nodes["endpoint"].data["h"],
        }
        with torch.no_grad():
            t_out = teacher.model(g_dev, inputs_surr)
            t_logits = t_out["flow"] if isinstance(t_out, dict) else t_out  # [N_surr, C]

        # Hard pseudo-labels: argmax of NIDS predictions
        hard_pseudo = t_logits.argmax(dim=1)                       # [N]

        logger.info(
            f"[Surrogate] Teacher queried on {self.graph.num_nodes('flow'):,} surr flows. "
            f"Pseudo-label distribution: "
            f"{ {int(c): int((hard_pseudo == c).sum()) for c in hard_pseudo.unique()} }"
        )

        # ── Step 2: surrogate training loop with standard CE ─────────────────
        # Reuse g_dev (already on device) so model weights and tensors are co-located.
        self.graph = g_dev
        train_mask = self.graph.nodes["flow"].data["train_mask"].bool()
        val_mask   = self.graph.nodes["flow"].data["val_mask"].bool()

        best_val_loss  = float("inf")
        best_model_state = None
        patience_counter = 0
        train_losses: List[float] = []
        val_losses:   List[float] = []

        inputs_student = {
            "flow":     self.graph.nodes["flow"].data["h"],
            "endpoint": self.graph.nodes["endpoint"].data["h"],
        }

        logger.info(f"[Surrogate] Starting training with NIDS hard pseudo-labels …")
        for epoch in range(self.epochs):
            self.model.train()
            s_out = self.model(self.graph, inputs_student)
            s_logits = s_out["flow"] if isinstance(s_out, dict) else s_out  # [N, C]

            loss = F.cross_entropy(s_logits[train_mask], hard_pseudo[train_mask])

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_losses.append(loss.item())

            # ── Validation ───────────────────────────────────────────────────
            self.model.eval()
            with torch.no_grad():
                v_out    = self.model(self.graph, inputs_student)
                v_logits = v_out["flow"] if isinstance(v_out, dict) else v_out
                v_loss   = F.cross_entropy(v_logits[val_mask], hard_pseudo[val_mask])

            val_losses.append(v_loss.item())

            if v_loss < best_val_loss:
                best_val_loss    = v_loss
                best_model_state = self.model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                logger.info(f"[Surrogate] Early stopping at epoch {epoch}")
                break

            if epoch % 10 == 0:
                logger.debug(
                    f"[Surrogate] Epoch {epoch}: "
                    f"Train Loss {loss.item():.4f}  Val Loss {v_loss.item():.4f}"
                )

        if best_model_state:
            self.model.load_state_dict(best_model_state)
        return train_losses, val_losses
        
    def evaluate(self, dataset_name, exp_dir):
        self.model.eval()
        # Prepare masks/labels
        if 'test_mask' in self.graph.nodes['flow'].data:
            mask = self.graph.nodes['flow'].data['test_mask'].bool()
        elif 'val_mask' in self.graph.nodes['flow'].data:
            mask = self.graph.nodes['flow'].data['val_mask'].bool()
        else:
            mask = torch.ones(self.graph.num_nodes('flow'), dtype=torch.bool)

        labels = self.graph.nodes['flow'].data['label']
        if labels.ndim > 1:
            labels = labels.argmax(dim=1)

        inputs = {
            'flow': self.graph.nodes['flow'].data['h'],
            'endpoint': self.graph.nodes['endpoint'].data['h']
        }

        with torch.no_grad():
            logits = self.model(self.graph, inputs)
            if isinstance(logits, dict):
                logits = logits['flow']

            selected_logits = logits[mask]
            selected_labels = labels[mask]

            preds = selected_logits.argmax(dim=1).cpu().numpy()
            y_true = selected_labels.cpu().numpy()

            # For ROC/DET we need probabilities; handle binary vs multiclass
            probs_all = F.softmax(selected_logits, dim=1).cpu().numpy()

        # Classification report
        report_dict = classification_report(y_true, preds, output_dict=True)
        report_str = classification_report(y_true, preds)
        logger.info("Classification Report:\n" + report_str)

        # Confusion matrix and rates
        cm = confusion_matrix(y_true, preds)
        n_classes = cm.shape[0]

        if n_classes == 2:
            tn, fp, fn, tp = cm.ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            logger.info(f"FPR: {fpr:.4f}, FNR: {fnr:.4f}, TPR: {tpr:.4f}, TNR: {tnr:.4f}")
        else:
            # Per-class rates (macro-averaged)
            tprs, fprs = [], []
            for cls in range(n_classes):
                tp = cm[cls, cls]
                fn = cm[cls, :].sum() - tp
                fp = cm[:, cls].sum() - tp
                tn = cm.sum() - (tp + fp + fn)
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                tprs.append(tpr)
                fprs.append(fpr)
                logger.info(f"Class {cls}: TPR={tpr:.4f}, FPR={fpr:.4f}")
            # Macro averages
            tpr = float(sum(tprs) / len(tprs))
            fpr = float(sum(fprs) / len(fprs))
            fnr = 1.0 - tpr
            tnr = 1.0 - fpr
            logger.info(f"Macro TPR: {tpr:.4f}, Macro FPR: {fpr:.4f}, Macro FNR: {fnr:.4f}, Macro TNR: {tnr:.4f}")

        # Plots: only meaningful for binary; for multiclass we skip ROC/DET plotting
        try:
            if n_classes == 2:
                probs = probs_all[:, 1]
                self._plot_curves(y_true, probs, dataset_name, exp_dir)
            else:
                logger.info("Skipping ROC/DET plots for multiclass results.")
        except Exception as e:
            logger.warning(f"Failed to plot curves: {e}")

        return {
            'report': report_dict,
            'fpr': float(fpr), 'fnr': float(fnr), 'tpr': float(tpr), 'tnr': float(tnr)
        }

    def _plot_curves(self, y_true, y_probs, dataset_name, exp_dir):
        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 5))
        
        # Full ROC
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{self.model_name} ROC')
        plt.legend(loc="lower right")
        
        # Zoomed ROC (First 2% FPR)
        plt.subplot(1, 2, 2)
        plt.plot(fpr, tpr, color='darkorange', lw=2)
        plt.xlim([0.0, 0.02])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Zoomed ROC (0-2% FPR)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(exp_dir, f"{self.model_name}_roc.png"))
        plt.close()
        
        # DET Curve
        fpr_det, fnr_det, _ = det_curve(y_true, y_probs)
        plt.figure()
        plt.plot(fpr_det, fnr_det)
        plt.xlabel('False Positive Rate')
        plt.ylabel('False Negative Rate')
        plt.title(f'{self.model_name} DET Curve')
        plt.grid(True)
        plt.xscale('log')
        plt.yscale('log')
        plt.savefig(os.path.join(exp_dir, f"{self.model_name}_det.png"))
        plt.close()

    def save(self, path):
         os.makedirs(os.path.dirname(path), exist_ok=True)
         torch.save(self.model, path)
         logger.info(f"Model saved to {path}")
