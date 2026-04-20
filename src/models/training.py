# Training functions for GNN models

import torch
import dgl
import numpy as np
import os
import contextlib
import logging
from sklearn.utils import class_weight
from tqdm import tqdm
from typing import Dict
from .utils import prepare_graph_for_device, create_edge_dataloader, prepare_minibatch, compute_accuracy, get_model

logger = logging.getLogger(__name__)

def train_worker(
    rank: int,
    world_size: int,
    model: torch.nn.Module,
    G: dgl.DGLGraph,
    epochs: int,
    shared_metrics: Dict,
    use_minibatching: bool = False,
    batch_size: int = 131072,
    micro_batch_size: int = 32768,
    lr: float = 0.001,
    optimizer_name: str = "adam",
    weight_decay: float = 0.0,
) -> dgl.DGLGraph:
    """Pickable training worker function for DDP training. Allows for micro-batching.

    Args:
        rank: Rank of the worker.
        world_size: Total number of workers.
        category_weighting: bool = False,
        model: Model to train.
        G: Graph to train on.
        epochs: Total number of epochs.
        shared_metrics: Shared metrics dictionary for DDP.
        use_minibatching (optional): Whether or not the use mb. Defaults to False.
        batch_size (optional): Batch size if minibatching. Defaults to 262_144=2**18.
        micro_batch_size (optional): Micro batch size. Defaults to 65_536=2**16.

    Raises:
        e: Exception raised during training.

    Returns:
        G: Graph after training.
    """
    try:
        # Initialize process group
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

        # Setup device
        device = torch.device(f"cuda:{rank}")  # cuda:0, cuda:1, ...
        torch.cuda.set_device(device)

        # Prepare model for DDP
        # Initialize lazy modules in this child process before moving to GPU
        try:
            import dgl as _dgl
            tmp_g = _dgl.graph(([0, 1], [1, 0]))
            # infer feature dims from provided G
            try:
                n_last = G.ndata['h'].shape[-1]
            except Exception:
                n_last = 1
            try:
                e_last = G.edata['h'].shape[-1]
            except Exception:
                e_last = 1
            n_feats = torch.zeros((tmp_g.num_nodes(), n_last), dtype=torch.float32)
            e_feats = torch.zeros((tmp_g.num_edges(), e_last), dtype=torch.float32)
            with torch.no_grad():
                # run a dummy forward to initialize lazy parameters
                try:
                    model(tmp_g, n_feats, e_feats)
                except Exception:
                    # If dummy forward fails, continue; initialization may still succeed later
                    pass
        except Exception:
            # If DGL not available or something else fails, skip dummy forward
            pass

        model = model.to(device)
        logger.debug(f"Model on device {device}, init DPP")
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[rank], find_unused_parameters=False
        )

        if use_minibatching:
            edge_dataloader = create_edge_dataloader(G, batch_size)
        else:  # Full batch training, prepare the tensor here to avoid re doing it in the training loop.
            G = prepare_graph_for_device(G, device)
            node_features = G.ndata["h"].to(device)
            edge_features = G.edata["h"].to(device)

        edge_label = (
            G.edata["label"]
            if len(G.edata["label"].shape) == 1
            else G.edata["label"].argmax(1)
        )  # One-hot to class index
        train_mask = G.edata["train_mask"].bool().to(device)
        test_mask = G.edata["test_mask"].bool().to(device)

        # Calculate class weights
        labels = edge_label.cpu().numpy()
        unique_label = np.unique(labels)
        if unique_label.size < 2:
            class_weights = None
            logger.warning("Single-class labels detected; disabling class weights.")
        else:
            class_weights = class_weight.compute_class_weight(
                class_weight="balanced", classes=unique_label, y=labels
            )
            class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

        # Setup optimizer and loss
        if optimizer_name.lower() in ("adam", "adamw"):
            opt_cls = torch.optim.AdamW if optimizer_name.lower() == "adamw" else torch.optim.Adam
            optimizer = opt_cls(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name.lower() == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights).to(device)
        get_model(model).criterion = criterion
        # Training loop
        train_losses, val_losses = [], []
        train_acc, val_acc = [], []

        for epoch in tqdm(range(epochs)) if rank == 0 else range(epochs):
            model.train()
            optimizer.zero_grad()
            epoch_loss = 0
            epoch_acc = 0
            num_batches = 0
            if use_minibatching:
                accumulation_steps = (
                )  # Req. for micro-batching
                for idx, batch_edge_ids in enumerate(
                    tqdm(edge_dataloader, desc=f"Batch, device = {device}", leave=False)
                ):
                    # Create mini-batch subgraph
                    subg, target_edges = prepare_minibatch(G, batch_edge_ids, device)

                    # Get features
                    node_features = subg.ndata["h"]
                    edge_features = subg.edata["h"]

                    # Get labels for target edges from original graph
                    batch_labels = G.edata["label"][batch_edge_ids].to(device)
                    if len(batch_labels.shape) > 1:
                        batch_labels = batch_labels.argmax(1)

                    # AMP disabled: run in full precision
                    with contextlib.nullcontext():
                        # Forward pass on entire subgraph
                        all_pred = model(subg, node_features, edge_features)

                        # Only use predictions for target edges
                        pred = all_pred[target_edges]

                        # Calculate loss and backward pass
                        loss = criterion(pred, batch_labels) / accumulation_steps
                    loss.backward()
                    # Accumulate gradients, otherwise too long.
                    if (idx + 1) % accumulation_steps == 0 or idx == len(
                        edge_dataloader
                    ) - 1:
                        optimizer.step()
                        optimizer.zero_grad()
                    epoch_loss += loss.item()
                    epoch_acc += compute_accuracy(pred, batch_labels)
                    num_batches += 1

                    # Clear some memory
                    del loss, pred, all_pred
                    torch.cuda.empty_cache()
                loss_history = epoch_loss / num_batches
                acc_history = epoch_acc / num_batches
            else:  # Full batch training
                # Forward pass

                pred = model(G, node_features, edge_features)
                loss = criterion(pred[train_mask], edge_label[train_mask])

                # Backward pass
                loss.backward()
                optimizer.step()

                loss_history = loss.item()
                acc_history = compute_accuracy(pred[train_mask], edge_label[train_mask])

            # Validation (only on rank 0)
            if rank == 0:
                model.eval()

                if use_minibatching:
                    with torch.no_grad():
                        val_loss = 0
                        val_acc_val = 0
                        num_batches = 0
                        val_acc_total = 0
                        val_batches = 0
                        val_edges = torch.where(G.edata["test_mask"])[0]
                        val_edge_dataloader = torch.utils.data.DataLoader(
                            val_edges.cpu(), batch_size=batch_size, shuffle=False
                        )

                        for val_edge_ids in tqdm(
                            val_edge_dataloader, desc="Validation", leave=False
                        ):
                            subg, val_target_edges = prepare_minibatch(
                                G, val_edge_ids, device
                            )
                            all_val_pred = model(subg, subg.ndata["h"], subg.edata["h"])
                            val_pred = all_val_pred[val_target_edges]

                            val_labels = G.edata["label"][val_edge_ids].to(device)
                            if len(val_labels.shape) > 1:
                                val_labels = val_labels.argmax(1)

                            val_batch_loss = criterion(val_pred, val_labels)
                            val_loss += val_batch_loss.item()
                            val_acc_total += compute_accuracy(val_pred, val_labels)
                            val_batches += 1
                        avg_val_loss = val_loss / val_batches
                        val_losses.append(avg_val_loss)
                        val_acc.append(val_acc_total / val_batches)
                else:
                    with torch.no_grad():
                        val_pred = model(G, node_features, edge_features)
                        val_loss = criterion(val_pred[test_mask], edge_label[test_mask])
                        val_acc_val = compute_accuracy(
                            val_pred[test_mask], edge_label[test_mask]
                        )

                        val_losses.append(val_loss.item())
                        val_acc.append(val_acc_val)
                train_acc.append(acc_history)
                train_losses.append(loss_history)

                if shared_metrics is not None:
                    shared_metrics["train_losses"][epoch] = train_losses[-1]
                    shared_metrics["val_losses"][epoch] = val_losses[-1]
                    shared_metrics["train_acc"][epoch] = train_acc[-1]
                    shared_metrics["val_acc"][epoch] = val_acc[-1]
                if epoch % 200 == 0:
                    logger.info(
                        f"Epoch {epoch}: Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}"
                    )
                    logger.info(
                        f"Train Acc: {train_acc[-1]:.4f}, Val Acc: {val_acc[-1]:.4f}"
                    )

        # Save metrics on rank 0
        if rank == 0:
            get_model(model).train_losses = train_losses
            get_model(model).val_losses = val_losses
            get_model(model).train_acc = train_acc
            get_model(model).val_acc = val_acc

        # Synchronize before cleaning up
        torch.distributed.barrier()
        torch.cuda.synchronize(device)

    except Exception as e:
        logging.error(f"Error in worker {rank}: {str(e)}")
        raise e

    finally:
        # Cleanup
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        torch.cuda.empty_cache()
    return G


def train_worker_heterogeneous(
    rank: int,
    world_size: int,
    model: torch.nn.Module,
    G: dgl.DGLGraph,
    epochs: int,
    shared_metrics: Dict,
    use_minibatching: bool = False,
    batch_size: int = 131072,
    micro_batch_size: int = 32768,
    margin = 1e-8,
    patience = 5,
    lr: float = 0.001,
    optimizer_name: str = "adam",
    weight_decay: float = 1e-4,
    surrogate_fanout: bool = False,
    category_weighting: bool = False,
) -> dgl.DGLGraph:
    """Training worker function for heterogeneous graph models with DDP support.

    Args:
        rank: Rank of the worker.
        world_size: Total number of workers.
        model: Model to train.
        G: Graph to train on.
        epochs: Total number of epochs.
        shared_metrics: Shared metrics dictionary for DDP.
        use_minibatching (optional): Whether to use minibatching. Defaults to False.
        batch_size (optional): Batch size if minibatching. Defaults to 131072.
        micro_batch_size (optional): Micro batch size. Defaults to 32768.
        margin (optional): Convergence margin for early stopping. Defaults to 1e-5.
        patience (optional): Number of epochs with small changes before early stopping. Defaults to 5.

    Returns:
        G: Graph after training.
    """
    try:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

        # Setup device
        device = torch.device(f"cuda:{rank}")  # cuda:0, cuda:1, ...
        torch.cuda.set_device(device)

        # Prepare model for DDP
        model = model.to(device)
        logger.debug(f"Model on device {device}, init DPP")
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[rank], find_unused_parameters=True
        )

        if use_minibatching:
            train_idx = torch.nonzero(G.nodes['flow'].data['train_mask'], as_tuple=True)[0]
            val_idx = torch.nonzero(G.nodes['flow'].data['val_mask'], as_tuple=True)[0]
            
            # For DDP to work correctly, all workers must process the same number of batches.
            # Split indices evenly and drop the remainder to ensure synchronization.
            # Rationale: DDP requires all workers to call backward() the same number of times.
            total_samples = len(train_idx)
            samples_per_worker = (total_samples // world_size) // batch_size * batch_size
            
            if samples_per_worker < batch_size:
                # Dataset too small for minibatch DDP with this batch_size/world_size.
                # Fall back to full-graph training to avoid an empty dataloader.
                logger.warning(
                    f"Rank {rank}: samples_per_worker={samples_per_worker} < batch_size={batch_size} "
                    f"(total={total_samples}, world_size={world_size}). "
                    "Falling back to full-graph training for this run."
                )
                use_minibatching = False
            
        if use_minibatching:
            # Each worker gets equal number of samples (drop some to ensure divisibility)
            start_idx = rank * samples_per_worker
            end_idx = start_idx + samples_per_worker
            train_idx_worker = train_idx[start_idx:end_idx]
            
            logger.debug(f"Rank {rank}: Processing {len(train_idx_worker)} / {total_samples} training nodes")
            
            # Adapt sampler to model layers - get number of layers from model
            model_layers = getattr(get_model(model), 'n_layers', 2)
            # Surrogate training uses a smaller fanout: it only needs to approximate
            # the NIDS decision boundary, not maximise accuracy.
            # NIDS  4L: [20, 15, 10, 10] ~30 000 chains/node
            # Surr  4L: [5,  5,  5,  5]  ~625 chains/node (48x fewer)
            if surrogate_fanout:
                fanout = [5] * model_layers
            elif model_layers <= 3:
                fanout = [15] * model_layers
            else:
                fanout = [20, 15, 10] + [10] * (model_layers - 3)
            logger.debug(f"Using sampler with fanout {fanout} for {model_layers} layer model (surrogate={surrogate_fanout})")
            
            sampler = dgl.dataloading.MultiLayerNeighborSampler(fanout)
            train_dataloader = dgl.dataloading.DataLoader(
                G,
                {'flow': train_idx_worker},
                sampler,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                device=device,
                drop_last=True  # Ensure consistent batch count across workers
            )
            
            # Validation only on rank 0
            if rank == 0:
                val_dataloader = dgl.dataloading.DataLoader(
                    G,
                    {'flow': val_idx},
                    sampler,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=0,
                    device=device
                )
            else:
                val_dataloader = None
        else:
            G = prepare_graph_for_device(G, device)
            G = G.to(device)
        node_label = G.nodes['flow'].data['label'] if len(G.nodes['flow'].data['label'].shape) == 1 else G.nodes['flow'].data['label'].argmax(dim=1)
        train_mask = G.nodes['flow'].data['train_mask'].bool().to(device)
        val_mask = G.nodes['flow'].data['val_mask'].bool().to(device)
        
        # Compute class weights (balanced)
        train_labels = node_label[train_mask.cpu()].cpu().numpy()
        unique_labels = np.unique(train_labels)
        if unique_labels.size < 2:
            weights = None
            logger.warning("Single-class labels detected; disabling class weights.")
        else:
            weights = class_weight.compute_class_weight(
                'balanced',
                classes=unique_labels,
                y=train_labels,
            )
            weights = torch.tensor(weights, dtype=torch.float32).to(device)
        if weights is not None and category_weighting and 'category' in G.nodes['flow'].data:
            cat = G.nodes['flow'].data['category']
            if cat.dim() > 1:
                cat = cat.argmax(dim=1)
            train_cat = cat[train_mask].cpu().numpy()
            if len(train_cat) > 0:
                uc, _ = np.unique(train_cat, return_counts=True)
                cat_weights = class_weight.compute_class_weight(
                    'balanced', classes=uc, y=train_cat
                )
                cat_map = {int(k): float(v) for k, v in zip(uc, cat_weights)}
                train_labels = node_label[train_mask].cpu().numpy()
                class_scale = []
                for c in np.unique(train_labels):
                    idx = np.where(train_labels == c)[0]
                    if idx.size == 0:
                        class_scale.append(1.0)
                    else:
                        cat_for_class = train_cat[idx]
                        cw = np.array([cat_map.get(int(x), 1.0) for x in cat_for_class], dtype=np.float32)
                        class_scale.append(float(cw.mean()) if cw.size > 0 else 1.0)
                class_scale = np.array(class_scale, dtype=np.float32)
                class_scale = class_scale / class_scale.mean()
                weights = weights * torch.tensor(class_scale, dtype=torch.float32).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=weights).to(device)
        # Build optimizer according to requested type
        if optimizer_name.lower() in ("adam", "adamw"):
            opt_cls = torch.optim.AdamW if optimizer_name.lower() == "adamw" else torch.optim.Adam
            optimizer = opt_cls(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name.lower() == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        get_model(model).criterion = criterion
        train_losses, val_losses = [], []
        train_acc, val_acc = [], []
        # Training loop
        for epoch in tqdm(range(epochs)) if rank == 0 else range(epochs):
            model.train()
            optimizer.zero_grad()
            epoch_loss = 0
            epoch_acc = 0
            num_batches = 0
            loss_history = float('nan')
            acc_history = float('nan')
            
            # Initialize stop_training for all workers
            stop_training = torch.tensor([0], dtype=torch.int).to(device)
            
            if use_minibatching:
                # Minibatch training
                for input_nodes, output_nodes, blocks in train_dataloader:
                    optimizer.zero_grad()
                    
                    # Get input features for both node types
                    h = {
                        'flow': blocks[0].srcnodes['flow'].data['h'],
                        'endpoint': blocks[0].srcnodes['endpoint'].data['h']
                    }
                    
                    # Forward pass
                    logits = model(blocks, h)
                    labels = blocks[-1].dstnodes['flow'].data['label'] if len(blocks[-1].dstnodes['flow'].data['label'].shape) == 1 else blocks[-1].dstnodes['flow'].data['label'].argmax(dim=1)

                    loss = criterion(logits, labels)
                    
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    epoch_acc += compute_accuracy(logits, labels)
                    num_batches += 1
                
                if num_batches > 0:
                    loss_history = epoch_loss / num_batches
                    acc_history = epoch_acc / num_batches
            else:
                logits = model(G)
                loss = criterion(logits[train_mask], node_label[train_mask])
                
                loss.backward()
                optimizer.step()
                
                loss_history = loss.item()
                acc_history = compute_accuracy(logits[val_mask], node_label[val_mask])
            
            # Synchronize all workers after training phase completes
            # Rationale: Ensures all workers finish training before rank 0 starts validation
            # Prevents rank 1 from racing ahead while rank 0 is still validating
            torch.distributed.barrier()
            
            # Validation (only on rank 0, and only every 5 epochs to reduce overhead)
            # Rationale: With minibatching, validation can take as long as training itself
            # Validating every epoch causes significant slowdown for large datasets
            if rank == 0:
                if epoch % 5 == 0 or epoch == epochs - 1:
                    model.eval()
                    
                    if use_minibatching:
                        with torch.no_grad():
                            val_loss = 0
                            val_epoch_acc = 0
                            val_num_batches = 0
                            for input_nodes, output_nodes, blocks in val_dataloader:
                                h = {
                                    'flow': blocks[0].srcnodes['flow'].data['h'],
                                    'endpoint': blocks[0].srcnodes['endpoint'].data['h']
                                }
                                logits = model(blocks,h)
                                labels = blocks[-1].dstnodes['flow'].data['label'] if len(blocks[-1].dstnodes['flow'].data['label'].shape) == 1 else blocks[-1].dstnodes['flow'].data['label'].argmax(dim=1)
                                
                                val_loss += criterion(logits, labels).item()
                                val_epoch_acc += compute_accuracy(logits, labels)
                                val_num_batches += 1
                            
                            if val_num_batches > 0:
                                
                                val_loss /= val_num_batches
                                val_epoch_acc /= val_num_batches
                                val_losses.append(val_loss)
                                val_acc.append(val_epoch_acc)
                    
                    else:
                        with torch.no_grad():
                            logits = model(G)
                            loss = criterion(logits[val_mask], node_label[val_mask])
                            acc = compute_accuracy(logits[val_mask], node_label[val_mask])
                            val_losses.append(loss.item())
                            val_acc.append(acc)
                else:
                    # Skip validation this epoch - reuse last validation metrics
                    if len(val_losses) > 0:
                        val_losses.append(val_losses[-1])
                        val_acc.append(val_acc[-1])
                
                train_acc.append(acc_history)
                train_losses.append(loss_history)
                
                if shared_metrics is not None:
                    shared_metrics["train_losses"][epoch] = train_losses[-1]
                    shared_metrics["val_losses"][epoch] = val_losses[-1]
                    shared_metrics["train_acc"][epoch] = train_acc[-1]
                    shared_metrics["val_acc"][epoch] = val_acc[-1]
                
                # Check for early stopping only every 20 epochs, but always set stop_training
                # Rationale: All workers must participate in broadcast every epoch to avoid deadlock
                if epoch % 20 == 0:
                    logger.info(
                        f"Epoch {epoch}: Train Loss: {train_losses[-1]:.7f}, Test Loss: {val_losses[-1]:.7f}"
                    )
                    logger.info(
                        f"Train Acc: {train_acc[-1]:.7f}, Test Acc: {val_acc[-1]:.7f}"
                    )
                    # Check for early stopping only every 20 epochs
                    if len(val_losses) > patience:
                        recent_losses = val_losses[-patience:]
                        loss_diffs = [abs(recent_losses[i] - recent_losses[i-1]) for i in range(1, len(recent_losses))]
                        if all(diff < margin for diff in loss_diffs):
                            logger.debug(f"Early stopping triggered at epoch {epoch} (loss change < {margin})")
                            stop_training[0] = 1
                
            # Synchronize all workers after validation completes on rank 0
            # Rationale: Rank 0 performs validation while other ranks wait here
            # Without this barrier, rank 1 would proceed to all_reduce while rank 0 is still validating
            torch.distributed.barrier()
            
            # Broadcast stop signal to all workers (must happen every epoch for synchronization)
            # Both workers are now synchronized and ready to participate in the all_reduce
            torch.distributed.all_reduce(stop_training, op=torch.distributed.ReduceOp.SUM)
                
            # Check stop condition for all workers
            if stop_training.item() > 0:
                logger.debug(f"Rank {rank} received stop signal, terminating.")
                break
        # Save metrics on rank 0
        get_model(model).train_losses = train_losses
        get_model(model).val_losses = val_losses
        get_model(model).train_acc = train_acc
        get_model(model).val_acc = val_acc
        
        torch.distributed.barrier()
        torch.cuda.synchronize(device)
    except Exception as e:
        logger.error(f"Error in worker {rank}: {e}")
        raise e
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        torch.cuda.empty_cache()
    return G
