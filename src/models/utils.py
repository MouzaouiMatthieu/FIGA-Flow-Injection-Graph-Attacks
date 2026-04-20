# Utility functions for GNN models

import torch
import dgl
import numpy as np
from typing import Tuple, Dict

from src.utils.logger import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

def get_model(model) -> torch.nn.Module:
    """Access the underlying model when using DDP (DPP.module) or directly return the model.

    Args:
        model : torch.nn.Module or DDP.

    Returns:
        The model.
    """
    if isinstance(
        model, (torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel)
    ):
        return model.module
    return model


def prepare_graph_for_device(G, device):
    """Prepare graph by moving it and its features to the specified device.
    
    Args:
        G: Input graph
        device: Target device
        
    Returns:
        G: Graph on the specified device with features moved
    """
    G = G.to(device)
    for ntype in G.ntypes:
        for key in G.nodes[ntype].data.keys():
            G.nodes[ntype].data[key] = G.nodes[ntype].data[key].to(device)
    return G


def create_edge_dataloader(
    G: dgl.DGLGraph, batch_size: int
) -> torch.utils.data.DataLoader:
    """Create a DataLoader for the edges in the graph, with a batch size.

    Args:
        G: Graph with edge features and train_mask.
        batch_size: Batch size for the DataLoader.

    Returns:
        train_sampler: Dataloader for the edges in the graph.
    """
    train_edges = torch.where(G.edata["train_mask"])[0]
    train_sampler = torch.utils.data.DataLoader(
        train_edges.cpu(), batch_size=batch_size, shuffle=True, drop_last=False
    )
    return train_sampler


def prepare_minibatch(
    G: dgl.DGLGraph, edge_ids: torch.Tensor, device: torch.device
) -> Tuple[dgl.DGLGraph, torch.Tensor]:
    """Create a subgraph for the given edge IDs and set the train

    Args:
        G: Graph.
        edge_ids: Edge IDs to create the subgraph for.
        device: Device to move the subgraph to.

    Returns:
        subg, subg_target_edges: subgraph and target edges.
    """
    # Get the nodes connected to these edges
    src, dst = G.find_edges(edge_ids)
    nodes = torch.cat([src, dst]).unique()

    # Create subgraph including connected nodes
    subg = G.subgraph(nodes, relabel_nodes=False, store_ids=True)

    # Get the local edge IDs in the subgraph that correspond to our target edges
    subg_target_edges = subg.edge_ids(src, dst, return_uv=False)

    # Create edge masks initialized to False
    subg.edata["train_mask"] = torch.zeros(subg.number_of_edges(), dtype=torch.bool)
    subg.edata["test_mask"] = torch.zeros(subg.number_of_edges(), dtype=torch.bool)

    # Set the target edges to True in train mask
    subg.edata["train_mask"][subg_target_edges] = True

    # Move to device
    subg = subg.to(device)
    return subg, subg_target_edges


def compute_accuracy(pred, labels):
    """Compute accuracy of predictions.
    
    Args:
        pred: Model predictions
        labels: Ground truth labels
        
    Returns:
        accuracy: Classification accuracy
    """
    return (pred.argmax(1) == labels).float().mean().item()


def create_node_dataloader(g, batch_size):
    """Create a dataloader for sampling nodes of type 'flow' while maintaining their relationships with 'endpoint' nodes.
    
    Args:
        g: Heterogeneous graph
        batch_size: Batch size for dataloader
        
    Returns:
        dataloader: Node dataloader for the graph
    """
    sampler = dgl.dataloading.MultiLayerNeighborSampler([15, 10])
    dataloader = dgl.dataloading.NodeDataLoader(
        g,
        {'flow': g.nodes('flow')}, 
        sampler,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    return dataloader


def apply_query_budget(
    G_surr: dgl.DGLGraph,
    query_budget: int,
    seed: int = 42,
    strategy: str = "chronological",
) -> dgl.DGLGraph:
    """Restrict G_surr to only query_budget flows for surrogate training.
    
    The attacker queries the first Q flows chronologically (by timestamp).
    Then splits those Q flows into train (80%) and val (20%) chronologically.
    
    Args:
        G_surr: Surrogate graph (all flows available for querying)
        query_budget: Number of flows to query (Q)
        seed: Not used for chronological split, kept for compatibility
    
    Returns:
        G_surr_restricted: Graph with train_mask and val_mask set
    """
    import copy
    import torch
    
    g = copy.deepcopy(G_surr)
    
    # Get ALL flow nodes in G_surr (the attacker can query any flow)
    all_flow_ids = torch.arange(g.num_nodes("flow"), device=g.device)
    n_available = len(all_flow_ids)
    
    logger.info(f"G_surr has {n_available} total flows available for querying")
    
    if query_budget >= n_available:
        logger.info(f"Query budget {query_budget} >= available flows {n_available}, using all.")
        sampled_ids = all_flow_ids
    else:
        if strategy == "random":
            g_cpu = torch.Generator(device=all_flow_ids.device)
            g_cpu.manual_seed(int(seed))
            sampled_ids = all_flow_ids[torch.randperm(n_available, generator=g_cpu, device=all_flow_ids.device)[:query_budget]]
        else:
            timestamps = g.nodes["flow"].data["timestamp"]
            sorted_indices = torch.argsort(timestamps)
            sampled_ids = sorted_indices[:query_budget]
    
    n_sampled = len(sampled_ids)
    if strategy == "random":
        logger.info(f"Selected {n_sampled} flows randomly")
    else:
        logger.info(f"Selected {n_sampled} flows chronologically (first {n_sampled} in time)")
    
    # Split chronologically: first 80% for training, last 20% for validation
    n_train = int(0.8 * n_sampled)
    train_ids = sampled_ids[:n_train]
    val_ids = sampled_ids[n_train:]
    
    # Set masks
    g.nodes["flow"].data["train_mask"] = torch.zeros(g.num_nodes("flow"), dtype=torch.bool, device=g.device)
    g.nodes["flow"].data["val_mask"] = torch.zeros(g.num_nodes("flow"), dtype=torch.bool, device=g.device)
    g.nodes["flow"].data["train_mask"][train_ids] = True
    g.nodes["flow"].data["val_mask"][val_ids] = True
    
    # Also store the full queried set
    g.nodes["flow"].data["surr_queried_mask"] = torch.zeros(g.num_nodes("flow"), dtype=torch.bool, device=g.device)
    g.nodes["flow"].data["surr_queried_mask"][sampled_ids] = True
    
    logger.info(f"Applied query budget {query_budget}: using {n_sampled}/{n_available} flows "
                f"({100*n_sampled/n_available:.1f}%)")
    logger.info(f"  Train (chronological first 80%): {len(train_ids)} flows")
    logger.info(f"  Val (chronological last 20%): {len(val_ids)} flows")
    
    return g


def pipeline_train_surrogate(
    nids_model: torch.nn.Module,
    G_surr: dgl.DGLGraph,
    model_name: str = "SAGE",
    hidden_dim: int = 120,
    out: int = 128,
    max_epochs: int = 500,
    n_layers: int = 4,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    aggr_type: str = "mean",
    lr: float = 0.001,
    optimizer_name: str = "adam",
    weight_decay: float = 1e-4,
    nids_tau: float = None,
    query_budget: int = None,
    query_strategy: str = "chronological",
    max_query_resample: int = 3,
    **kwargs,
):
    """Train a surrogate model on G_surr using hard pseudo-labels from a trained NIDS.

    Threat model:
      1. The attacker queries the NIDS on the first Q flows chronologically from G_surr.
      2. Splits those Q flows into train (80% oldest) and val (20% newest).
      3. Trains the surrogate with standard CE using the NIDS pseudo-labels.

    Args:
        nids_model: Trained NIDS ``torch.nn.Module`` (already on its device).
        G_surr:     Surrogate-partition DGL graph (CPU or GPU; moved internally).
        model_name: Architecture name.
        nids_tau:   Optional score threshold (probability of the malicious class).
        query_budget: Number of flows to query (Q). If None, uses all G_surr flows.
        hidden_dim, out, max_epochs, n_layers, aggr_type, lr, optimizer_name,
        weight_decay, **kwargs: Forwarded to the underlying training call.

    Returns:
        (surrogate_model, G_surr_pseudo): trained model and the graph with
        pseudo-labels stored under ``nodes['flow'].data['label']``.
    """
    num_heads = kwargs.pop("num_heads", 4)
    seed = kwargs.get("seed", 42)

    # ── Step 0/1: Apply query budget and query NIDS ──────────────────────────
    nids_model.eval()
    G_surr_base = G_surr
    hard_pseudo = None
    t_logits_masked = None
    query_mask = None
    g_dev = None
    for attempt in range(max_query_resample + 1):
        if query_budget is not None:
            strategy = query_strategy if attempt == 0 else "random"
            G_surr = apply_query_budget(G_surr_base, query_budget, seed=seed + attempt, strategy=strategy)
        else:
            G_surr = G_surr_base

        g_dev = G_surr.to(device)

        if "surr_queried_mask" in g_dev.nodes["flow"].data:
            query_mask = g_dev.nodes["flow"].data["surr_queried_mask"].bool()
        elif query_budget is not None:
            query_mask = (g_dev.nodes["flow"].data["train_mask"].bool() |
                          g_dev.nodes["flow"].data["val_mask"].bool())
        else:
            query_mask = torch.ones(g_dev.num_nodes("flow"), dtype=torch.bool, device=device)

        n_queried = query_mask.sum().item()
        logger.info(f"Querying NIDS on {n_queried} flows (budget={query_budget})")

        inputs_surr = {
            "flow":     g_dev.nodes["flow"].data["h"],
            "endpoint": g_dev.nodes["endpoint"].data["h"],
        }

        with torch.no_grad():
            t_out    = nids_model(g_dev, inputs_surr)
            t_logits = t_out["flow"] if isinstance(t_out, dict) else t_out

        t_logits_masked = t_logits[query_mask]

        if nids_tau is not None:
            probs_mal = torch.softmax(t_logits_masked, dim=1)[:, 1].cpu()
            hard_pseudo = (probs_mal >= nids_tau).long()
            logger.info(
                "Surrogate: NIDS queried on %d flows (tau=%.4f). Pseudo-label dist: %s",
                hard_pseudo.numel(), nids_tau,
                {int(c): int((hard_pseudo == c).sum()) for c in hard_pseudo.unique()},
            )
        else:
            hard_pseudo = t_logits_masked.argmax(dim=1).cpu()
            logger.info(
                "Surrogate: NIDS queried on %d flows (argmax). Pseudo-label dist: %s",
                hard_pseudo.numel(),
                {int(c): int((hard_pseudo == c).sum()) for c in hard_pseudo.unique()},
            )

        if hard_pseudo.unique().numel() >= 2 or query_budget is None:
            break
        if attempt < max_query_resample:
            logger.warning(
                "Degenerate pseudo-label distribution; resampling query flows (%d/%d)",
                attempt + 1,
                max_query_resample,
            )
    
    # ── Step 2: Assign pseudo-labels to the queried flows ────────────────────
    G_pseudo = g_dev.cpu()
    
    # Get indices of queried flows
    queried_indices = torch.where(query_mask.cpu())[0]
    
    # Initialize labels for all flows (default 0)
    if "label" not in G_pseudo.nodes["flow"].data:
        G_pseudo.nodes["flow"].data["label"] = torch.zeros(G_pseudo.num_nodes("flow"), dtype=torch.long)
    
    G_pseudo.nodes["flow"].data["label"][queried_indices] = hard_pseudo
    
    # Ensure train_mask and val_mask are set correctly (already set by apply_query_budget)
    if "train_mask" not in G_pseudo.nodes["flow"].data:
        # If no budget was applied, create default train/val split
        all_indices = torch.arange(G_pseudo.num_nodes("flow"))
        n_train = int(0.8 * len(all_indices))
        G_pseudo.nodes["flow"].data["train_mask"] = torch.zeros(G_pseudo.num_nodes("flow"), dtype=torch.bool)
        G_pseudo.nodes["flow"].data["val_mask"] = torch.zeros(G_pseudo.num_nodes("flow"), dtype=torch.bool)
        G_pseudo.nodes["flow"].data["train_mask"][:n_train] = True
        G_pseudo.nodes["flow"].data["val_mask"][n_train:] = True
    
    # ── Step 3: Build model and train via the standard backend ───────────────
    in_feats  = G_pseudo.nodes["flow"].data["h"].shape[1]
    n_classes = int(t_logits_masked.shape[1])
    
    # Print classification report on G_surr (pseudo-labels vs surrogate predictions after training)
    logger.info("=" * 60)
    logger.info("Surrogate Training - Initial Evaluation on G_surr")
    logger.info("=" * 60)
    
    # For now, just log the pseudo-label distribution
    unique, counts = torch.unique(hard_pseudo, return_counts=True)
    logger.info(f"Pseudo-label distribution on G_surr (queried flows):")
    for u, c in zip(unique.tolist(), counts.tolist()):
        logger.info(f"  Label {u}: {c} flows ({100*c/len(hard_pseudo):.1f}%)")
    
    # Build model
    if model_name == "SAGE":
        from src.models import HeteroGraphSAGE
        surr_model = HeteroGraphSAGE(
            in_feats, hidden_dim, out, n_classes, n_layers=n_layers, aggr_type=aggr_type, **kwargs
        )
    elif model_name == "GAT":
        from src.models import HeteroGAT
        surr_model = HeteroGAT(in_feats, hidden_dim, out, num_heads, n_classes, n_layers=n_layers, **kwargs)
    elif model_name == "GCN":
        from src.models import HeteroGCN
        surr_model = HeteroGCN(in_feats, hidden_dim, out, n_classes, n_layers=n_layers, **kwargs)
    else:
        raise ValueError(
            "Model %s not recognized. Choose from ['SAGE', 'GAT', 'GCN']"
            % model_name
        )

    surr_model = surr_model.to(device)
    
    # Train the model
    total_samples = len(queried_indices)
    
    if model_name == "GAT":
        surr_model.set_optimal_batch_size(G_pseudo)
        # Adjust batch size if total_samples is small
        batch_size = min(131072, max(32768, total_samples // 4))
        G_pseudo = surr_model.train_model(
            G_pseudo, max_epochs,
            use_ddp=total_samples > 50000,  # Only use DDP for large graphs
            use_minibatching=total_samples > 50000,
            batch_size=batch_size,
            lr=lr, optimizer_name=optimizer_name, weight_decay=weight_decay,
            surrogate_fanout=True,
        )
    else:
        G_pseudo = surr_model.train_model(
            G_pseudo, max_epochs,
            use_ddp=False,
            use_minibatching=False,
            lr=lr, optimizer_name=optimizer_name, weight_decay=weight_decay,
            surrogate_fanout=True,
        )
    
    # ── Step 4: Print classification reports after training ───────────────────
    logger.info("=" * 60)
    logger.info("Surrogate Training - Final Evaluation")
    logger.info("=" * 60)
    
    # Evaluate on G_surr validation set
    surr_model.eval()
    with torch.no_grad():
        g_eval = G_pseudo.to(device)
        out = surr_model(g_eval)
        logits = out["flow"] if isinstance(out, dict) else out
        
        # Get validation mask
        if "val_mask" in g_eval.nodes["flow"].data:
            val_mask = g_eval.nodes["flow"].data["val_mask"].bool()
        else:
            val_mask = torch.ones(g_eval.num_nodes("flow"), dtype=torch.bool, device=device)
        
        # Get pseudo-labels (NIDS predictions) for validation set
        pseudo_labels = g_eval.nodes["flow"].data["label"][val_mask]
        preds = logits[val_mask].argmax(dim=1)
        
        from sklearn.metrics import classification_report
        labels = [0, 1]
        report = classification_report(
            pseudo_labels.cpu().numpy(), 
            preds.cpu().numpy(),
            labels=labels,
            target_names=['Benign', 'Malicious'],
            zero_division=0
        )
        logger.info(f"Classification Report on G_surr (validation set):\n{report}")
    
    return surr_model, G_pseudo

def pipeline_train_dgl(
    dataset,
    model_name: str = "SAGE",
    hidden_dim: int = 120,
    out: int = 128,
    max_epochs: int = 500,
    n_layers: int = 4,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    aggr_type: str = "mean",
    lr: float = 0.001,
    optimizer_name: str = "adam",
    weight_decay: float = 1e-4,
    **kwargs,
):
    """Create and train a hetero DGL model with dataset-specific settings."""
    use_ip_port_init = False
    num_heads = kwargs.pop("num_heads", 4)

    if use_ip_port_init:
        logger.info("Using IP:Port initialization for %s model", model_name)
        G = dataset.get_dgl(0, ip_port_init=True)
    else:
        G = dataset.get_dgl(0)

    G = G.to(device)
    in_feats = G.nodes["flow"].data["h"].shape[1]
    n_classes = (
        G.nodes["flow"].data["label"].shape[-1]
        if len(G.nodes["flow"].data["label"].shape) > 1
        else len(G.nodes["flow"].data["label"].unique())
    )

    if model_name == "SAGE":
        from src.models import HeteroGraphSAGE

        model = HeteroGraphSAGE(
            in_feats, hidden_dim, out, n_classes, n_layers=n_layers, aggr_type=aggr_type, **kwargs
        )
    elif model_name == "GAT":
        from src.models import HeteroGAT
        model = HeteroGAT(in_feats, hidden_dim, out, num_heads, n_classes, n_layers=n_layers, **kwargs)
    elif model_name == "GCN":
        from src.models import HeteroGCN

        model = HeteroGCN(in_feats, hidden_dim, out, n_classes, n_layers=n_layers, **kwargs)
    else:
        raise ValueError(
            "Model %s not recognized. Choose from ['SAGE', 'GAT', 'GCN']"
            % model_name
        )

    model = model.to(device)

    if model_name == "GAT":
        optimal_bs = model.set_optimal_batch_size(G)
        batch_size = optimal_bs if optimal_bs is not None else kwargs.get("batch_size", 32768)
        use_minibatch = True
        use_ddp = True
        G = model.train_model(
            G,
            max_epochs,
            use_ddp=use_ddp,
            use_minibatching=use_minibatch,
            batch_size=batch_size,
            lr=lr,
            optimizer_name=optimizer_name,
            weight_decay=weight_decay,
        )
        return model, G

    use_minibatch = False
    use_ddp = False

    G = model.train_model(
        G,
        max_epochs,
        use_ddp=use_ddp,
        use_minibatching=use_minibatch,
        lr=lr,
        optimizer_name=optimizer_name,
        weight_decay=weight_decay,
        category_weighting=kwargs.get("category_weighting", False),
    )
    return model, G