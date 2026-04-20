import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def ensure_node_capacity(graph, ntype: str, new_idx: int) -> None:
    """Ensure node data tensors have capacity for index ``new_idx`` (inclusive).

    This is a best-effort pad operation that mirrors the behavior used in
    attack modules. It pads each per-node tensor along dim=0 with zeros so
    that assignment at index ``new_idx`` will succeed.
    """
    if graph is None:
        return
    data = graph.nodes[ntype].data
    for key, tensor in list(data.items()):
        try:
            needed = (new_idx + 1) - tensor.shape[0]
            if needed <= 0:
                continue
            pad_shape = (needed,) + tensor.shape[1:]
            pad = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
            data[key] = torch.cat([tensor, pad], dim=0)
        except Exception:
            # Best-effort: skip any attribute we cannot safely pad
            continue


def inject_cover_flow(
    graph,
    selected_features,
    feature_idx: int,
    label_benign: int,
    label_malicious: int,
    sender_endpoint_id: int,
    target_endpoint_id: int,
    pool_indices: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = "cpu",
    verbose: bool = False,
):
    """Inject a single cover flow node into ``graph`` and wire it to the
    provided sender and target endpoints.

    This helper preserves the exact per-field semantics used across attack
    implementations: sets `h` when present, writes `label`, `is_malicious`,
    and boolean masks (`train_mask`/`val_mask`/`test_mask`) when those
    attributes exist on the graph. Edges `depends_on` and `links_to` are
    created for sender and target.

    Returns a cover_info dict consistent with prior implementations.
    """
    if graph is None:
        raise RuntimeError("Graph is None")

    device = torch.device(device) if isinstance(device, (str,)) else device

    # Create flow node and ensure capacity
    graph.add_nodes(1, ntype="flow")
    new_flow_id = graph.num_nodes("flow") - 1
    ensure_node_capacity(graph, "flow", new_flow_id)

    # Set features if available
    try:
        if selected_features is not None and "h" in graph.nodes["flow"].data:
            try:
                graph.nodes["flow"].data["h"][new_flow_id] = selected_features.to(device)
            except Exception:
                # Fallback: try plain assignment
                graph.nodes["flow"].data["h"][new_flow_id] = selected_features
    except Exception:
        pass

    flow_data = graph.nodes["flow"].data

    # Try to determine the source flow id (if pool indices were provided)
    source_flow_id = None
    try:
        if pool_indices is not None and feature_idx < len(pool_indices):
            source_flow_id = int(pool_indices[feature_idx].item())
    except Exception:
        source_flow_id = None

    # Set label/malicious/masks as best-effort (preserve dtype/shape semantics)
    if "label" in flow_data:
        try:
            flow_data["label"][new_flow_id] = int(label_benign)
        except Exception:
            pass
    if "is_malicious" in flow_data:
        try:
            flow_data["is_malicious"][new_flow_id] = False
        except Exception:
            pass
    if "train_mask" in flow_data:
        try:
            flow_data["train_mask"][new_flow_id] = False
        except Exception:
            pass
    if "val_mask" in flow_data:
        try:
            flow_data["val_mask"][new_flow_id] = True
        except Exception:
            pass
    if "test_mask" in flow_data:
        try:
            flow_data["test_mask"][new_flow_id] = False
        except Exception:
            pass

    # Wire edges (Flow -> Endpoint :depends_on) and (Endpoint -> Flow :links_to)
    try:
        u_flow = torch.tensor([new_flow_id], device=device)
        v_sender = torch.tensor([sender_endpoint_id], device=device)
        v_target = torch.tensor([target_endpoint_id], device=device)

        # Sender connection
        graph.add_edges(u_flow, v_sender, etype="depends_on")
        graph.add_edges(v_sender, u_flow, etype="links_to")

        # Target connection (if different)
        if target_endpoint_id != sender_endpoint_id:
            graph.add_edges(u_flow, v_target, etype="depends_on")
            graph.add_edges(v_target, u_flow, etype="links_to")
    except Exception:
        # If edge wiring fails, continue — caller should check graph integrity
        pass

    # Prepare a compact feature snapshot for metadata (best-effort)
    feat_list = None
    try:
        if hasattr(selected_features, "detach"):
            feat_list = selected_features.detach().cpu().tolist()
        else:
            feat_list = list(map(float, selected_features))
    except Exception:
        feat_list = None

    cover_info = {
        "cover_flow_id": int(new_flow_id),
        "cover_pool_idx": int(feature_idx),
        "cover_source_flow_id": int(source_flow_id) if source_flow_id is not None else None,
        "cover_sender_endpoint_id": int(sender_endpoint_id),
        "cover_features": feat_list,
    }

    if verbose:
        logger.info(
            f"Added cover flow {new_flow_id} between Ep:{sender_endpoint_id} and Ep:{target_endpoint_id} (pool_idx={feature_idx}, source_flow_id={source_flow_id})"
        )

    return cover_info
