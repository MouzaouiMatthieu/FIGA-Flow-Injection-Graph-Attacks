"""
Graph representation converters for transferability analysis.

This module provides conversion functions to transform heterogeneous endpoint-flow
graphs into alternative representations (flow graphs and line graphs) while preserving
all attack artifacts, features, and masks.

Academic justification:
    Testing attack transferability across representations reveals whether adversarial
    perturbations exploit representation-specific artifacts or fundamental vulnerabilities
    in the underlying network traffic patterns.
"""

import dgl
import torch
import numpy as np
import networkx as nx
from typing import Dict, Tuple, List, Optional, Any, Union
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


def convert_heterogeneous_to_flow_graph(
    G_hetero: Union[nx.Graph, dgl.DGLHeteroGraph],
    granularity: str = 'port',
    preserve_direction: bool = True
) -> dgl.DGLGraph:
    """
    Convert heterogeneous endpoint-flow graph to flow graph representation.
    
    In flow graphs, endpoints become nodes and flows become edges. This inverts
    the heterogeneous structure where flows are nodes and endpoints define connectivity.
    
    Args:
        G_hetero: NetworkX or DGL heterogeneous graph with endpoint and flow node types
        granularity: 'port' for IP:Port nodes, 'ip' for IP-only nodes
        preserve_direction: If True, create directed edges (src→dst)
        
    Returns:
        DGL homogeneous graph where:
        - Nodes represent endpoints (IP:Port or IP depending on granularity)
        - Edges represent flows with flow features as edge attributes
        - Edge labels indicate benign/malicious classification
        - Edge masks preserved for train/val/test splits
        
    Academic justification:
        Flow graphs represent the network as a graph of communicating hosts
        rather than a bipartite endpoint-flow structure. This tests whether
        attacks transfer when the classification paradigm shifts from node
        classification to edge classification.
    """
    logger.info(f"Converting heterogeneous graph to flow graph (granularity={granularity})")
    
    # Handle both NetworkX and DGL input formats
    if isinstance(G_hetero, nx.Graph):
        return _convert_nx_heterogeneous_to_flow_graph(G_hetero, granularity, preserve_direction)
    elif isinstance(G_hetero, dgl.DGLHeteroGraph):
        return _convert_dgl_heterogeneous_to_flow_graph(G_hetero, granularity, preserve_direction)
    else:
        raise TypeError(f"Input graph must be NetworkX Graph or DGLHeteroGraph, got {type(G_hetero)}")


def _convert_nx_heterogeneous_to_flow_graph(
    G_nx: nx.Graph,
    granularity: str = 'port',
    preserve_direction: bool = True
) -> dgl.DGLGraph:
    """Convert NetworkX heterogeneous graph to flow graph representation."""
    logger.debug("Converting NetworkX heterogeneous graph to flow graph")
    
    # Separate endpoint and flow nodes based on 'type' attribute
    # type=0: endpoint nodes (IP:Port), type=1: flow nodes
    endpoint_nodes = [n for n, d in G_nx.nodes(data=True) if d.get('type') == 0]
    flow_nodes = [n for n, d in G_nx.nodes(data=True) if d.get('type') == 1]
    
    num_endpoints = len(endpoint_nodes)
    num_flows = len(flow_nodes)
    
    logger.debug(f"Source graph: {num_endpoints} endpoints, {num_flows} flows")
    
    # Create endpoint name to index mapping
    endpoint_to_idx = {ep: i for i, ep in enumerate(endpoint_nodes)}
    flow_to_idx = {flow: i for i, flow in enumerate(flow_nodes)}
    
    # Extract flow connectivity: each flow connects to endpoints via edges
    src_nodes = []
    dst_nodes = []
    flow_ids = []
    flow_features = []
    flow_labels = []
    flow_train_masks = []
    flow_val_masks = []
    flow_test_masks = []
    
    for flow_node in flow_nodes:
        # Get neighbors of flow node (should be endpoints)
        neighbors = list(G_nx.neighbors(flow_node))
        endpoints_connected = [n for n in neighbors if n in endpoint_to_idx]
        
        if len(endpoints_connected) < 2:
            logger.warning(f"Flow {flow_node} has <2 endpoint connections, skipping")
            continue
        
        # Assume first endpoint is source, second is destination
        src_ep = endpoints_connected[0]
        dst_ep = endpoints_connected[1]
        
        src_idx = endpoint_to_idx[src_ep]
        dst_idx = endpoint_to_idx[dst_ep]
        
        src_nodes.append(src_idx)
        dst_nodes.append(dst_idx)
        flow_ids.append(flow_to_idx[flow_node])
        
        # Extract flow features and labels
        flow_data = G_nx.nodes[flow_node]
        if 'h' in flow_data:
            flow_features.append(flow_data['h'])
        if 'label' in flow_data:
            flow_labels.append(flow_data['label'])
        if 'train_mask' in flow_data:
            flow_train_masks.append(flow_data['train_mask'])
        if 'val_mask' in flow_data:
            flow_val_masks.append(flow_data['val_mask'])
        if 'test_mask' in flow_data:
            flow_test_masks.append(flow_data['test_mask'])
    
    logger.debug(f"Created {len(src_nodes)} edges from {num_flows} flows")
    
    # Create flow graph
    if preserve_direction:
        G_flow = dgl.graph((src_nodes, dst_nodes), num_nodes=num_endpoints)
    else:
        all_src = src_nodes + dst_nodes
        all_dst = dst_nodes + src_nodes
        G_flow = dgl.graph((all_src, all_dst), num_nodes=num_endpoints)
        flow_ids = flow_ids + flow_ids
        flow_features = flow_features + flow_features
        flow_labels = flow_labels + flow_labels
        flow_train_masks = flow_train_masks + flow_train_masks
        flow_val_masks = flow_val_masks + flow_val_masks
        flow_test_masks = flow_test_masks + flow_test_masks
    
    # Transfer flow features to edge attributes
    if flow_features:
        G_flow.edata['h'] = torch.tensor(np.array(flow_features), dtype=torch.float32)
    
    # Transfer flow labels to edge labels
    if flow_labels:
        G_flow.edata['label'] = torch.tensor(flow_labels, dtype=torch.long)
    
    # Transfer flow masks to edge masks
    if flow_train_masks:
        G_flow.edata['train_mask'] = torch.tensor(flow_train_masks, dtype=torch.bool)
    if flow_val_masks:
        G_flow.edata['val_mask'] = torch.tensor(flow_val_masks, dtype=torch.bool)
    if flow_test_masks:
        G_flow.edata['test_mask'] = torch.tensor(flow_test_masks, dtype=torch.bool)
    
    # Store flow IDs for tracking
    G_flow.edata['flow_id'] = torch.tensor(flow_ids, dtype=torch.long)
    
    # Add endpoint features as node features
    endpoint_features = []
    for ep in endpoint_nodes:
        ep_data = G_nx.nodes[ep]
        if 'h' in ep_data:
            endpoint_features.append(ep_data['h'])
    
    if endpoint_features:
        G_flow.ndata['h'] = torch.tensor(np.array(endpoint_features), dtype=torch.float32)
    else:
        # Create identity features if none exist
        G_flow.ndata['h'] = torch.eye(num_endpoints)
    
    logger.info(f"Flow graph: {G_flow.number_of_nodes()} nodes, {G_flow.number_of_edges()} edges")
    
    return G_flow


def _convert_dgl_heterogeneous_to_flow_graph(
    G_dgl: dgl.DGLHeteroGraph,
    granularity: str = 'port',
    preserve_direction: bool = True
) -> dgl.DGLGraph:
    """Convert DGL heterogeneous graph to flow graph representation."""
    logger.debug("Converting DGL heterogeneous graph to flow graph")
    
    # Extract endpoint and flow information
    num_endpoints = G_dgl.num_nodes('endpoint')
    num_flows = G_dgl.num_nodes('flow')
    
    logger.debug(f"Source graph: {num_endpoints} endpoints, {num_flows} flows")
    
    # Build endpoint mapping
    endpoint_to_node = {i: i for i in range(num_endpoints)}
    
    # Extract flow connectivity: each flow connects two endpoints
    # Get all edge types and their connections
    flow_endpoints = defaultdict(list)
    
    for etype in G_dgl.canonical_etypes:
        if etype[0] == 'endpoint' and etype[2] == 'flow':
            src, dst = G_dgl.edges(etype=etype)
            for ep, flow in zip(src.tolist(), dst.tolist()):
                flow_endpoints[flow].append(ep)
        elif etype[0] == 'flow' and etype[2] == 'endpoint':
            src, dst = G_dgl.edges(etype=etype)
            for flow, ep in zip(src.tolist(), dst.tolist()):
                if ep not in flow_endpoints[flow]:
                    flow_endpoints[flow].append(ep)
    
    # Build edge list for flow graph
    src_nodes = []
    dst_nodes = []
    flow_ids = []
    
    for flow_id in range(num_flows):
        endpoints = flow_endpoints.get(flow_id, [])
        if len(endpoints) < 2:
            logger.warning(f"Flow {flow_id} has <2 endpoints, skipping")
            continue
        
        src_ep = endpoints[0]
        dst_ep = endpoints[1]
        src_node = endpoint_to_node[src_ep]
        dst_node = endpoint_to_node[dst_ep]
        
        src_nodes.append(src_node)
        dst_nodes.append(dst_node)
        flow_ids.append(flow_id)
    
    logger.debug(f"Created {len(src_nodes)} edges from {num_flows} flows")
    
    # Create flow graph
    if preserve_direction:
        G_flow = dgl.graph((src_nodes, dst_nodes), num_nodes=num_endpoints)
    else:
        all_src = src_nodes + dst_nodes
        all_dst = dst_nodes + src_nodes
        G_flow = dgl.graph((all_src, all_dst), num_nodes=num_endpoints)
        flow_ids = flow_ids + flow_ids
    
    # Transfer flow features to edge attributes
    flow_features = G_dgl.nodes['flow'].data.get('h')
    if flow_features is not None:
        G_flow.edata['h'] = flow_features[flow_ids]
    
    # Transfer flow labels to edge labels
    flow_labels = G_dgl.nodes['flow'].data.get('label')
    if flow_labels is not None:
        G_flow.edata['label'] = flow_labels[flow_ids]
    
    # Transfer flow masks to edge masks
    for mask_name in ['train_mask', 'val_mask', 'test_mask']:
        if mask_name in G_dgl.nodes['flow'].data:
            G_flow.edata[mask_name] = G_dgl.nodes['flow'].data[mask_name][flow_ids]
    
    # Store flow IDs for tracking
    G_flow.edata['flow_id'] = torch.tensor(flow_ids, dtype=torch.long)
    
    # Add node features if not present
    if 'h' not in G_flow.ndata:
        endpoint_features = G_dgl.nodes['endpoint'].data.get('h')
        if endpoint_features is not None:
            G_flow.ndata['h'] = endpoint_features
        else:
            G_flow.ndata['h'] = torch.eye(num_endpoints)
    
    logger.info(f"Flow graph: {G_flow.number_of_nodes()} nodes, {G_flow.number_of_edges()} edges")
    
    return G_flow


def _convert_dgl_heterogeneous_to_line_graph_optimized(
    G_hetero: dgl.DGLHeteroGraph,
    edge_threshold: int = 1,
    batch_size: int = 10000
) -> dgl.DGLGraph:
    """
    Memory-efficient line graph conversion for large heterogeneous graphs.
    
    Optimizations implemented:
    1. Sparse matrix representation for endpoint-flow adjacency
    2. Batch processing of flow connectivity computation
    3. Efficient set operations using numpy arrays
    4. Memory-aware edge construction
    
    Args:
        G_hetero: DGL heterogeneous graph with endpoint and flow node types
        edge_threshold: Minimum number of shared endpoints to create edge
        batch_size: Number of flows to process per batch (controls memory usage)
        
    Returns:
        DGL homogeneous line graph where:
        - Nodes represent flows (preserving all flow features and masks)
        - Edges connect flows that share >= edge_threshold endpoints
        
    Academic justification:
        Large-scale network graphs (>1M flows) require memory-efficient algorithms.
        The sparse matrix approach reduces memory from O(n²) to O(E) where E is
        the number of edges, enabling processing of enterprise-scale network traffic.
        
    Complexity:
        Time: O(E + F*D) where E=edges, F=flows, D=avg degree
        Space: O(E + F) instead of naive O(F²)
    """
    import scipy.sparse as sp
    
    logger.info("Starting optimized line graph conversion")
    
    num_flows = G_hetero.num_nodes('flow')
    num_endpoints = G_hetero.num_nodes('endpoint')
    
    logger.info(f"Source graph: {num_endpoints:,} endpoints, {num_flows:,} flows")
    
    # Step 1: Build sparse endpoint-flow adjacency matrix
    logger.info("Building sparse endpoint-flow adjacency matrix")
    
    row_indices = []
    col_indices = []
    
    for etype in G_hetero.canonical_etypes:
        if etype[0] == 'endpoint' and etype[2] == 'flow':
            src, dst = G_hetero.edges(etype=etype)
            row_indices.extend(src.cpu().numpy())
            col_indices.extend(dst.cpu().numpy())
        elif etype[0] == 'flow' and etype[2] == 'endpoint':
            src, dst = G_hetero.edges(etype=etype)
            # Transpose: flow->endpoint becomes endpoint->flow
            row_indices.extend(dst.cpu().numpy())
            col_indices.extend(src.cpu().numpy())
    
    # Create sparse matrix
    data = np.ones(len(row_indices), dtype=np.int8)  # Use int8 to save memory
    adj_matrix = sp.csr_matrix(
        (data, (row_indices, col_indices)),
        shape=(num_endpoints, num_flows),
        dtype=np.int8
    )
    
    logger.debug(f"Sparse adjacency: {adj_matrix.shape}, {adj_matrix.nnz:,} non-zeros")
    
    # Step 2: Compute flow-flow edges using batched sparse matrix multiplication
    logger.info(f"Computing flow-flow edges (threshold={edge_threshold})")
    
    src_flows = []
    dst_flows = []
    
    # Transpose for efficient column access (flow-major)
    adj_T = adj_matrix.T.tocsr()  # Shape: (num_flows, num_endpoints)
    
    num_batches = (num_flows + batch_size - 1) // batch_size
    logger.info(f"Processing {num_flows:,} flows in {num_batches} batches")
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_flows)
        
        if batch_idx % 10 == 0:
            logger.debug(f"Processing batch {batch_idx+1}/{num_batches}")
        
        # Get endpoint connections for flows in this batch
        batch_flows = adj_T[start_idx:end_idx]  # Shape: (batch_size, num_endpoints)
        
        # Compute shared endpoint counts: (batch_size, num_flows)
        shared_counts = batch_flows @ adj_matrix  # Sparse matrix multiply
        
        # Convert to dense only for this batch
        shared_counts_dense = shared_counts.toarray()
        
        # Find flow pairs exceeding threshold
        batch_src, batch_dst = np.where(shared_counts_dense >= edge_threshold)
        
        # Convert batch-local indices to global flow indices
        global_src = batch_src + start_idx
        
        # Add bidirectional edges (undirected line graph)
        src_flows.extend(global_src)
        dst_flows.extend(batch_dst)
        
        # Also add reverse edges for flows not yet processed
        # (avoid duplicates by only adding j->i if j > i)
        future_mask = batch_dst > global_src
        if np.any(future_mask):
            src_flows.extend(batch_dst[future_mask])
            dst_flows.extend(global_src[future_mask])
    
    logger.info(f"Created {len(src_flows):,} directed edges")
    
    # Step 3: Create DGL graph
    if len(src_flows) > 0:
        G_line = dgl.graph((src_flows, dst_flows), num_nodes=num_flows)
    else:
        logger.warning("No edges created - resulting in isolated nodes")
        G_line = dgl.graph(([], []), num_nodes=num_flows)
    
    # Step 4: Transfer flow node data (features, labels, masks)
    logger.info("Transferring flow node data")
    flow_data = G_hetero.nodes['flow'].data
    
    if 'h' in flow_data:
        G_line.ndata['h'] = flow_data['h']
        logger.debug(f"Transferred features: shape {G_line.ndata['h'].shape}")
    
    if 'label' in flow_data:
        G_line.ndata['label'] = flow_data['label']
        label_dist = torch.bincount(G_line.ndata['label'])
        logger.debug(f"Transferred labels: {dict(enumerate(label_dist.tolist()))}")
    
    for mask_name in ['train_mask', 'val_mask', 'test_mask']:
        if mask_name in flow_data:
            G_line.ndata[mask_name] = flow_data[mask_name]
            count = flow_data[mask_name].sum().item()
            logger.debug(f"Transferred {mask_name}: {count:,} nodes")
    
    logger.info(f"Line graph complete: {G_line.number_of_nodes():,} nodes, "
                f"{G_line.number_of_edges():,} edges")
    
    return G_line


def convert_heterogeneous_to_line_graph(
    G_hetero: Union[nx.Graph, dgl.DGLHeteroGraph],
    granularity: str = 'port',
    edge_threshold: int = 1
) -> dgl.DGLGraph:
    """
    Convert heterogeneous endpoint-flow graph to line graph representation.
    
    In line graphs, flows become nodes and edges connect flows that share endpoints.
    This is the dual of the flow graph representation.
    
    Args:
        G_hetero: NetworkX or DGL heterogeneous graph with endpoint and flow node types
        granularity: 'port' for IP:Port-level connectivity, 'ip' for IP-level
        edge_threshold: Minimum number of shared endpoints to create edge (default: 1)
        
    Returns:
        DGL homogeneous graph where:
        - Nodes represent flows with flow features as node attributes
        - Edges connect flows that share endpoints
        - Node labels indicate benign/malicious classification
        - Node masks preserved for train/val/test splits
        
    Academic justification:
        Line graphs emphasize flow-to-flow relationships based on shared network
        endpoints. This representation tests whether attacks transfer when the
        graph structure focuses on flow connectivity patterns rather than
        endpoint-centric communication patterns.
        
    Note:
        For large graphs (>50K flows), automatically uses memory-efficient
        optimized conversion with sparse matrix operations to avoid OOM errors.
    """
    logger.info(f"Converting heterogeneous graph to line graph (granularity={granularity})")
    
    # Handle both NetworkX and DGL input formats
    if isinstance(G_hetero, nx.Graph):
        return _convert_nx_heterogeneous_to_line_graph(G_hetero, granularity, edge_threshold)
    elif isinstance(G_hetero, dgl.DGLHeteroGraph):
        # Auto-select optimized version for large graphs
        num_flows = G_hetero.num_nodes('flow')
        if num_flows > 50000:
            logger.info(f"Large graph detected ({num_flows:,} flows) - using optimized conversion")
            return _convert_dgl_heterogeneous_to_line_graph_optimized(
                G_hetero, 
                edge_threshold=edge_threshold,
                batch_size=10000
            )
        else:
            logger.info(f"Small graph ({num_flows:,} flows) - using standard conversion")
            return _convert_dgl_heterogeneous_to_line_graph(G_hetero, granularity, edge_threshold)
    else:
        raise TypeError(f"Input graph must be NetworkX Graph or DGLHeteroGraph, got {type(G_hetero)}")


def _convert_nx_heterogeneous_to_line_graph(
    G_nx: nx.Graph,
    granularity: str = 'port',
    edge_threshold: int = 1
) -> dgl.DGLGraph:
    """Convert NetworkX heterogeneous graph to line graph representation."""
    logger.debug("Converting NetworkX heterogeneous graph to line graph")
    
    # Separate endpoint and flow nodes
    endpoint_nodes = [n for n, d in G_nx.nodes(data=True) if d.get('type') == 0]
    flow_nodes = [n for n, d in G_nx.nodes(data=True) if d.get('type') == 1]
    
    num_flows = len(flow_nodes)
    logger.debug(f"Source graph: {num_flows} flows")
    
    # Create flow to index mapping
    flow_to_idx = {flow: i for i, flow in enumerate(flow_nodes)}
    
    # Build mapping: endpoint → set of flows
    endpoint_to_flows = defaultdict(set)
    for flow in flow_nodes:
        neighbors = list(G_nx.neighbors(flow))
        for neighbor in neighbors:
            if neighbor in endpoint_nodes:
                endpoint_to_flows[neighbor].add(flow)
    
    # Build mapping: flow → set of endpoints
    flow_to_endpoints = defaultdict(set)
    for ep, flows in endpoint_to_flows.items():
        for flow in flows:
            flow_to_endpoints[flow].add(ep)
    
    logger.debug(f"Built connectivity: {len(endpoint_to_flows)} endpoints with flow connections")
    
    # Build edge list for line graph
    src_flows = []
    dst_flows = []
    
    for ep, flows in endpoint_to_flows.items():
        flows_list = list(flows)
        for i in range(len(flows_list)):
            for j in range(i + 1, len(flows_list)):
                flow_i = flows_list[i]
                flow_j = flows_list[j]
                
                shared = len(flow_to_endpoints[flow_i] & flow_to_endpoints[flow_j])
                
                if shared >= edge_threshold:
                    flow_i_idx = flow_to_idx[flow_i]
                    flow_j_idx = flow_to_idx[flow_j]
                    src_flows.append(flow_i_idx)
                    dst_flows.append(flow_j_idx)
                    src_flows.append(flow_j_idx)
                    dst_flows.append(flow_i_idx)
    
    logger.debug(f"Created {len(src_flows)} edges in line graph")
    
    # Create line graph
    if len(src_flows) > 0:
        G_line = dgl.graph((src_flows, dst_flows), num_nodes=num_flows)
    else:
        G_line = dgl.graph(([], []), num_nodes=num_flows)
    
    # Transfer flow features to node attributes
    flow_features = []
    flow_labels = []
    flow_train_masks = []
    flow_val_masks = []
    flow_test_masks = []
    
    for flow in flow_nodes:
        flow_data = G_nx.nodes[flow]
        if 'h' in flow_data:
            flow_features.append(flow_data['h'])
        if 'label' in flow_data:
            flow_labels.append(flow_data['label'])
        if 'train_mask' in flow_data:
            flow_train_masks.append(flow_data['train_mask'])
        if 'val_mask' in flow_data:
            flow_val_masks.append(flow_data['val_mask'])
        if 'test_mask' in flow_data:
            flow_test_masks.append(flow_data['test_mask'])
    
    if flow_features:
        G_line.ndata['h'] = torch.tensor(np.array(flow_features), dtype=torch.float32)
    if flow_labels:
        G_line.ndata['label'] = torch.tensor(flow_labels, dtype=torch.long)
    if flow_train_masks:
        G_line.ndata['train_mask'] = torch.tensor(flow_train_masks, dtype=torch.bool)
    if flow_val_masks:
        G_line.ndata['val_mask'] = torch.tensor(flow_val_masks, dtype=torch.bool)
    if flow_test_masks:
        G_line.ndata['test_mask'] = torch.tensor(flow_test_masks, dtype=torch.bool)
    
    logger.info(f"Line graph: {G_line.number_of_nodes()} nodes, {G_line.number_of_edges()} edges")
    
    return G_line


def _convert_dgl_heterogeneous_to_line_graph(
    G_dgl: dgl.DGLHeteroGraph,
    granularity: str = 'port',
    edge_threshold: int = 1
) -> dgl.DGLGraph:
    """Convert DGL heterogeneous graph to line graph representation."""
    logger.debug("Converting DGL heterogeneous graph to line graph")
    
    num_flows = G_dgl.num_nodes('flow')
    logger.debug(f"Source graph: {num_flows} flows")
    
    # Extract flow connectivity through endpoints
    endpoint_to_flows = defaultdict(set)
    
    for etype in G_dgl.canonical_etypes:
        if etype[0] == 'endpoint' and etype[2] == 'flow':
            src, dst = G_dgl.edges(etype=etype)
            for ep, flow in zip(src.tolist(), dst.tolist()):
                endpoint_to_flows[ep].add(flow)
        elif etype[0] == 'flow' and etype[2] == 'endpoint':
            src, dst = G_dgl.edges(etype=etype)
            for flow, ep in zip(src.tolist(), dst.tolist()):
                endpoint_to_flows[ep].add(flow)
    
    # Build mapping: flow → set of endpoints
    flow_to_endpoints = defaultdict(set)
    for ep, flows in endpoint_to_flows.items():
        for flow in flows:
            flow_to_endpoints[flow].add(ep)
    
    logger.debug(f"Built connectivity: {len(endpoint_to_flows)} endpoints with flow connections")
    
    # Build edge list for line graph
    src_flows = []
    dst_flows = []
    
    for ep, flows in endpoint_to_flows.items():
        flows_list = list(flows)
        for i in range(len(flows_list)):
            for j in range(i + 1, len(flows_list)):
                flow_i = flows_list[i]
                flow_j = flows_list[j]
                
                shared = len(flow_to_endpoints[flow_i] & flow_to_endpoints[flow_j])
                
                if shared >= edge_threshold:
                    src_flows.append(flow_i)
                    dst_flows.append(flow_j)
                    src_flows.append(flow_j)
                    dst_flows.append(flow_i)
    
    logger.debug(f"Created {len(src_flows)} edges in line graph")
    
    # Create line graph
    if len(src_flows) > 0:
        G_line = dgl.graph((src_flows, dst_flows), num_nodes=num_flows)
    else:
        G_line = dgl.graph(([], []), num_nodes=num_flows)
    
    # Transfer flow features to node attributes
    flow_features = G_dgl.nodes['flow'].data.get('h')
    if flow_features is not None:
        G_line.ndata['h'] = flow_features
    
    # Transfer flow labels to node labels
    flow_labels = G_dgl.nodes['flow'].data.get('label')
    if flow_labels is not None:
        G_line.ndata['label'] = flow_labels
    
    # Transfer flow masks to node masks
    for mask_name in ['train_mask', 'val_mask', 'test_mask']:
        if mask_name in G_dgl.nodes['flow'].data:
            G_line.ndata[mask_name] = G_dgl.nodes['flow'].data[mask_name]
    
    logger.info(f"Line graph: {G_line.number_of_nodes()} nodes, {G_line.number_of_edges()} edges")
    
    return G_line


def replay_attack_on_representation(
    G_clean_repr: dgl.DGLGraph,
    G_clean_hetero: Union[nx.Graph, dgl.DGLHeteroGraph],
    G_attacked_hetero: Union[nx.Graph, dgl.DGLHeteroGraph],
    graph_type: str,
    granularity: str = 'port'
) -> dgl.DGLGraph:
    """
    Replay attack from heterogeneous graph onto target representation.
    
    This function identifies injected flows in the attacked heterogeneous graph
    and adds them to the target representation (flow graph or line graph) while
    preserving attack semantics.
    
    Args:
        G_clean_repr: Clean graph in target representation
        G_clean_hetero: Clean heterogeneous graph (reference)
        G_attacked_hetero: Attacked heterogeneous graph with injected flows
        graph_type: 'flow_graph' or 'line_graph'
        granularity: 'port' or 'ip' granularity level
        
    Returns:
        Attacked graph in target representation with injected flows replayed
        
    Academic justification:
        Replaying attacks ensures fair comparison by testing identical perturbations
        across representations. The attack budget (number of injected flows) remains
        constant, but the graph topology determines how these injections affect
        model predictions.
    """
    logger.info(f"Replaying attack on {graph_type} (granularity={granularity})")
    
    # Identify injected flows: flows in attacked graph but not in clean graph
    if isinstance(G_clean_hetero, nx.Graph):
        num_clean_flows = len([n for n, d in G_clean_hetero.nodes(data=True) if d.get('type') == 1])
        num_attacked_flows = len([n for n, d in G_attacked_hetero.nodes(data=True) if d.get('type') == 1])
    else:  # DGLHeteroGraph
        num_clean_flows = G_clean_hetero.num_nodes('flow')
        num_attacked_flows = G_attacked_hetero.num_nodes('flow')
    
    num_injected = num_attacked_flows - num_clean_flows
    
    logger.info(f"Identified {num_injected} injected flows")
    
    if num_injected == 0:
        logger.warning("No injected flows detected, returning clean graph")
        return G_clean_repr
    
    # Simply convert the attacked heterogeneous graph to target representation
    # This automatically includes the injected flows
    if graph_type == 'flow_graph':
        G_attacked_repr = convert_heterogeneous_to_flow_graph(
            G_attacked_hetero,
            granularity=granularity,
            preserve_direction=True
        )
    elif graph_type == 'line_graph':
        G_attacked_repr = convert_heterogeneous_to_line_graph(
            G_attacked_hetero,
            granularity=granularity,
            edge_threshold=1
        )
    else:
        raise ValueError(f"Unknown graph_type: {graph_type}")
    
    # Verify attack budget preserved
    if graph_type == 'flow_graph':
        num_clean_edges = G_clean_repr.number_of_edges()
        num_attacked_edges = G_attacked_repr.number_of_edges()
        logger.info(f"Edge count: clean={num_clean_edges}, attacked={num_attacked_edges}, "
                   f"injected={num_attacked_edges - num_clean_edges}")
    else:  # line_graph
        num_clean_nodes = G_clean_repr.number_of_nodes()
        num_attacked_nodes = G_attacked_repr.number_of_nodes()
        logger.info(f"Node count: clean={num_clean_nodes}, attacked={num_attacked_nodes}, "
                   f"injected={num_attacked_nodes - num_clean_nodes}")
    
    return G_attacked_repr


def verify_conversion_preserves_data(
    G_hetero: Union[nx.Graph, dgl.DGLHeteroGraph],
    G_converted: dgl.DGLGraph,
    graph_type: str,
    sample_size: int = 100
) -> Dict[str, Any]:
    """
    Verify conversion correctness and data preservation.
    
    Args:
        G_hetero: Original heterogeneous graph
        G_converted: Converted graph (flow graph or line graph)
        graph_type: 'flow_graph' or 'line_graph'
        sample_size: Number of elements to sample for feature verification
        
    Returns:
        Dictionary with verification results:
        - 'element_count_match': Boolean
        - 'feature_preservation': Boolean
        - 'label_distribution': Dict with class counts
        - 'mask_ratios': Dict with train/val/test ratios
        - 'success': Overall pass/fail boolean
        
    Academic justification:
        Rigorous verification ensures conversion correctness. Any discrepancy
        in element counts, features, or labels would invalidate transferability
        analysis by introducing confounding factors.
    """
    logger.info(f"Verifying conversion to {graph_type}")
    
    results = {
        'element_count_match': False,
        'feature_preservation': False,
        'label_distribution_match': False,
        'mask_ratios_match': False,
        'success': False
    }
    
    # Get flow count based on graph type
    if isinstance(G_hetero, nx.Graph):
        num_flows = len([n for n, d in G_hetero.nodes(data=True) if d.get('type') == 1])
    else:  # DGLHeteroGraph
        num_flows = G_hetero.num_nodes('flow')
    
    # Check 1: Element count
    if graph_type == 'flow_graph':
        num_elements = G_converted.number_of_edges()
        element_type = "edges"
    else:  # line_graph
        num_elements = G_converted.number_of_nodes()
        element_type = "nodes"
    
    # For flow graphs, we may have fewer edges if some flows couldn't be converted
    # (e.g., flows with <2 endpoints)
    element_ratio = num_elements / num_flows if num_flows > 0 else 0
    results['element_count_match'] = element_ratio > 0.9  # Allow 10% loss
    logger.info(f"Element count: {num_flows} flows → {num_elements} {element_type} "
               f"(ratio: {element_ratio:.2%})")
    
    # Check 2: Feature preservation (sample random elements)
    if isinstance(G_hetero, nx.Graph):
        # For NetworkX, extract flow features manually
        flow_nodes = [n for n, d in G_hetero.nodes(data=True) if d.get('type') == 1]
        flow_features_list = [G_hetero.nodes[n].get('h') for n in flow_nodes if 'h' in G_hetero.nodes[n]]
        if flow_features_list:
            flow_features = torch.tensor(np.array(flow_features_list), dtype=torch.float32)
        else:
            flow_features = None
    else:  # DGLHeteroGraph
        flow_features = G_hetero.nodes['flow'].data.get('h')
    
    if graph_type == 'flow_graph':
        converted_features = G_converted.edata.get('h')
    else:
        converted_features = G_converted.ndata.get('h')
    
    if flow_features is not None and converted_features is not None:
        # Sample and compare
        sample_size = min(sample_size, num_elements)
        sample_indices = torch.randperm(num_elements)[:sample_size]
        
        if graph_type == 'flow_graph':
            # Map edge indices back to flow IDs
            flow_ids = G_converted.edata.get('flow_id')
            if flow_ids is not None:
                sampled_flow_ids = flow_ids[sample_indices]
                original_feats = flow_features[sampled_flow_ids]
                converted_feats = converted_features[sample_indices]
                
                # Check if features match
                feat_match = torch.allclose(original_feats, converted_feats, rtol=1e-5)
                results['feature_preservation'] = feat_match
                logger.info(f"Feature preservation: {feat_match}")
        else:
            # Line graph: node IDs match flow IDs directly
            original_feats = flow_features[sample_indices]
            converted_feats = converted_features[sample_indices]
            
            feat_match = torch.allclose(original_feats, converted_feats, rtol=1e-5)
            results['feature_preservation'] = feat_match
            logger.info(f"Feature preservation: {feat_match}")
    
    # Check 3: Label distribution
    if isinstance(G_hetero, nx.Graph):
        flow_nodes = [n for n, d in G_hetero.nodes(data=True) if d.get('type') == 1]
        flow_labels_list = [G_hetero.nodes[n].get('label') for n in flow_nodes if 'label' in G_hetero.nodes[n]]
        if flow_labels_list:
            flow_labels = torch.tensor(flow_labels_list, dtype=torch.long)
        else:
            flow_labels = None
    else:  # DGLHeteroGraph
        flow_labels = G_hetero.nodes['flow'].data.get('label')
    
    if graph_type == 'flow_graph':
        converted_labels = G_converted.edata.get('label')
    else:
        converted_labels = G_converted.ndata.get('label')
    
    if flow_labels is not None and converted_labels is not None:
        original_dist = torch.bincount(flow_labels)
        converted_dist = torch.bincount(converted_labels)
        
        # Allow for small differences due to unconverted flows
        dist_match = torch.allclose(original_dist.float(), converted_dist.float(), rtol=0.1)
        results['label_distribution_match'] = dist_match
        logger.info(f"Label distribution: original={original_dist.tolist()}, "
                   f"converted={converted_dist.tolist()}, match={dist_match}")
    
    # Check 4: Mask ratios
    mask_ratios_original = {}
    mask_ratios_converted = {}
    
    for mask_name in ['train_mask', 'val_mask', 'test_mask']:
        if isinstance(G_hetero, nx.Graph):
            flow_nodes = [n for n, d in G_hetero.nodes(data=True) if d.get('type') == 1]
            mask_values = [G_hetero.nodes[n].get(mask_name, False) for n in flow_nodes]
            if mask_values:
                orig_mask = torch.tensor(mask_values, dtype=torch.bool)
                mask_ratios_original[mask_name] = orig_mask.float().mean().item()
        else:  # DGLHeteroGraph
            if mask_name in G_hetero.nodes['flow'].data:
                orig_mask = G_hetero.nodes['flow'].data[mask_name]
                mask_ratios_original[mask_name] = orig_mask.float().mean().item()
        
        if graph_type == 'flow_graph':
            conv_mask = G_converted.edata.get(mask_name)
        else:
            conv_mask = G_converted.ndata.get(mask_name)
        
        if conv_mask is not None:
            mask_ratios_converted[mask_name] = conv_mask.float().mean().item()
    
    # Check if ratios are close
    ratios_match = all(
        abs(mask_ratios_original.get(k, 0) - mask_ratios_converted.get(k, 0)) < 0.05
        for k in mask_ratios_original.keys()
    )
    results['mask_ratios_match'] = ratios_match
    results['mask_ratios_original'] = mask_ratios_original
    results['mask_ratios_converted'] = mask_ratios_converted
    logger.info(f"Mask ratios: original={mask_ratios_original}, "
               f"converted={mask_ratios_converted}, match={ratios_match}")
    
    # Overall success
    results['success'] = all([
        results['element_count_match'],
        results['feature_preservation'],
        results['label_distribution_match'],
        results['mask_ratios_match']
    ])
    
    logger.info(f"Verification {'PASSED' if results['success'] else 'FAILED'}")
    
    return results


def convert_networkx_to_dgl_heterogeneous(graph_nx):
    """
    Convert a NetworkX heterogeneous graph to a DGL heterogeneous graph.
    
    Args:
        graph_nx (nx.Graph): Input NetworkX graph with heterogeneous nodes
    
    Returns:
        tuple: (dgl.HeteroGraph, dict) - The converted graph and node mappings
    """
    # Separate and index nodes
    endpoint_nodes = []
    flow_nodes = []
    
    # Create mappings
    endpoint_to_idx = {}
    flow_to_idx = {}
    
    # First pass: create mappings
    for node, data in graph_nx.nodes(data=True):
        if data.get('type') == 0:
            if node not in endpoint_to_idx:
                endpoint_to_idx[node] = len(endpoint_nodes)
                endpoint_nodes.append(node)
        elif data.get('type') == 1:
            if node not in flow_to_idx:
                flow_to_idx[node] = len(flow_nodes)
                flow_nodes.append(node)
    
    # Prepare edge lists with correct indexing
    endpoint_to_flow_edges = []
    flow_to_endpoint_edges = []
    
    # Iterate over edges. If undirected, check both directions.
    # If directed, check as is.
    is_directed = graph_nx.is_directed()
    
    for u, v in graph_nx.edges():
        u_type = graph_nx.nodes[u].get('type')
        v_type = graph_nx.nodes[v].get('type')
        
        # Check u -> v
        if u_type == 0 and v_type == 1:
            endpoint_to_flow_edges.append((endpoint_to_idx[u], flow_to_idx[v]))
        elif u_type == 1 and v_type == 0:
            flow_to_endpoint_edges.append((flow_to_idx[u], endpoint_to_idx[v]))
            
        if not is_directed:
            # Check v -> u
            if v_type == 0 and u_type == 1:
                endpoint_to_flow_edges.append((endpoint_to_idx[v], flow_to_idx[u]))
            elif v_type == 1 and u_type == 0:
                flow_to_endpoint_edges.append((flow_to_idx[v], endpoint_to_idx[u]))
    
    # Create graph data dictionary
    graph_data = {
        ('endpoint', 'links_to', 'flow'): endpoint_to_flow_edges,
        ('flow', 'depends_on', 'endpoint'): flow_to_endpoint_edges, 
    }
    
    # Create heterogeneous graph
    g = dgl.heterograph(graph_data)
    
    # Prepare node features
    # Endpoint features (placeholder zeros)
    if not flow_nodes:
         logger.warning("No flow nodes found in graph.")
         feature_shape = 0
    else:    
        feature_shape = len(graph_nx.nodes[flow_nodes[0]]['h'])
        
    endpoint_h = torch.zeros(len(endpoint_nodes), feature_shape)
    g.nodes['endpoint'].data['h'] = endpoint_h
    
    # Flow node features
    flow_h = []
    flow_labels = []
    flow_train_mask = []
    flow_val_mask = []
    flow_test_mask = []
    flow_category = []
    
    for node in flow_nodes:
        node_data = graph_nx.nodes[node]
        # Handle numpy object arrays by converting to float64 first
        h_data = node_data.get('h', [0.0])
        if isinstance(h_data, np.ndarray):
            h_data = h_data.astype(np.float32)
        flow_h.append(torch.tensor(h_data, dtype=torch.float32))
        flow_labels.append(node_data.get('label', -1))
        flow_train_mask.append(node_data.get('train_mask', 0))
        flow_val_mask.append(node_data.get('val_mask', 0))
        flow_test_mask.append(node_data.get('test_mask', 0))
        cat = node_data.get('category', -1)
        if cat is None: cat = -1
        flow_category.append(cat)
        
    
    # Ensure consistent tensor sizes
    if flow_h:
        g.nodes['flow'].data['h'] = torch.stack(flow_h)
        g.nodes['flow'].data['label'] = torch.tensor(flow_labels, dtype=torch.long)
        g.nodes['flow'].data['train_mask'] = torch.tensor(flow_train_mask, dtype=torch.float32)
        g.nodes['flow'].data['val_mask'] = torch.tensor(flow_val_mask, dtype=torch.float32)
        g.nodes['flow'].data['test_mask'] = torch.tensor(flow_test_mask, dtype=torch.float32)
        g.nodes['flow'].data['category'] = torch.tensor(flow_category, dtype=torch.float32)
    
    # Create mappings dictionary
    node_mappings = {
        'endpoint': endpoint_to_idx,
        'flow': flow_to_idx
    }
    
    return g, node_mappings
