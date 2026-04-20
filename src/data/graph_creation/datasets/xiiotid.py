"""
X-IIoTID Dataset Module for Heterogeneous Graph Construction.

This module implements the X-IIoTID dataset with three-way temporal split,
mirroring the CIC-IDS-2017 implementation.
"""

import os
import os.path as osp
import logging
import random
import numpy as np
import pandas
import torch
import networkx as nx
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch_geometric.data import Dataset

from src.data.processing import PreprocessingXIIOTID
from ..utils.splitting import temporal_three_way_split, temporal_three_way_split_stratified
from src.data.graph_creation import utils

logger = logging.getLogger(__name__)


def _build_partition_graph(
    edge_df: pandas.DataFrame,
    classes: str,
    scaler,
    cols_to_norm,
    label_names,
    partition_name: str,
    metadata: dict = None,
    seed: int = 42,
) -> nx.MultiGraph:
    """Build one partition NetworkX graph from a pre-split X-IIoTID DataFrame.

    All three partitions (train, surr, test) are scaled with the NIDS scaler
    fitted on G_train. The attacker submits raw flows to the NIDS, which
    applies its own scaler internally — no separate attacker scaler exists.

    Args:
        edge_df:        Pre-split partition DataFrame.
        classes:        ``"binary"`` or ``"category"``.
        scaler:         Fitted NIDS scaler to apply.
        cols_to_norm:   Columns to standardise.
        label_names:    Class name list.
        partition_name: ``"train"`` / ``"surr"`` / ``"test"``.
        metadata:       Additional metadata to attach to the graph.
        seed:           Random seed for reproducibility.

    Returns:
        nx.MultiGraph with metadata attributes.
    """
    edge_df = edge_df.copy()

    # Numeric coercion + drop NaN
    edge_df[list(cols_to_norm)] = edge_df[list(cols_to_norm)].apply(
        pandas.to_numeric, errors="coerce"
    )
    edge_df.dropna(subset=list(cols_to_norm), inplace=True)

    # Apply NIDS scaler — same for all three partitions
    logger.info(f"[{partition_name}] Applying NIDS scaler")
    edge_df[list(cols_to_norm)] = scaler.transform(edge_df[list(cols_to_norm)])

    # Feature vectors
    drop_cols = ["label", "sID", "dID", "category", "Timestamp", "Flow ID"]
    edge_df["h"] = edge_df.drop(columns=drop_cols, errors="ignore").apply(
        lambda row: row.tolist(), axis=1
    )

    # ── Temporal train/val/test masks ─────────────────────────────────────────
    n_flows = len(edge_df)
    cut = int(0.8 * n_flows)
    if partition_name == "test":
        train_masks = [False] * n_flows
        val_masks   = [False] * n_flows
        test_masks  = [True]  * n_flows
    else:
        train_masks = [i < cut for i in range(n_flows)]
        val_masks = [i >= cut for i in range(n_flows)]
        test_masks = [False] * n_flows

    edge_df = edge_df.reset_index(drop=True)
    edge_df["_train_mask"] = train_masks
    edge_df["_val_mask"]   = val_masks
    edge_df["_test_mask"]  = test_masks

    # ── Build NetworkX graph ──────────────────────────────────────────────────
    G = nx.MultiGraph()

    source_nodes = set(edge_df["sID"])
    dest_nodes   = set(edge_df["dID"])
    
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Get feature dimension from flows
    flow_feat_dim = len(edge_df["h"].iloc[0]) if len(edge_df) > 0 else 79
    
    # Local network prefixes used to mark internal endpoints
    local_prefixes = ["10.0.", "192.168", "172"]
    for ep in source_nodes | dest_nodes:
        # Create Gaussian N(0,1) features for endpoints
        endpoint_features = torch.randn(flow_feat_dim).tolist()
        is_internal = any(ep.startswith(p) for p in local_prefixes)

        G.add_node(
            ep,
            type=0,
            real_id=ep,
            h=endpoint_features,
            is_source=(ep in source_nodes),
            is_destination=(ep in dest_nodes),
            is_internal=is_internal,
        )

    for local_idx, (_, row) in enumerate(edge_df.iterrows()):
        flow_node = f"flow_{local_idx}"
        node_attrs = {
            "type":       1,
            "h":          row["h"],
            "label":      row["label"],
            "timestamp":  row["Timestamp"],
            "train_mask": row["_train_mask"],
            "val_mask":   row["_val_mask"],
            "test_mask":  row["_test_mask"],
        }
        if "category" in row and not pandas.isna(row.get("category", float("nan"))):
            node_attrs["category"] = row["category"]
        if "Flow ID" in row:
            node_attrs["flow_id"] = row["Flow ID"]

        G.add_node(flow_node, **node_attrs)
        G.add_edge(row["sID"], flow_node)
        # Also connect the flow to its destination endpoint so downstream
        # converters produce the ('flow','depends_on','endpoint') etype
        G.add_edge(flow_node, row["dID"])
    # ── Graph-level attributes ────────────────────────────────────────────────
    G.partition    = partition_name
    G.label_names  = label_names
    G.scaler       = scaler
    G.cols_to_norm = list(cols_to_norm)

    endpoint_nodes   = [n for n, d in G.nodes(data=True) if d.get("type") == 0]
    G.node_mapping   = {node: i for i, node in enumerate(endpoint_nodes)}
    local_prefixes   = ["10.0.", "192.168", "172"]
    G.local_indicator = torch.tensor(
        [any(n.startswith(p) for p in local_prefixes) for n in endpoint_nodes]
    )
    
    # Attach metadata
    if metadata:
        G.metadata = metadata

    logger.debug(
        f"[{partition_name}] Graph: {len(endpoint_nodes)} endpoints, "
        f"{sum(1 for _, d in G.nodes(data=True) if d.get('type') == 1)} flows"
    )
    assert not G.is_directed()
    return G


def pipeline_create_heterogeneous_xiiotid(
    path_raw: str,
    classes: str,
    seed: int = 42,
    train_frac: float = 0.4,
    surr_frac: float = 0.4,
    stratify_attack: bool = False,
) -> tuple:
    """
    Full preprocessing pipeline → three independent heterogeneous NetworkX graphs.

    The dataset is sorted by timestamp and cut into three **contiguous**
    partitions (stratified by class):

    * **G_train** (``train_frac``) — NIDS victim training graph.
      The NIDS scaler is *fitted* on this partition.
    * **G_surr**  (``surr_frac``) — attacker surrogate graph.
      Scaled with the NIDS scaler: the attacker submits flows to the NIDS,
      which applies its own scaler internally before inference.
    * **G_test**  (remainder)    — evaluation / attack execution graph.
      Scaled by the NIDS scaler.

    Each graph has locally-remapped endpoint node IDs; cross-graph endpoint
    matching is done via the ``real_id`` node attribute (``IP:port`` string).

    Args:
        path_raw:        Path to directory containing ``X-IIoTID.csv``.
        classes:         ``"binary"`` or ``"category"``.
        seed:            Random seed for reproducibility.
        train_frac:      Fraction of flows for NIDS training (default 0.4).
        surr_frac:       Fraction of flows for attacker surrogate (default 0.4).
        stratify_attack: When ``True``, stratify on the fine-grained ``"Attack"``
                         string column (e.g. ``"DDoS"``, ``"Ransomware"``) even in
                         binary mode.
            stratify_attack: When ``True``, perform stratified temporal split (by Attack/label).
                             When ``False``, perform pure chronological split.

    Returns:
        ``(G_train, G_surr, G_test)`` — three ``nx.MultiGraph`` objects.
    """
    # Set seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    logger.debug("Preprocessing X-IIoTID data")
    df = PreprocessingXIIOTID(path_raw, classes).df

    # ── Stage 1: Remove constant features ────────────────────────────────────
    constant_features = df.columns[df.nunique() <= 1]
    df.drop(columns=constant_features, inplace=True)
    logger.debug(f"Removed {len(constant_features)} constant features")

    # ── Stage 2: Remove rare binary features (<0.1%) ─────────────────────────
    rare_threshold = 0.001
    rare_features = [
        col for col in df.columns
        if df[col].nunique() == 2
        and df[col].value_counts(normalize=True).min() < rare_threshold
    ]
    logger.info(f"Removing {len(rare_features)} rare binary features (< {rare_threshold*100}% occurrence)")
    df.drop(columns=rare_features, inplace=True)

    # ── Stage 3: Quantile-based outlier removal ───────────────────────────────
    max_std      = 100
    max_quantile = 0.99
    max_iters    = 5
    it           = 0
    n_before     = df.shape[0]

    numeric_cols      = df.select_dtypes(include=[float, int]).columns
    excluded_patterns = ["Port", "IP", "Label", "Attack", "category", "sID", "dID"]
    feature_cols      = [
        col for col in numeric_cols
        if not any(pattern in col for pattern in excluded_patterns)
    ]

    stds_original          = pandas.Series({col: df[col].std() for col in feature_cols})
    high_var_features_init = stds_original[stds_original > max_std].index.tolist()

    if not high_var_features_init:
        logger.info(f"No high-variance features detected (threshold: std > {max_std})")
    else:
        quantile_thresholds = {
            col: df[col].quantile(max_quantile) for col in high_var_features_init
        }
        logger.info(f"Identified {len(high_var_features_init)} high-variance features")

        while it < max_iters:
            stds_current          = pandas.Series({col: df[col].std() for col in feature_cols if col in df.columns})
            high_var_current      = stds_current[stds_current > max_std].index.tolist()

            if not high_var_current:
                logger.info(f"[Iter {it+1}] Convergence achieved")
                break

            mask_outlier = pandas.Series(False, index=df.index)
            for col in high_var_current:
                if col not in quantile_thresholds:
                    quantile_thresholds[col] = df[col].quantile(max_quantile)
                q_thresh    = quantile_thresholds[col]
                outlier_col = df[col] > q_thresh
                if outlier_col.sum() > 0:
                    logger.info(f"  {col}: flagging {outlier_col.sum()} samples above {q_thresh:.2f}")
                    mask_outlier |= outlier_col

            if mask_outlier.sum() == 0:
                break
            df = df[~mask_outlier]
            logger.info(f"[Iter {it+1}] Removed {mask_outlier.sum()} rows (remaining: {df.shape[0]})")
            it += 1

    n_after = df.shape[0]
    logger.info(f"Outlier removal: {n_before - n_after} rows removed ({100*(n_before-n_after)/n_before:.1f}%)")

    # ── Stage 4: Create endpoint identifiers (IP:Port) ───────────────────────
    df["sID"] = df["Scr_IP"] + ":" + df["Scr_port"].astype(str)
    df["dID"] = df["Des_IP"] + ":" + df["Des_port"].astype(str)
    df.drop(columns=["Scr_IP", "Scr_port", "Des_IP", "Des_port"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ── Stage 5: Boolean → int ────────────────────────────────────────────────
    bool_columns = df.select_dtypes(include=["bool"]).columns
    if len(bool_columns) > 0:
        df[bool_columns] = df[bool_columns].astype(int)

    # ── Stage 6: Label schema ─────────────────────────────────────────────────
    if classes == "category":
        label_columns = [col for col in df.columns if col.startswith("class2")]
        label_names   = label_columns
    elif classes == "binary":
        label_columns    = "class3"
        label_names      = ["Benign", "Malicious"]
        label_categories = "Attack"
    else:
        raise ValueError(f"Unsupported classes value: {classes}")

    cache_dir = os.path.join(os.path.dirname(path_raw), "cache")
    os.makedirs(cache_dir, exist_ok=True)

    # ── Stage 7: Stratified temporal 3-way split ────────────────────────────────
    if stratify_attack:
        logger.debug("Performing stratified temporal 3-way split")
        if classes == "category":
            _label_col_split = "Attack"
        else:
            _label_col_split = "Attack"  # binary still uses Attack when stratifying
        df_train, df_surr, df_test = temporal_three_way_split_stratified(
            df,
            label_col=_label_col_split,
            train_frac=train_frac,
            surr_frac=surr_frac,
            timestamp_col="Timestamp",
            seed=seed,
        )
    else:
        logger.debug("Performing non-stratified temporal 3-way split")
        df_train, df_surr, df_test = temporal_three_way_split(
            df,
            train_frac=train_frac,
            surr_frac=surr_frac,
            timestamp_col="Timestamp",
        )

    # ── Stage 8: Process labels ───────────────────────────────────────────────
    for part in [df_train, df_surr, df_test]:
        if classes == "category":
            part["label"] = df.loc[part.index, label_columns].values.tolist()
            part.drop(columns=["Attack"] + label_columns, errors="ignore", inplace=True)
        elif classes == "binary":
            part["label"]    = df.loc[part.index, label_columns]
            part["category"] = df.loc[part.index, label_categories].map(lambda x: [str(x)])
            part.drop(columns=[label_columns, label_categories], errors="ignore", inplace=True)

    # ── Stage 9: Determine columns to normalise ───────────────────────────────
    cols_to_norm = df_train.columns.difference(
        ["label", "sID", "dID", "Label", "category", "Timestamp", "Flow ID"]
    )
    cols_to_norm = cols_to_norm[~cols_to_norm.str.startswith(
        ("Service_", "ProtocolO_", "TCP_FLAGS_", "Attack_")
    )]
    logger.debug(f"Normalising {len(cols_to_norm)} features")

    # ── Stage 9a: Category encoding (binary mode) — fit on full dataset ───────
    if classes == "binary":
        le_category = LabelEncoder()
        all_cats = pandas.concat(
            [df_train["category"], df_surr["category"], df_test["category"]]
        ).apply(lambda x: x[0] if isinstance(x, list) else x)
        le_category.fit(all_cats)
        for part in [df_train, df_surr, df_test]:
            part["category"] = le_category.transform(
                part["category"].apply(lambda x: x[0] if isinstance(x, list) else x)
            )

    # ── Stage 10: Convert to numeric and drop NaN ──────────────────────────────
    for part in [df_train, df_surr, df_test]:
        part[cols_to_norm] = part[cols_to_norm].apply(pandas.to_numeric, errors="coerce")
        part.dropna(subset=cols_to_norm, inplace=True)

    # ── Stage 11: Fit NIDS scaler on train ─────────────────────────────────────
    nids_scaler = StandardScaler()
    nids_scaler.fit(df_train[cols_to_norm])
    logger.info("NIDS scaler fitted on df_train")

    # ── Stage 12: Create metadata dictionary ───────────────────────────────────
    split_tag = f"t{int(train_frac*100)}s{int(surr_frac*100)}e{int(100-train_frac*100-surr_frac*100)}"
    sa_tag = "_sa" if stratify_attack else ""
    ns_tag = "_ns" if not stratify_attack else ""
    
    metadata = {
        'classes_def': classes,
        'seed': seed,
        'train_frac': train_frac,
        'surr_frac': surr_frac,
        'stratify_attack': stratify_attack,
        'feature_names': list(cols_to_norm),
        'label_names': label_names,
        'normalized_columns': list(cols_to_norm),
        'split_tag': split_tag + sa_tag + ns_tag,
    }

    # ── Stage 13: Build the three graphs ─────────────────────────────────────
    G_train = _build_partition_graph(
        df_train, classes, nids_scaler, cols_to_norm, label_names, "train", metadata, seed
    )
    G_surr = _build_partition_graph(
        df_surr, classes, nids_scaler, cols_to_norm, label_names, "surr", metadata, seed
    )
    G_test = _build_partition_graph(
        df_test, classes, nids_scaler, cols_to_norm, label_names, "test", metadata, seed
    )

    # ── Stage 14: Save shared cache artefacts ──────────────────────────────────
    suffix = ""
    
    torch.save(
        label_names,
        osp.join(cache_dir, f"label_names_XIIoTID_3way_{classes}{suffix}_seed{seed}.pth")
    )
    torch.save(
        {"scaler": nids_scaler, "normalized_columns": cols_to_norm.tolist()},
        osp.join(cache_dir, f"scaler_XIIoTID_3way_{classes}{suffix}_seed{seed}.pth")
    )
    torch.save(
        list(cols_to_norm),
        osp.join(cache_dir, f"feature_names_XIIoTID_3way_{classes}{suffix}_seed{seed}.pth")
    )
    if classes == "binary":
        torch.save(
            le_category.classes_,
            osp.join(cache_dir, f"encoder_XIIoTID_3way_category_{classes}{suffix}_seed{seed}.pth")
        )

    logger.debug("All three partition graphs created")
    return G_train, G_surr, G_test


class XIIoTIDHeterogeneousGraph(Dataset):
    """
    PyTorch Geometric Dataset implementation for X-IIoTID heterogeneous graphs
    with three-way temporal split.
    
    * ``get_dgl(0)`` → G_train (NIDS victim, NIDS scaler)
    * ``get_dgl(1)`` → G_surr  (attacker surrogate, own scaler)
    * ``get_dgl(2)`` → G_test  (evaluation, NIDS scaler)
    * ``get_dgl(3)`` → G_merged (train + test flows, for merged evaluation mode)
    """
    
    def __init__(
        self, 
        root: str, 
        classes_def: str,
        seed: int = 42,
        train_frac: float = 0.4,
        surr_frac: float = 0.4,
        stratify_attack: bool = True,
        gaussian_init: bool = True,
        transform=None, 
        pre_transform=None, 
        pre_filter=None
    ):
        """
        Initialize X-IIoTID dataset.
        
        Args:
            root:            Root directory containing raw/ and processed/ subdirectories.
            classes_def:     Classification mode ("binary" or "category").
            seed:            Random seed for reproducible preprocessing (default: 42).
            train_frac:      Fraction of flows for NIDS training graph (default 0.4).
            surr_frac:       Fraction of flows for attacker surrogate graph (default 0.4).
            stratify_attack: When True, stratify the temporal split on the fine-grained
                             "Attack" column (e.g. "DDoS", "Ransomware") even in binary
                             mode.  Default False (stratify on class3 0/1 in binary mode).
            gaussian_init:   Whether to initialize endpoint features with N(0,1) Gaussian
                             distribution (default: True).
            transform, pre_transform, pre_filter: standard PyG arguments.
        """
        self.classes_def     = classes_def
        self.seed            = seed
        self.train_frac      = train_frac
        self.surr_frac       = surr_frac
        self.stratify_attack = stratify_attack
        self.gaussian_init   = gaussian_init
        super().__init__(root, transform, pre_transform, pre_filter)
        self.name = f"XIIoTIDHeterogeneousGraph_{classes_def}"
        
    @property
    def raw_file_names(self):
        """
        Specify expected raw data files.
        """
        return ["X-IIoTID.csv"]

    def download(self):
        """
        Download method not implemented.
        
        Raises:
            NotImplementedError: X-IIoTID requires manual download
            
        Rationale:
            Academic datasets often require license agreements that prevent
            automated downloads. Users must obtain data through official channels.
        """
        raise NotImplementedError("Download not implemented")

    # ── filename helpers ──────────────────────────────────────────────────────

    def _split_tag(self) -> str:
        """Short string that encodes the split config, used in filenames."""
        tr = int(self.train_frac * 100)
        sr = int(self.surr_frac * 100)
        te = 100 - tr - sr
        sa = "_sa" if self.stratify_attack else ""
        ns = "_ns" if not self.stratify_attack else ""
        return f"t{tr}s{sr}e{te}{sa}{ns}"

    def _nx_fname(self, partition: str) -> str:
        tag = self._split_tag()
        return f"hetero_graph_XIIoTID_{self.classes_def}_{tag}_{partition}_seed{self.seed}.pt"

    def _dgl_fname(self, partition: str) -> str:
        tag = self._split_tag()
        return f"hetero_graph_XIIoTID_{self.classes_def}_{tag}_{partition}_dgl_seed{self.seed}.pt"

    # ── PyG Dataset interface ─────────────────────────────────────────────────

    @property
    def processed_file_names(self):
        """
        Processed file names — one NetworkX file per partition.

        PyG re-runs ``process()`` only when any of these are missing, so
        changing ``train_frac`` / ``surr_frac`` triggers a rebuild automatically
        because the filenames encode the split configuration.
        """
        # 4 partitions: train, surr, test, merged
        return [self._nx_fname(p) for p in ("train", "surr", "test", "merged")]

    def _convert_and_save_partition(
        self,
        nx_graph: nx.Graph,
        partition: str,
    ) -> None:
        """
        Convert a single NetworkX partition graph to DGL and persist both formats.

        Attaches endpoint role tensors, timestamps, ``local_indicator``, and
        the string ``real_id`` list so callers can match endpoints across graphs.
        """
        dgl_graph, node_mappings = utils.convert_networkx_to_dgl_heterogeneous(nx_graph)
        flow_to_idx     = node_mappings["flow"]      # {node_name: dgl_idx}
        endpoint_to_idx = node_mappings["endpoint"]  # {node_name: dgl_idx}

        # ── Timestamps on flow nodes ──────────────────────────────────────────
        # sort by DGL index so the tensor is correctly aligned
        sorted_flow_nodes = sorted(flow_to_idx, key=flow_to_idx.get)
        timestamps = [nx_graph.nodes[n]["timestamp"] for n in sorted_flow_nodes]
        if timestamps and isinstance(timestamps[0], (str, pandas.Timestamp)):
            timestamps = [pandas.Timestamp(ts).value / 1e9 for ts in timestamps]
        dgl_graph.nodes["flow"].data["timestamp"] = torch.tensor(timestamps, dtype=torch.float64)

        # ── Endpoint metadata ─────────────────────────────────────────────────
        # sort endpoint names by their DGL index to keep everything aligned
        endpoint_nodes = sorted(endpoint_to_idx, key=endpoint_to_idx.get)
        node_mapping   = endpoint_to_idx  # already {node_name: dgl_idx}
        nx_graph.node_mapping = node_mapping

        local_prefixes   = ["10.0.", "192.168", "172"]
        local_indicator  = [any(n.startswith(p) for p in local_prefixes) for n in endpoint_nodes]
        nx_graph.local_indicator = torch.tensor(local_indicator)
        dgl_graph.local_indicator = torch.tensor(local_indicator)

        is_src  = [nx_graph.nodes[n].get("is_source",      False) for n in endpoint_nodes]
        is_dst  = [nx_graph.nodes[n].get("is_destination", False) for n in endpoint_nodes]
        is_internal = [nx_graph.nodes[n].get("is_internal", False) for n in endpoint_nodes]

        dgl_graph.nodes["endpoint"].data["is_source"]      = torch.tensor(is_src,       dtype=torch.bool)
        dgl_graph.nodes["endpoint"].data["is_destination"] = torch.tensor(is_dst,       dtype=torch.bool)
        dgl_graph.nodes["endpoint"].data["is_internal"]    = torch.tensor(is_internal, dtype=torch.bool)

        # Store the real_id (IP:port string) as a list on the DGL graph so it can be
        # retrieved without loading the NetworkX graph (used for cross-graph matching).
        dgl_graph.endpoint_real_ids = [nx_graph.nodes[n]["real_id"] for n in endpoint_nodes]
        
        # Attach metadata if available
        if hasattr(nx_graph, 'metadata'):
            dgl_graph.metadata = nx_graph.metadata

        logger.debug(
            f"[{partition}] src={sum(is_src)}, dst={sum(is_dst)}, "
            f"endpoints={len(endpoint_nodes)}, flows={dgl_graph.num_nodes('flow')}"
        )

        # ── Persist ───────────────────────────────────────────────────────────
        torch.save(nx_graph,  osp.join(self.processed_dir, self._nx_fname(partition)))
        torch.save(dgl_graph, osp.join(self.processed_dir, self._dgl_fname(partition)))

    def _create_merged_graph_dgl(self, g_train_dgl, g_test_dgl):
        """
        Create merged graph using direct DGL operations (no NetworkX conversion).
        This is much faster and avoids the bottleneck of converting to NetworkX.
        """
        import copy
        
        # Deep copy test graph as base
        g_merged = copy.deepcopy(g_test_dgl)
        
        # Edge types in canonical form
        FLOW_TO_ENDPOINT = ('flow', 'depends_on', 'endpoint')
        ENDPOINT_TO_FLOW = ('endpoint', 'links_to', 'flow')
        
        # 1. Extract train flow data
        train_flows = torch.where(g_train_dgl.nodes['flow'].data.get('train_mask', 
            torch.ones(g_train_dgl.num_nodes('flow'), dtype=torch.bool)))[0]
        
        train_features = g_train_dgl.nodes['flow'].data['h'][train_flows]
        train_labels = g_train_dgl.nodes['flow'].data['label'][train_flows]
        if train_labels.ndim > 1:
            train_labels = train_labels.argmax(dim=1)
        
        train_timestamps = g_train_dgl.nodes['flow'].data.get('timestamp', 
            torch.zeros(len(train_flows), dtype=torch.float64))[train_flows]
        
        # Get train flow categories if available
        train_categories = None
        if 'category' in g_train_dgl.nodes['flow'].data:
            train_categories = g_train_dgl.nodes['flow'].data['category'][train_flows]
        
        # 2. Add train flows to merged graph
        n_train_flows = len(train_flows)
        g_merged.add_nodes(n_train_flows, ntype='flow')
        start_idx = g_merged.num_nodes('flow') - n_train_flows
        
        # Set flow features
        g_merged.nodes['flow'].data['h'][start_idx:] = train_features
        g_merged.nodes['flow'].data['label'][start_idx:] = train_labels
        g_merged.nodes['flow'].data['timestamp'][start_idx:] = train_timestamps
        
        if train_categories is not None:
            g_merged.nodes['flow'].data['category'][start_idx:] = train_categories
        
        # Add merged_mask to identify flows from train
        if 'merged_mask' not in g_merged.nodes['flow'].data:
            g_merged.nodes['flow'].data['merged_mask'] = torch.zeros(
                g_merged.num_nodes('flow'), dtype=torch.bool, device=g_merged.device
            )
        g_merged.nodes['flow'].data['merged_mask'][start_idx:] = True
        
        # Mark these flows as not test flows
        if 'test_mask' in g_merged.nodes['flow'].data:
            g_merged.nodes['flow'].data['test_mask'][start_idx:] = False
        if 'train_mask' in g_merged.nodes['flow'].data:
            g_merged.nodes['flow'].data['train_mask'][start_idx:] = False
        if 'val_mask' in g_merged.nodes['flow'].data:
            g_merged.nodes['flow'].data['val_mask'][start_idx:] = False
        
        # 3. Extract and add edges from train flows
        try:
            # Get flow->endpoint edges
            train_flow_src, train_flow_dst = g_train_dgl.edges(etype=FLOW_TO_ENDPOINT)
            train_flow_src = train_flow_src.cpu().numpy()
            train_flow_dst = train_flow_dst.cpu().numpy()
            
            # Create mapping from source flow index to merged flow index
            old_to_new_flow = {}
            for i, old_idx in enumerate(train_flows):
                old_to_new_flow[int(old_idx)] = start_idx + i
            
            # Build edges for merged graph
            new_flow_src = []
            new_flow_dst = []
            
            for i, old_flow_idx in enumerate(train_flow_src):
                if old_flow_idx in old_to_new_flow:
                    new_flow_src.append(old_to_new_flow[old_flow_idx])
                    new_flow_dst.append(train_flow_dst[i])
            
            # Add edges to merged graph
            if new_flow_src:
                g_merged.add_edges(
                    torch.tensor(new_flow_src, device=g_merged.device),
                    torch.tensor(new_flow_dst, device=g_merged.device),
                    etype=FLOW_TO_ENDPOINT
                )
                # Add reverse edges
                g_merged.add_edges(
                    torch.tensor(new_flow_dst, device=g_merged.device),
                    torch.tensor(new_flow_src, device=g_merged.device),
                    etype=ENDPOINT_TO_FLOW
                )
                
        except Exception as e:
            logger.warning(f"Error adding train edges: {e}")
        
        # Ensure all endpoint attributes are consistent
        endpoint_attrs = ['is_source', 'is_destination', 'is_internal']
        for attr in endpoint_attrs:
            if attr in g_train_dgl.nodes['endpoint'].data and attr not in g_merged.nodes['endpoint'].data:
                g_merged.nodes['endpoint'].data[attr] = torch.zeros(
                    g_merged.num_nodes('endpoint'), dtype=torch.bool, device=g_merged.device
                )
        
        logger.info(f"  Merged graph: {g_merged.num_nodes('flow')} flows, "
                    f"{g_merged.num_nodes('endpoint')} endpoints")
        
        return g_merged

    def process(self):
        """Process raw X-IIoTID CSV → three independent partition graphs."""
        logger.info(
            f"Processing XIIoTID — split: train={self.train_frac:.0%} "
            f"surr={self.surr_frac:.0%} "
            f"test={1-self.train_frac-self.surr_frac:.0%}"
        )
        
        # 1. Build the three base graphs
        G_train, G_surr, G_test = pipeline_create_heterogeneous_xiiotid(
            self.raw_dir,
            self.classes_def,
            seed=self.seed,
            train_frac=self.train_frac,
            surr_frac=self.surr_frac,
            stratify_attack=self.stratify_attack,
        )
        
        # 2. Save the three base partitions
        for nx_g, part in [(G_train, "train"), (G_surr, "surr"), (G_test, "test")]:
            self._convert_and_save_partition(nx_g, part)
        
        # 3. Create the merged graph directly in DGL
        logger.info("Creating merged graph (train + test) for evaluation using DGL...")
        
        # Load saved DGL partitions
        g_train_dgl_path = osp.join(self.processed_dir, self._dgl_fname("train"))
        g_test_dgl_path = osp.join(self.processed_dir, self._dgl_fname("test"))
        
        g_train_dgl = torch.load(g_train_dgl_path)
        g_test_dgl = torch.load(g_test_dgl_path)
        
        # Create merged graph (DGL only)
        g_merged_dgl = self._create_merged_graph_dgl(g_train_dgl, g_test_dgl)
        
        # Create a minimal NetworkX graph for metadata
        import networkx as nx
        G_merged_nx = nx.MultiGraph()
        G_merged_nx.partition = "merged"
        G_merged_nx.label_names = G_train.label_names if hasattr(G_train, 'label_names') else ["Benign", "Malicious"]
        G_merged_nx.scaler = G_train.scaler if hasattr(G_train, 'scaler') else None
        G_merged_nx.cols_to_norm = G_train.cols_to_norm if hasattr(G_train, 'cols_to_norm') else []
        
        if hasattr(G_train, 'metadata'):
            G_merged_nx.metadata = {**G_train.metadata, 'merged': True}
        
        # Add compatibility attributes
        G_merged_nx.node_mapping = {ep: i for i, ep in enumerate(g_merged_dgl.endpoint_real_ids)} if hasattr(g_merged_dgl, 'endpoint_real_ids') else {}
        G_merged_nx.local_indicator = g_merged_dgl.local_indicator if hasattr(g_merged_dgl, 'local_indicator') else torch.tensor([])
        
        # Save merged graph
        torch.save(G_merged_nx, osp.join(self.processed_dir, self._nx_fname("merged")))
        torch.save(g_merged_dgl, osp.join(self.processed_dir, self._dgl_fname("merged")))
        
        logger.info("XIIoTID processing complete — all three partition graphs saved with merged graph")

    def len(self) -> int:
        """Return number of partitions (4: train, surr, test, merged)."""
        return 4

    # ── Graph retrieval ───────────────────────────────────────────────────────

    _PARTITION_NAMES = ("train", "surr", "test", "merged")

    def _cache_dir(self) -> str:
        return osp.join(osp.dirname(self.raw_dir), "cache")

    def _load_shared_metadata(self, dgl_graph):
        """Attach shared cache artefacts (scaler, encoders) to a DGL graph."""
        cache_dir = self._cache_dir()
        suffix = ""

        # Load feature names - consistent naming for 3-way split
        feat_path = osp.join(
            cache_dir,
            f"feature_names_XIIoTID_3way_{self.classes_def}{suffix}_seed{self.seed}.pth"
        )
        if osp.exists(feat_path):
            dgl_graph.feature_names = torch.load(feat_path)
        else:
            dgl_graph.feature_names = None
            logger.warning(f"Feature names not found at {feat_path}")

        # Load label names
        label_path = osp.join(
            cache_dir,
            f"label_names_XIIoTID_3way_{self.classes_def}{suffix}_seed{self.seed}.pth"
        )
        if osp.exists(label_path):
            dgl_graph.label_names = torch.load(label_path)

        # Load category encoder
        cat_path = osp.join(
            cache_dir,
            f"encoder_XIIoTID_3way_category_{self.classes_def}{suffix}_seed{self.seed}.pth"
        )
        dgl_graph.encoder_category = torch.load(cat_path) if osp.exists(cat_path) else None

        # Load scaler
        scaler_path = osp.join(
            cache_dir,
            f"scaler_XIIoTID_3way_{self.classes_def}{suffix}_seed{self.seed}.pth"
        )
        if osp.exists(scaler_path):
            sd = torch.load(scaler_path)
            dgl_graph.scaler = sd["scaler"]
            dgl_graph.normalized_columns = sd["normalized_columns"]
        else:
            dgl_graph.scaler = None
            dgl_graph.normalized_columns = None
            logger.warning(f"NIDS scaler not found at {scaler_path}")
        
        return dgl_graph

    def get(self, idx: int):
        """
        Retrieve a NetworkX partition graph with metadata.

        Args:
            idx: 0 → train (NIDS), 1 → surr (attacker), 2 → test, 3 → merged.

        Returns:
            nx.MultiGraph with attributes.
        """
        partition = self._PARTITION_NAMES[idx]
        data = torch.load(osp.join(self.processed_dir, self._nx_fname(partition)))
        return data

    def get_dgl(self, idx: int):
        """
        Retrieve a DGL partition graph with Gaussian-initialized endpoints by default.

        Args:
            idx: 0 → train (NIDS), 1 → surr (attacker), 2 → test, 3 → merged.

        Returns:
            DGL heterograph with Gaussian-initialized endpoint features.
        """
        partition = self._PARTITION_NAMES[idx]
        g = torch.load(osp.join(self.processed_dir, self._dgl_fname(partition)))
        
        # Apply Gaussian initialization if requested
        if self.gaussian_init:
            if 'endpoint' in g.ntypes and 'h' in g.nodes['endpoint'].data:
                num_endpoints = g.num_nodes('endpoint')
                feature_dim = g.nodes['endpoint'].data['h'].shape[1]
                
                # Use dataset seed + idx for reproducibility
                torch.manual_seed(self.seed + idx)
                
                # Generate features from standard normal distribution N(0,1)
                gaussian_features = torch.randn(num_endpoints, feature_dim)
                
                # Replace endpoint features
                g.nodes['endpoint'].data['h'] = gaussian_features
                logger.debug(f"Initialized endpoints with N(0,1) features (dim={feature_dim})")
        
        return self._load_shared_metadata(g)