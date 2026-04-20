"""
CIC-IDS-2017 dataset heterogeneous graph creation.
"""

import os
import os.path as osp
import torch
import pandas
import networkx as nx
from torch_geometric.data import Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
import random
import numpy as np

from src.data.processing import PreprocessingCICIDS2017
from ..utils.splitting import temporal_stratified_split, temporal_three_way_split, temporal_three_way_split_stratified
from src.data.graph_creation import utils
from src.utils.logger import setup_logging

setup_logging()
import logging
logger = logging.getLogger(__name__)
import dgl

def pipeline_create_heterogeneous_cicids2017(path_raw, classes, apply_undersampling=True, seed=42):
    """Create a heterogeneous graph from CICIDS2017 data.
    
    Args:
        path_raw: Path to raw dataset
        classes: Classification type ("binary" or "category")
        apply_undersampling: Whether to undersample the majority class
        seed: Random seed for reproducible splits (default: 42)
        
    Returns:
        nx.Graph: Heterogeneous graph representation
        
    """
    # Set all random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    logger.debug(f"Preprocessing CICIDS2017 data with seed={seed}")
    df = PreprocessingCICIDS2017(
        path_raw_dataset=path_raw,
        classes=classes,
        apply_undersampling=apply_undersampling,
    ).df
    
    # Remove constant features
    constant_features = df.columns[df.nunique() <= 1]
    df.drop(columns=constant_features, inplace=True)
    
    # Remove rare binary features (<0.1%)
    rare_threshold = 0.001
    rare_features = [col for col in df.columns
                     if df[col].nunique() == 2 and df[col].value_counts(normalize=True).min() < rare_threshold]
    logger.info(f"Removing rare binary features: {rare_features}")
    df.drop(columns=rare_features, inplace=True)

    # Adaptive removal of extreme values using fixed quantile thresholds
    from scipy.stats import iqr
    max_std = 100
    max_quantile = 0.99
    max_iters = 5
    it = 0
    n_before = df.shape[0]
    
    logger.debug("Computing fixed quantile thresholds on original distribution")
    
    # Step 1: Identify high-variance features and compute fixed thresholds
    numeric_cols = df.select_dtypes(include=[float, int]).columns
    excluded_patterns = ["Port", "IP", "Label", "Attack", "category", "sID", "dID"]
    feature_cols = [col for col in numeric_cols 
                    if not any(pattern in col for pattern in excluded_patterns)]
    
    # Compute standard deviations on original data
    stds_original = pandas.Series({col: df[col].std() for col in feature_cols})
    high_var_features_initial = stds_original[stds_original > max_std].index.tolist()
    
    if not high_var_features_initial:
        logger.info(f"No high-variance features detected (threshold: std > {max_std})")
    else:
        # Compute and store fixed quantile thresholds from original distribution
        quantile_thresholds = {}
        for col in high_var_features_initial:
            quantile_thresholds[col] = df[col].quantile(max_quantile)
            logger.debug(f"{col}: fixed threshold at {max_quantile*100:.0f}% = {quantile_thresholds[col]:.2f}")
        
        logger.info(f"Identified {len(high_var_features_initial)} high-variance features for outlier removal")
        
        # Step 2: Iterative removal using fixed thresholds
        while it < max_iters:
            # Recompute variances on current (cleaned) data to check convergence
            stds_current = pandas.Series({col: df[col].std() for col in feature_cols if col in df.columns})
            high_var_features_current = stds_current[stds_current > max_std].index.tolist()
            
            if not high_var_features_current:
                logger.info(f"[Iter {it+1}] Convergence achieved: no remaining high-variance features")
                break
            
            logger.info(f"[Iter {it+1}] Checking {len(high_var_features_current)} high-variance features")
            mask_outlier = pandas.Series(False, index=df.index)
            
            # Apply fixed thresholds to identify outliers
            for col in high_var_features_current:
                if col in quantile_thresholds:
                    # Use pre-computed threshold from original distribution
                    q_thresh = quantile_thresholds[col]
                else:
                    # Feature became high-variance during cleaning (rare case)
                    # Compute threshold from current distribution
                    q_thresh = df[col].quantile(max_quantile)
                    quantile_thresholds[col] = q_thresh
                    logger.warning(f"{col} became high-variance during cleaning, computing new threshold: {q_thresh:.2f}")
                
                outlier_col = df[col] > q_thresh
                n_outliers = outlier_col.sum()
                
                if n_outliers > 0:
                    logger.info(f"  {col}: flagging {n_outliers} samples above {max_quantile*100:.0f}% quantile (>{q_thresh:.2f})")
                    mask_outlier |= outlier_col
            
            # Remove flagged outliers
            n_flagged = mask_outlier.sum()
            if n_flagged == 0:
                logger.info(f"[Iter {it+1}] No outliers detected, stopping iteration")
                break
            
            df = df[~mask_outlier]
            n_current = df.shape[0]
            logger.info(f"[Iter {it+1}] Removed {n_flagged} rows (remaining: {n_current}, {100*n_flagged/n_before:.2f}% of original)")
            
            it += 1
    
    n_after = df.shape[0]
    n_removed = n_before - n_after
    pct_removed = 100 * n_removed / n_before if n_before > 0 else 0
    logger.info(f"Outlier removal complete after {it} iterations:")
    logger.info(f"  Removed: {n_removed} rows ({pct_removed:.1f}% of original)")
    logger.info(f"  Remaining: {n_after} rows ({100-pct_removed:.1f}% of original)")
    
    # Add suffix based on undersampling
    suffix = "_undersample" if apply_undersampling else ""
    
    # Create node IDs
    df["sID"] = df["Source IP"].astype(str) + ":" + df["Source Port"].astype(str)
    df["dID"] = df["Destination IP"].astype(str) + ":" + df["Destination Port"].astype(str)
    df.drop(columns=["Source IP", "Source Port", "Destination IP", "Destination Port"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Convert bools to ints
    bool_columns = df.select_dtypes(include=[bool]).columns
    df[bool_columns] = df[bool_columns].astype(int)
    
    # Define label columns
    if classes =="category":
        label_columns = [col for col in df.columns if col.startswith("Attack_")]
        label_names = label_columns
    elif classes == "binary":
        label_columns = "Label"
        label_names = ["BENIGN", "ATTACK"]
        label_categories = "Attack"
    
    edge_df = df.copy()
    
    logger.debug(f"Label columns: {label_columns}")
    logger.debug(f" df shape: {df.shape}")
    
    # Save feature columns to file
    cache_dir = os.path.join(os.path.dirname(path_raw), "cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Split data
    logger.debug(f"Performing stratified temporal split with seed={seed}")
    edge_df_train, edge_df_val, edge_df_test = temporal_stratified_split(
        edge_df,
        label_columns,
        train_frac=0.6,
        val_frac=0.2,
        seed=seed
    )

    # Process labels based on classification type
    if classes == "category":
        # Assign list of Attack_* values as label
        edge_df_train["label"] = df.loc[edge_df_train.index, label_columns].values.tolist()
        edge_df_val["label"] = df.loc[edge_df_val.index, label_columns].values.tolist()
        edge_df_test["label"] = df.loc[edge_df_test.index, label_columns].values.tolist()

        # Remove "Attack" column if present
        for edge_df_part in [edge_df_train, edge_df_val, edge_df_test]:
            edge_df_part.drop(columns=["Attack"], errors="ignore", inplace=True)
            edge_df_part.drop(columns=label_columns, errors="ignore", inplace=True)

    elif classes == "binary":
        # Binary labels (0/1)
        edge_df_train["label"] = df.loc[edge_df_train.index, label_columns]
        edge_df_val["label"] = df.loc[edge_df_val.index, label_columns]
        edge_df_test["label"] = df.loc[edge_df_test.index, label_columns]

        # Also store attack categories
        edge_df_train["category"] = df.loc[edge_df_train.index, label_categories].apply(lambda x: [x])
        edge_df_val["category"] = df.loc[edge_df_val.index, label_categories].apply(lambda x: [x])
        edge_df_test["category"] = df.loc[edge_df_test.index, label_categories].apply(lambda x: [x])

        # Drop original label columns
        for edge_df_part in [edge_df_train, edge_df_val, edge_df_test]:
            edge_df_part.drop(columns=["Label", label_categories], inplace=True)

    # Normalize features
    logger.debug("Scaling")
    cols_to_norm = edge_df_train.columns.difference(
        ["sID", "dID", "label", "category", "Timestamp", "Flow ID"]
    )
    cols_to_norm = cols_to_norm[
        ~cols_to_norm.str.startswith(("Attack_", "Protocol"))
    ]
    
    scaler = StandardScaler()
    scaler.fit(edge_df_train[cols_to_norm])
    edge_df_train[cols_to_norm] = scaler.transform(edge_df_train[cols_to_norm])
    edge_df_val[cols_to_norm] = scaler.transform(edge_df_val[cols_to_norm])
    edge_df_test[cols_to_norm] = scaler.transform(edge_df_test[cols_to_norm])
    
    # Save scaler for inverse transformation
    scaler_path = os.path.join(cache_dir, f"scaler_CIC2017_{classes}{suffix}_seed{seed}.pth")
    torch.save({
        'scaler': scaler,
        'normalized_columns': cols_to_norm.tolist()
    }, scaler_path)
    logger.debug(f"Saved StandardScaler to {scaler_path} for inverse transformation")

    # Combine data
    logger.debug("Concatenating")
    edge_df = pandas.concat([edge_df_train, edge_df_val, edge_df_test])
    edge_df.reset_index(drop=True, inplace=True)

    # Create feature vectors
    edge_df["h"] = edge_df.drop(columns=["label", "sID", "dID", "category", "Timestamp", "Flow ID"], errors="ignore").apply(
        lambda row: row.tolist(), axis=1
    )
    
    # Encode category labels if binary classification
    if classes == "binary" and "category" in edge_df.columns:
        le_category = LabelEncoder()
        edge_df["category"] = le_category.fit_transform(edge_df["category"].apply(lambda x: x[0] if isinstance(x, list) else x))
        torch.save(le_category.classes_, os.path.join(cache_dir, f"encoder_CIC2017_category_{classes}{suffix}_seed{seed}.pth"))

    # Save feature column names - CONSISTENT NAMING: use 'feature_names' not 'features_name'
    feature_cols = edge_df.drop(columns=["label", "sID", "dID", "h", "category"], errors="ignore").columns
    torch.save(feature_cols, os.path.join(cache_dir, f"feature_names_CIC2017_{classes}{suffix}_seed{seed}.pth"))
    torch.save(label_names, os.path.join(cache_dir, f"label_names_CIC2017_{classes}{suffix}_seed{seed}.pth"))
    
    # Keep only needed columns
    columns_to_keep = ["h", "label", "sID", "dID", "category", "Timestamp", "Flow ID"] if classes == "binary" else ["h", "label", "sID", "dID", "Timestamp", "Flow ID"]
    edge_df = edge_df[columns_to_keep]

    # Create masks
    logger.debug("Creating masks")
    num_edges = edge_df.shape[0]
    train_mask = torch.zeros(num_edges, dtype=torch.bool)
    val_mask = torch.zeros(num_edges, dtype=torch.bool)
    test_mask = torch.zeros(num_edges, dtype=torch.bool)
    train_mask[edge_df_train.index] = True
    val_mask[edge_df_val.index] = True
    test_mask[edge_df_test.index] = True
    edge_df["train_mask"] = train_mask
    edge_df["val_mask"] = val_mask
    edge_df["test_mask"] = test_mask

    # Create heterogeneous graph
    logger.debug("Creating networkx graph")
    G = nx.MultiGraph()
    
    # Track source and destination nodes for mask creation
    source_nodes = set(edge_df["sID"])
    dest_nodes = set(edge_df["dID"])
    
    endpoints = source_nodes | dest_nodes
    for ep in endpoints:
        G.add_node(ep, 
                   type=0,
                   is_source=(ep in source_nodes),
                   is_destination=(ep in dest_nodes))

    for idx, row in edge_df.iterrows():
        flow_node = f"flow_{idx}"
        G.add_node(
            flow_node,
            type=1,
            h=row["h"],
            label=row["label"],
            category=row.get("category"),
            train_mask=row["train_mask"],
            val_mask=row["val_mask"],
            test_mask=row["test_mask"],
            timestamp=row["Timestamp"],
            flow_id=row["Flow ID"],
        )
        G.add_edge(row["sID"], flow_node)
        G.add_edge(flow_node, row["dID"])

    # Attach metadata to graph for easy access without loading
    G.metadata = {
        'classes_def': classes,
        'apply_undersampling': apply_undersampling,
        'seed': seed,
        'feature_names': list(feature_cols),
        'label_names': label_names,
        'normalized_columns': list(cols_to_norm),
        'scaler_path': scaler_path
    }

    logger.debug("Graph creation done")
    assert not G.is_directed()
    return G


class CICIDS2017HeterogeneousGraph(Dataset):
    """Dataset class for CICIDS2017 heterogeneous graphs."""
    
    def __init__(
        self, root, classes_def, apply_undersampling=True, seed=42, 
        gaussian_init=True,
        transform=None, pre_transform=None, pre_filter=None 
    ):
        self.classes_def = classes_def
        self.apply_undersampling = apply_undersampling
        self.seed = seed
        self.gaussian_init = gaussian_init
        self.suffix = ""
        super().__init__(root, transform, pre_transform, pre_filter)
        
        self.name = f"CICIDS2017_Heterogeneous_{classes_def}_seed{seed}"
        
    @property
    def raw_file_names(self):
        return [
            "Friday-WorkingHours.csv",
            "Thursday-WorkingHours.csv",
            "Friday-WorkingHours.csv",
            "Tuesday-WorkingHours.csv",
            "Monday-WorkingHours.csv",
            "Wednesday-WorkingHours.csv",
        ]
    
    def download(self):
        raise NotImplementedError("Download is not implemented for this dataset")
        
    @property
    def processed_file_names(self):
        return [f"hetero_graph_CICIDS2017_{self.classes_def}{self.suffix}_seed{self.seed}.pt"]
    
    def process(self):
        idx = 0
        for raw_path in [self.raw_dir]:
            logger.debug(f"Creating heterogeneous graph with seed={self.seed}... Processing")
            data = pipeline_create_heterogeneous_cicids2017(
                path_raw=raw_path,
                classes=self.classes_def,
                apply_undersampling=self.apply_undersampling,
                seed=self.seed
            )
            dgl_graph, node_mapping = utils.convert_networkx_to_dgl_heterogeneous(data)

            # Manually add timestamp to DGL graph
            nx_flow_nodes = [n for n, d in data.nodes(data=True) if d.get('type') == 1]
            flow_node_indices = node_mapping['flow']
            sorted_flow_nodes = sorted(flow_node_indices, key=flow_node_indices.get)
            
            timestamps = [data.nodes[n]['timestamp'] for n in sorted_flow_nodes]
            # convert to float if they are strings/timestamps
            if len(timestamps) > 0 and isinstance(timestamps[0], (str, pandas.Timestamp)):
                 timestamps = [pandas.Timestamp(ts).value / 10**9 for ts in timestamps]
            
            dgl_graph.nodes['flow'].data['timestamp'] = torch.tensor(timestamps, dtype=torch.float64)

            endpoint_nodes = [n for n in data.nodes() if data.nodes[n]['type'] == 0]
            node_mapping = {node: i for i, node in enumerate(endpoint_nodes)}
            
            # Identify local network nodes
            local_network_prefix = "192.168"
            local_indicator = [node.startswith(local_network_prefix) for node in endpoint_nodes]
            data.node_mapping = node_mapping
            data.local_indicator = torch.tensor(local_indicator)
            
            # Add local indicator to DGL graph
            dgl_graph.local_indicator = torch.tensor(local_indicator)
            
            # Extract and add source/destination masks from NetworkX graph
            is_source_list = [data.nodes[node].get('is_source', False) for node in endpoint_nodes]
            is_destination_list = [data.nodes[node].get('is_destination', False) for node in endpoint_nodes]
            
            # Add masks to DGL graph as endpoint node data
            dgl_graph.nodes['endpoint'].data['is_source'] = torch.tensor(is_source_list, dtype=torch.bool)
            dgl_graph.nodes['endpoint'].data['is_destination'] = torch.tensor(is_destination_list, dtype=torch.bool)
            
            # Attach metadata to DGL graph
            if hasattr(data, 'metadata'):
                dgl_graph.metadata = data.metadata
            
            logger.debug(f"Source endpoints: {sum(is_source_list)}, Destination endpoints: {sum(is_destination_list)}")
            
            # Save graphs
            file_name = f"hetero_graph_CICIDS2017_{self.classes_def}{self.suffix}_seed{self.seed}.pt"
            logger.debug(f"Saving processed data to {file_name}")
            torch.save(data, osp.join(self.processed_dir, file_name))
            
            file_name_dgl = f"hetero_graph_CICIDS2017_{self.classes_def}_dgl{self.suffix}_seed{self.seed}.pt"
            logger.debug(f"Saving processed DGL graph to {file_name_dgl}")
            torch.save(dgl_graph, osp.join(self.processed_dir, file_name_dgl))
            idx += 1
            
    def len(self) -> int:
        return len(self.processed_file_names)
    
    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f"hetero_graph_CICIDS2017_{self.classes_def}{self.suffix}_seed{self.seed}.pt"))
        cache_dir = os.path.join(os.path.dirname(self.raw_dir), "cache")
        
        # Load metadata - CONSISTENT NAMING: use 'feature_names' not 'features_name'
        feature_names_path = os.path.join(cache_dir, f"feature_names_CIC2017_{self.classes_def}{self.suffix}_seed{self.seed}.pth")
        label_names_path = os.path.join(cache_dir, f"label_names_CIC2017_{self.classes_def}{self.suffix}_seed{self.seed}.pth")
        
        if osp.exists(feature_names_path):
            data.feature_names = torch.load(feature_names_path)
        else:
            data.feature_names = None
            logger.warning(f"Feature names not found at {feature_names_path}")
            
        if osp.exists(label_names_path):
            data.label_names = torch.load(label_names_path)
        else:
            data.label_names = None
        
        # Load category encoder if available
        encoder_category_path = os.path.join(cache_dir, f"encoder_CIC2017_category_{self.classes_def}{self.suffix}_seed{self.seed}.pth")
        if osp.exists(encoder_category_path):
            data.encoder_category = torch.load(encoder_category_path)
        else:
            data.encoder_category = None
        
        # Load scaler for inverse transformation
        scaler_path = os.path.join(cache_dir, f"scaler_CIC2017_{self.classes_def}{self.suffix}_seed{self.seed}.pth")
        if osp.exists(scaler_path):
            scaler_data = torch.load(scaler_path)
            data.scaler = scaler_data['scaler']
            data.normalized_columns = scaler_data['normalized_columns']
            logger.debug(f"Loaded StandardScaler with {len(data.normalized_columns)} normalized features")
        else:
            data.scaler = None
            data.normalized_columns = None
            logger.warning(f"Scaler not found - inverse transformation unavailable")
            
        return data
    
    def get_dgl(self, idx, ip_port_init=None): 
        """
        Load the DGL heterogeneous graph with Gaussian initialization by default.
        
        Args:
            idx: Index of the graph to load
            ip_port_init: Deprecated, kept for backward compatibility
                         (use gaussian_init in constructor instead)
                         
        Returns:
            DGL heterogeneous graph with Gaussian-initialized endpoint features
        """
        base_path = osp.join(self.processed_dir, f"hetero_graph_CICIDS2017_{self.classes_def}_dgl{self.suffix}_seed{self.seed}.pt")
        
        # Load base graph
        g = torch.load(base_path)
        
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
        
        # Load metadata
        cache_dir = os.path.join(os.path.dirname(self.raw_dir), "cache")
        self._load_metadata_to_graph(g, cache_dir)
        
        return g
    
    def _load_metadata_to_graph(self, g, cache_dir):
        """Helper method to load metadata into a graph."""
        # Load feature names
        feature_names_path = os.path.join(cache_dir, f"feature_names_CIC2017_{self.classes_def}{self.suffix}_seed{self.seed}.pth")
        if osp.exists(feature_names_path):
            g.feature_names = torch.load(feature_names_path)
        else:
            g.feature_names = None
            
        # Load label names
        label_names_path = os.path.join(cache_dir, f"label_names_CIC2017_{self.classes_def}{self.suffix}_seed{self.seed}.pth")
        if osp.exists(label_names_path):
            g.label_names = torch.load(label_names_path)
        
        # Load category encoder
        encoder_category_path = os.path.join(cache_dir, f"encoder_CIC2017_category_{self.classes_def}{self.suffix}_seed{self.seed}.pth")
        if osp.exists(encoder_category_path):
            g.encoder_category = torch.load(encoder_category_path)
        else:
            g.encoder_category = None
        
        # Load scaler
        scaler_path = os.path.join(cache_dir, f"scaler_CIC2017_{self.classes_def}{self.suffix}_seed{self.seed}.pth")
        if osp.exists(scaler_path):
            scaler_data = torch.load(scaler_path)
            g.scaler = scaler_data['scaler']
            g.normalized_columns = scaler_data['normalized_columns']


# ══════════════════════════════════════════════════════════════════════════════
# Three-way temporal split — mirrors XIIoTIDHeterogeneousGraph
# ══════════════════════════════════════════════════════════════════════════════

def _build_cicids_partition_graph(
    edge_df: pandas.DataFrame,
    classes: str,
    scaler,
    cols_to_norm,
    label_names,
    partition_name: str,
    metadata: dict = None,
) -> nx.MultiGraph:
    """Build one partition NetworkX graph from a pre-split CIC-IDS-2017 DataFrame.

    All three partitions (train, surr, test) are scaled with the NIDS scaler
    fitted on G_train.  The attacker submits raw flows to the NIDS, which
    applies its own scaler internally — no separate attacker scaler exists.

    Args:
        edge_df:        Pre-split partition DataFrame.
        classes:        ``"binary"`` or ``"category"``.
        scaler:         Fitted NIDS scaler to apply.
        cols_to_norm:   Columns to standardise.
        label_names:    Class name list.
        partition_name: ``"train"`` / ``"surr"`` / ``"test"``.
        metadata:       Additional metadata to attach to the graph.

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
    if partition_name == "test":
        train_masks = [False] * n_flows
        val_masks   = [False] * n_flows
        test_masks  = [True]  * n_flows
    else:
        cut = int(0.8 * n_flows)
        train_masks = [True]  * cut + [False] * (n_flows - cut)
        val_masks   = [False] * cut + [True]  * (n_flows - cut)
        test_masks  = [False] * n_flows

    edge_df = edge_df.reset_index(drop=True)
    edge_df["_train_mask"] = train_masks
    edge_df["_val_mask"]   = val_masks
    edge_df["_test_mask"]  = test_masks

    # ── Build NetworkX graph ──────────────────────────────────────────────────
    G = nx.MultiGraph()

    source_nodes = set(edge_df["sID"])
    dest_nodes   = set(edge_df["dID"])
    for ep in source_nodes | dest_nodes:
        G.add_node(
            ep,
            type=0,
            real_id=ep,
            is_source=(ep in source_nodes),
            is_destination=(ep in dest_nodes),
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
        G.add_edge(flow_node, row["dID"])

    # ── Graph-level attributes ────────────────────────────────────────────────
    G.partition    = partition_name
    G.label_names  = label_names
    G.scaler       = scaler
    G.cols_to_norm = list(cols_to_norm)

    endpoint_nodes   = [n for n, d in G.nodes(data=True) if d.get("type") == 0]
    G.node_mapping   = {node: i for i, node in enumerate(endpoint_nodes)}
    local_prefixes   = ["192.168", "10.", "172."]
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


def pipeline_create_heterogeneous_cicids2017_three_way(
    path_raw: str,
    classes: str,
    apply_undersampling: bool = True,
    seed: int = 42,
    train_frac: float = 0.4,
    surr_frac: float = 0.4,
    stratify_attack: bool = False,
) -> tuple:
    """Full preprocessing → three independent CIC-IDS-2017 partition graphs.

    Mirrors ``pipeline_create_heterogeneous_xiiotid`` exactly:

    * **G_train** — NIDS victim graph (NIDS scaler fitted here).
    * **G_surr**  — Attacker surrogate graph (own scaler fitted independently).
    * **G_test**  — Evaluation graph (NIDS scaler applied).

    Args:
        stratify_attack: When True, stratify on the fine-grained ``"Attack"``
                         column (e.g. ``"DDoS"``, ``"PortScan"``) even in binary
                         mode.  Default False (stratify on ``Label`` 0/1).

    Returns:
        ``(G_train, G_surr, G_test)`` — three nx.MultiGraph objects.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    logger.info(
        f"Processing CIC-IDS-2017 three-way split: "
        f"train={train_frac:.0%} surr={surr_frac:.0%} "
        f"test={1-train_frac-surr_frac:.0%}"
    )

    # ── Preprocessing ─────────────────────────────────────────────────────────
    df = PreprocessingCICIDS2017(
        path_raw_dataset=path_raw,
        classes=classes,
        apply_undersampling=apply_undersampling,
    ).df

    # Remove constant / rare binary features
    constant_features = df.columns[df.nunique() <= 1]
    df.drop(columns=constant_features, inplace=True)
    rare_threshold = 0.001
    rare_features = [
        col for col in df.columns
        if df[col].nunique() == 2
        and df[col].value_counts(normalize=True).min() < rare_threshold
    ]
    df.drop(columns=rare_features, inplace=True)

    # Node IDs
    df["sID"] = df["Source IP"].astype(str) + ":" + df["Source Port"].astype(str)
    df["dID"] = df["Destination IP"].astype(str) + ":" + df["Destination Port"].astype(str)
    df.drop(columns=["Source IP", "Source Port", "Destination IP", "Destination Port"],
            inplace=True)
    bool_cols = df.select_dtypes(include=[bool]).columns
    df[bool_cols] = df[bool_cols].astype(int)
    df.reset_index(drop=True, inplace=True)

    # Labels
    if classes == "category":
        label_columns = [col for col in df.columns if col.startswith("Attack_")]
        label_names   = label_columns
    else:
        label_columns = "Label"
        label_names   = ["BENIGN", "ATTACK"]

    # ── Stratified temporal three-way split ───────────────────────────────────────
    if classes == "category" or stratify_attack:
        _label_col_split = "Attack"
    else:
        _label_col_split = label_columns  # "Label" (0/1)
    df_train, df_surr, df_test = temporal_three_way_split_stratified(
        df,
        label_col=_label_col_split,
        train_frac=train_frac,
        surr_frac=surr_frac,
        timestamp_col="Timestamp",
        seed=seed,
    )

    # ── Assign labels ─────────────────────────────────────────────────────────
    for part_df in [df_train, df_surr, df_test]:
        if classes == "category":
            part_df["label"] = df.loc[part_df.index, label_columns].values.tolist()
            part_df.drop(columns=["Attack"] + label_columns, errors="ignore", inplace=True)
        else:
            part_df["label"]    = df.loc[part_df.index, label_columns].values
            part_df["category"] = df.loc[part_df.index, "Attack"].apply(
                lambda x: [x] if not isinstance(x, list) else x
            )
            part_df.drop(columns=["Label", "Attack"], errors="ignore", inplace=True)

    # Encode category column (binary mode)
    if classes == "binary":
        le_cat = LabelEncoder()
        all_cats = pandas.concat([
            df_train["category"], df_surr["category"], df_test["category"]
        ]).apply(lambda x: x[0] if isinstance(x, list) else x)
        le_cat.fit(all_cats)
        for part_df in [df_train, df_surr, df_test]:
            part_df["category"] = le_cat.transform(
                part_df["category"].apply(lambda x: x[0] if isinstance(x, list) else x)
            )

    # ── Columns to normalise ──────────────────────────────────────────────────
    exclude = {"sID", "dID", "label", "category", "Timestamp", "Flow ID"}
    cols_to_norm = [
        c for c in df_train.columns
        if c not in exclude
        and not c.startswith(("Attack_", "Protocol"))
    ]

    # ── Fit NIDS scaler on G_train, then build graphs ────────────────────────
    nids_scaler = StandardScaler()
    nids_scaler.fit(df_train[cols_to_norm])
    
    # Create metadata dictionary
    suffix = "_undersample" if apply_undersampling else ""
    split_tag = f"t{int(train_frac*100)}s{int(surr_frac*100)}e{int(100-train_frac*100-surr_frac*100)}"
    sa_tag = "_sa" if stratify_attack else ""
    
    metadata = {
        'classes_def': classes,
        'apply_undersampling': apply_undersampling,
        'seed': seed,
        'train_frac': train_frac,
        'surr_frac': surr_frac,
        'stratify_attack': stratify_attack,
        'feature_names': list(cols_to_norm),
        'label_names': label_names,
        'normalized_columns': list(cols_to_norm),
        'split_tag': split_tag + sa_tag,
        'suffix': suffix
    }

    G_train = _build_cicids_partition_graph(
        df_train, classes, nids_scaler, cols_to_norm, label_names, "train", metadata
    )
    G_surr = _build_cicids_partition_graph(
        df_surr, classes, nids_scaler, cols_to_norm, label_names, "surr", metadata
    )
    G_test = _build_cicids_partition_graph(
        df_test, classes, nids_scaler, cols_to_norm, label_names, "test", metadata
    )

    # ── Save shared cache artefacts ───────────────────────────────────────────
    cache_dir = osp.join(osp.dirname(path_raw), "cache")
    os.makedirs(cache_dir, exist_ok=True)

    # Save with CONSISTENT NAMING
    torch.save(label_names,
               osp.join(cache_dir, f"label_names_CIC2017_3way_{classes}{suffix}_seed{seed}.pth"))
    torch.save(
        {"scaler": nids_scaler, "normalized_columns": cols_to_norm},
        osp.join(cache_dir, f"scaler_CIC2017_3way_{classes}{suffix}_seed{seed}.pth"),
    )
    # Save feature column names - use 'feature_names'
    torch.save(
        cols_to_norm,
        osp.join(cache_dir, f"feature_names_CIC2017_3way_{classes}{suffix}_seed{seed}.pth"),
    )
    if classes == "binary":
        torch.save(le_cat.classes_,
                   osp.join(cache_dir, f"encoder_CIC2017_3way_category_{classes}{suffix}_seed{seed}.pth"))

    logger.info("CIC-IDS-2017 three-way pipeline complete")
    return G_train, G_surr, G_test

# CICIDS2017HeterogeneousGraphThreeWay

class CICIDS2017HeterogeneousGraphThreeWay(Dataset):
    """CIC-IDS-2017 dataset with three independent temporal partition graphs.

    * ``get_dgl(0)`` → G_train (NIDS victim, NIDS scaler)
    * ``get_dgl(1)`` → G_surr  (attacker surrogate, own scaler)
    * ``get_dgl(2)`` → G_test  (evaluation, NIDS scaler)
    * ``get_dgl(3)`` → G_merged (train + test flows, for merged evaluation mode)
    """

    def __init__(
        self,
        root: str,
        classes_def: str,
        apply_undersampling: bool = True,
        seed: int = 42,
        train_frac: float = 0.4,
        surr_frac: float = 0.4,
        stratify_attack: bool = False,
        gaussian_init: bool = True,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.classes_def         = classes_def
        self.apply_undersampling = apply_undersampling
        self.seed                = seed
        self.train_frac          = train_frac
        self.surr_frac           = surr_frac
        self.stratify_attack     = stratify_attack
        self.gaussian_init       = gaussian_init
        self._suffix             = ""
        super().__init__(root, transform, pre_transform, pre_filter)
        self.name = f"CICIDS2017ThreeWay_{classes_def}"

    def _split_tag(self) -> str:
        tr = int(self.train_frac * 100)
        sr = int(self.surr_frac  * 100)
        te = 100 - tr - sr
        sa = "_sa" if self.stratify_attack else ""
        return f"t{tr}s{sr}e{te}{sa}"

    def _nx_fname(self, partition: str) -> str:
        suffix_part = f"{self._suffix}_" if self._suffix else ""
        return (
            f"hetero_graph_CIC2017_{self.classes_def}"
            f"{suffix_part}{self._split_tag()}_{partition}_seed{self.seed}.pt"
        )

    def _dgl_fname(self, partition: str) -> str:
        suffix_part = f"{self._suffix}_" if self._suffix else ""
        return (
            f"hetero_graph_CIC2017_{self.classes_def}"
            f"{suffix_part}{self._split_tag()}_{partition}_dgl_seed{self.seed}.pt"
        )

    @property
    def raw_file_names(self):
        return ["Friday-WorkingHours.csv"]

    def download(self):
        raise NotImplementedError("Download not implemented — provide raw CSVs manually")

    @property
    def processed_file_names(self):
        # 4 partitions: train, surr, test, merged
        return [self._nx_fname(p) for p in ("train", "surr", "test", "merged")]

    def _convert_and_save_partition(self, nx_graph: nx.Graph, partition: str) -> None:
        """Convert one NX partition graph to DGL and persist both."""
        dgl_graph, node_mappings = utils.convert_networkx_to_dgl_heterogeneous(nx_graph)
        flow_to_idx     = node_mappings["flow"]
        endpoint_to_idx = node_mappings["endpoint"]

        # Timestamps
        sorted_flow_nodes = sorted(flow_to_idx, key=flow_to_idx.get)
        timestamps = [nx_graph.nodes[n]["timestamp"] for n in sorted_flow_nodes]
        if timestamps and isinstance(timestamps[0], (str, pandas.Timestamp)):
            timestamps = [pandas.Timestamp(ts).value / 1e9 for ts in timestamps]
        dgl_graph.nodes["flow"].data["timestamp"] = torch.tensor(
            timestamps, dtype=torch.float64
        )

        # Endpoint metadata
        endpoint_nodes = sorted(endpoint_to_idx, key=endpoint_to_idx.get)
        local_prefixes = ["192.168", "10.", "172."]
        local_indicator = [
            any(n.startswith(p) for p in local_prefixes) for n in endpoint_nodes
        ]
        dgl_graph.local_indicator = torch.tensor(local_indicator)
        nx_graph.local_indicator  = torch.tensor(local_indicator)
        nx_graph.node_mapping     = endpoint_to_idx

        is_src = [nx_graph.nodes[n].get("is_source",      False) for n in endpoint_nodes]
        is_dst = [nx_graph.nodes[n].get("is_destination", False) for n in endpoint_nodes]
        dgl_graph.nodes["endpoint"].data["is_source"]      = torch.tensor(is_src,  dtype=torch.bool)
        dgl_graph.nodes["endpoint"].data["is_destination"] = torch.tensor(is_dst,  dtype=torch.bool)
        dgl_graph.endpoint_real_ids = [
            nx_graph.nodes[n]["real_id"] for n in endpoint_nodes
        ]
        
        # Attach metadata if available
        if hasattr(nx_graph, 'metadata'):
            dgl_graph.metadata = nx_graph.metadata

        torch.save(nx_graph,  osp.join(self.processed_dir, self._nx_fname(partition)))
        torch.save(dgl_graph, osp.join(self.processed_dir, self._dgl_fname(partition)))

    def _create_merged_graph_dgl(self, g_train_dgl: dgl.DGLGraph, g_test_dgl: dgl.DGLGraph) -> dgl.DGLGraph:
        """
        Create merged graph using direct DGL operations (no NetworkX conversion).
        This is much faster and avoids the bottleneck of converting to NetworkX.
        """
        import copy
        import torch
        
        log = logging.getLogger(__name__)
        
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
        # Get all edges from train graph
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
            log.warning(f"Error adding train edges: {e}")
        
        # 4. Ensure endpoint metadata is present for extra train endpoints (if any)
        # The merged graph already has all endpoints from test graph
        # We need to ensure endpoint attributes are properly set
        # Get train endpoint IDs
        train_endpoint_ids = []
        if hasattr(g_train_dgl, 'endpoint_real_ids'):
            train_endpoint_ids = set(g_train_dgl.endpoint_real_ids)
        else:
            # Fallback: get from node data
            pass
        
        # Get test endpoint IDs
        test_endpoint_ids = set()
        if hasattr(g_test_dgl, 'endpoint_real_ids'):
            test_endpoint_ids = set(g_test_dgl.endpoint_real_ids)
        
        # Identify endpoints that appear only in train
        train_only_endpoints = train_endpoint_ids - test_endpoint_ids
        
        if train_only_endpoints:
            log.info(f"  Found {len(train_only_endpoints)} train-only endpoints")
            # We need to add these endpoints to the merged graph
            # This requires extracting their features and attributes
            # For now, we rely on the fact that endpoints are shared across graphs
            pass
        
        # Ensure all endpoint attributes are consistent
        endpoint_attrs = ['is_source', 'is_destination', 'is_internal']
        for attr in endpoint_attrs:
            if attr in g_train_dgl.nodes['endpoint'].data and attr not in g_merged.nodes['endpoint'].data:
                # Initialize with False for all endpoints
                g_merged.nodes['endpoint'].data[attr] = torch.zeros(
                    g_merged.num_nodes('endpoint'), dtype=torch.bool, device=g_merged.device
                )
        
        log.info(f"  Merged graph: {g_merged.num_nodes('flow')} flows, "
                 f"{g_merged.num_nodes('endpoint')} endpoints")
        
        return g_merged

    def process(self):
        logger.info(
            f"Processing CIC-IDS-2017 three-way — "
            f"train={self.train_frac:.0%} surr={self.surr_frac:.0%}"
        )
        
        # 1. Build the three base graphs (NetworkX then DGL)
        G_train, G_surr, G_test = pipeline_create_heterogeneous_cicids2017_three_way(
            path_raw=self.raw_dir,
            classes=self.classes_def,
            apply_undersampling=self.apply_undersampling,
            seed=self.seed,
            train_frac=self.train_frac,
            surr_frac=self.surr_frac,
            stratify_attack=self.stratify_attack,
        )
        
        # 2. Save the three base partitions (NetworkX and DGL)
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
        
        # Create a minimal NetworkX graph for metadata (optional, for compatibility)
        import networkx as nx
        G_merged_nx = nx.MultiGraph()
        G_merged_nx.partition = "merged"
        G_merged_nx.label_names = G_train.label_names if hasattr(G_train, 'label_names') else ["BENIGN", "ATTACK"]
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
        
        logger.info("CIC-IDS-2017 three-way processing complete with merged graph")

    def len(self) -> int:
        """Return number of partitions (4: train, surr, test, merged)."""
        return 4

    _PARTITION_NAMES = ("train", "surr", "test", "merged")

    def _cache_dir(self) -> str:
        return osp.join(osp.dirname(self.raw_dir), "cache")

    def _load_shared_metadata(self, dgl_graph: "dgl.DGLGraph") -> "dgl.DGLGraph":
        """Attach feature names, label encoder, and scaler to a DGL graph."""
        cache_dir = self._cache_dir()
        suffix = self._suffix

        # Load feature names
        feat_path = osp.join(
            cache_dir,
            f"feature_names_CIC2017_3way_{self.classes_def}{suffix}_seed{self.seed}.pth"
        )
        if osp.exists(feat_path):
            dgl_graph.feature_names = torch.load(feat_path)
        else:
            dgl_graph.feature_names = None
            logger.warning(f"Feature names not found at {feat_path}")

        # Load label names
        label_path = osp.join(
            cache_dir,
            f"label_names_CIC2017_3way_{self.classes_def}{suffix}_seed{self.seed}.pth"
        )
        if osp.exists(label_path):
            dgl_graph.label_names = torch.load(label_path)

        # Load category encoder
        cat_path = osp.join(
            cache_dir,
            f"encoder_CIC2017_3way_category_{self.classes_def}{suffix}_seed{self.seed}.pth"
        )
        dgl_graph.encoder_category = torch.load(cat_path) if osp.exists(cat_path) else None

        # Load scaler
        scaler_path = osp.join(
            cache_dir,
            f"scaler_CIC2017_3way_{self.classes_def}{suffix}_seed{self.seed}.pth"
        )
        if osp.exists(scaler_path):
            sd = torch.load(scaler_path)
            dgl_graph.scaler             = sd["scaler"]
            dgl_graph.normalized_columns = sd["normalized_columns"]
        else:
            dgl_graph.scaler             = None
            dgl_graph.normalized_columns = None
            logger.warning(f"NIDS scaler not found at {scaler_path}")

        return dgl_graph

    def get(self, idx: int):
        partition = self._PARTITION_NAMES[idx]
        nx_graph = torch.load(osp.join(self.processed_dir, self._nx_fname(partition)))
        return nx_graph

    def get_dgl(self, idx: int) -> "dgl.DGLGraph":
        partition = self._PARTITION_NAMES[idx]
        g = torch.load(osp.join(self.processed_dir, self._dgl_fname(partition)))
        
        # Apply Gaussian initialization if requested
        if self.gaussian_init:
            if 'endpoint' in g.ntypes and 'h' in g.nodes['endpoint'].data:
                num_endpoints = g.num_nodes('endpoint')
                feature_dim = g.nodes['endpoint'].data['h'].shape[1]
                torch.manual_seed(self.seed + idx)
                gaussian_features = torch.randn(num_endpoints, feature_dim)
                g.nodes['endpoint'].data['h'] = gaussian_features
                logger.debug(f"Initialized endpoints with N(0,1) features (dim={feature_dim})")
        
        return self._load_shared_metadata(g)