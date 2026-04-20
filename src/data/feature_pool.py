# src/data/feature_pool.py
"""
Global feature pool manager for consistent feature selection across experiments.
"""

import os
import pickle
import hashlib
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

logger = logging.getLogger(__name__)


class GlobalFeaturePool:
    """
    Global feature pool that is built once from the entire dataset and reused across experiments.
    
    The pool is identified by:
    - dataset_name
    - pool_strategy (centroid, clustering, etc.)
    - pool_k (if applicable)
    - seed
    - classification mode (binary/category)
    - label_filter (0 for benign, 1 for malicious)
    
    The pool is cached to disk and loaded on subsequent runs.
    """
    
    def __init__(
        self,
        dataset_name: str,
        classes_def: str = "binary",
        pool_strategy: str = "centroid",
        pool_k: Optional[int] = None,
        label_filter: int = 0,  # 0 for benign, 1 for malicious
        seed: int = 42,
        dataset_path: Optional[str] = None,
        train_frac: float = 0.5,
        surr_frac: float = 0.3,
        cache_dir: Optional[str] = None,
    ):
        self.dataset_name = dataset_name
        self.classes_def = classes_def
        self.pool_strategy = pool_strategy
        self.pool_k = pool_k
        self.label_filter = label_filter
        self.seed = seed
        self.dataset_path = dataset_path
        self.train_frac = train_frac
        self.surr_frac = surr_frac
        
        # Set cache directory
        if cache_dir is None:
            cache_dir = str(Path.cwd() / ".cache")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate cache key
        self.cache_key = self._generate_cache_key()
        self.cache_path = self.cache_dir / f"feature_pool_{self.cache_key}.pt"
        
        # Load or build pool
        self._feature_pool = None
        self._pool_indices = None
        self._pool_metadata = None
        
        self._load_or_build()
    
    def _generate_cache_key(self) -> str:
        """Generate a unique cache key based on parameters."""
        key_dict = {
            'dataset': self.dataset_name,
            'classes': self.classes_def,
            'strategy': self.pool_strategy,
            'k': self.pool_k,
            'label': self.label_filter,
            'seed': self.seed,
            'dataset_path': self.dataset_path,
            'train_frac': self.train_frac,
            'surr_frac': self.surr_frac,
        }
        key_str = str(sorted(key_dict.items()))
        return hashlib.md5(key_str.encode()).hexdigest()[:16]
    
    def _load_or_build(self):
        """Load cached pool or build one."""
        if self.cache_path.exists():
            logger.info(f"Loading cached feature pool from {self.cache_path}")
            try:
                cached = torch.load(self.cache_path)
                self._feature_pool = cached['feature_pool']
                self._pool_indices = cached.get('pool_indices')
                self._pool_metadata = cached.get('metadata', {})
                logger.info(f"Loaded pool: {len(self._feature_pool)} vectors")
                return
            except Exception as e:
                logger.warning(f"Failed to load cached pool: {e}, rebuilding...")
        
        # Build pool
        logger.info(f"Building global feature pool for {self.dataset_name}")
        self._build_pool()
        
        # Save to cache
        self._save_to_cache()
    
    def _build_pool(self):
        """Build the feature pool from the entire dataset."""
        # Import here to avoid circular imports
        from src.data.loaders import load_dataset

        _, g_train, _, _ = load_dataset(
            self.dataset_name,
            self.classes_def,
            train_frac=self.train_frac,
            surr_frac=self.surr_frac,
            seed=self.seed,
            path=self.dataset_path,
        )

        g_full = g_train
        
        # Ensure we're on CPU for clustering
        g_full = g_full.cpu()
        
        # Get flow features and labels
        flow_features = g_full.nodes['flow'].data['h']
        flow_labels = g_full.nodes['flow'].data['label']
        
        # Filter by label if specified
        if self.label_filter is not None:
            mask = (flow_labels == self.label_filter)
            flow_features = flow_features[mask]
            logger.info(f"Filtered {mask.sum().item()} flows with label={self.label_filter}")
        
        n_flows = len(flow_features)
        logger.info(f"Building pool from {n_flows} flows")
        
        # Build pool based on strategy
        if self.pool_strategy == "centroid":
            self._build_centroid_pool(flow_features)
        elif self.pool_strategy == "clustering":
            self._build_clustering_pool(flow_features)
        elif self.pool_strategy == "random":
            self._build_random_pool(flow_features)
        elif self.pool_strategy == "all":
            self._build_all_pool(flow_features)
        else:
            raise ValueError(f"Unknown pool strategy: {self.pool_strategy}")
    
    def _build_centroid_pool(self, features: torch.Tensor):
        """Build pool using centroid (closest to mean)."""
        # Compute mean of all features
        mean_vec = features.mean(dim=0, keepdim=True)
        
        # Find flow closest to mean
        distances = torch.norm(features - mean_vec, dim=1)
        best_idx = torch.argmin(distances).item()
        
        self._feature_pool = features[best_idx:best_idx+1]
        self._pool_indices = torch.tensor([best_idx])
        
        self._pool_metadata = {
            'pool_size': 1,
            'strategy': 'centroid',
            'n_candidates': len(features),
            'mean_vector': mean_vec.squeeze().tolist(),
        }
        
        logger.info(f"Centroid pool: 1 vector selected from {len(features)} candidates")
    
    def _build_clustering_pool(self, features: torch.Tensor):
        """Build pool using k-means clustering."""
        # Determine number of clusters
        if self.pool_k is None:
            self.pool_k = self._auto_select_k(len(features))
            logger.info(f"Auto-selected k={self.pool_k} for clustering")
        
        n_samples = len(features)
        k = min(self.pool_k, n_samples)
        
        # Convert to numpy for sklearn
        features_np = features.numpy()
        
        # Set random seed for reproducibility
        np.random.seed(self.seed)
        
        # Perform k-means clustering
        logger.info(f"Running k-means with k={k} on {n_samples} samples...")
        kmeans = KMeans(
            n_clusters=k,
            random_state=self.seed,
            n_init=10,
            max_iter=300,
            verbose=0
        )
        kmeans.fit(features_np)
        
        # Find the closest point to each centroid (medoid)
        cluster_centers = kmeans.cluster_centers_
        cluster_labels = kmeans.labels_
        
        medoid_indices = []
        for cluster_id in range(k):
            # Get all points in this cluster
            cluster_mask = (cluster_labels == cluster_id)
            cluster_points = features_np[cluster_mask]
            
            if len(cluster_points) == 0:
                continue
            
            # Find point closest to centroid
            centroid = cluster_centers[cluster_id]
            distances = np.linalg.norm(cluster_points - centroid, axis=1)
            closest_idx_in_cluster = np.argmin(distances)
            
            # Map back to original indices
            original_indices = np.where(cluster_mask)[0]
            medoid_indices.append(original_indices[closest_idx_in_cluster])
        
        # Convert back to torch
        self._pool_indices = torch.tensor(medoid_indices)
        self._feature_pool = features[self._pool_indices]
        
        self._pool_metadata = {
            'pool_size': len(medoid_indices),
            'strategy': 'clustering',
            'k_requested': self.pool_k,
            'k_actual': len(medoid_indices),
            'n_candidates': n_samples,
            'inertia': float(kmeans.inertia_),
            'cluster_sizes': [int(np.sum(cluster_labels == i)) for i in range(k)],
        }
        
        logger.info(f"Clustering pool: {len(medoid_indices)} medoids selected from {n_samples} candidates")
    
    def _build_random_pool(self, features: torch.Tensor):
        """Build pool by random sampling."""
        n_samples = len(features)
        k = self.pool_k if self.pool_k is not None else min(100, n_samples)
        k = min(k, n_samples)
        
        # Set random seed
        torch.manual_seed(self.seed)
        
        # Random sample
        indices = torch.randperm(n_samples)[:k]
        self._pool_indices = indices
        self._feature_pool = features[indices]
        
        self._pool_metadata = {
            'pool_size': k,
            'strategy': 'random',
            'n_candidates': n_samples,
        }
        
        logger.info(f"Random pool: {k} vectors sampled from {n_samples} candidates")
    
    def _build_all_pool(self, features: torch.Tensor):
        """Use all flows as the pool."""
        self._feature_pool = features
        self._pool_indices = torch.arange(len(features))
        
        self._pool_metadata = {
            'pool_size': len(features),
            'strategy': 'all',
            'n_candidates': len(features),
        }
        
        logger.info(f"All pool: {len(features)} vectors")
    
    def _auto_select_k(self, n_samples: int) -> int:
        """Auto-select number of clusters based on dataset size."""
        if n_samples > 500000:
            return 300
        elif n_samples > 200000:
            return 250
        elif n_samples > 100000:
            return 200
        elif n_samples > 50000:
            return 150
        elif n_samples > 20000:
            return 100
        elif n_samples > 10000:
            return 50
        elif n_samples > 5000:
            return 30
        elif n_samples > 1000:
            return 20
        else:
            return max(10, n_samples // 50)
    
    def _save_to_cache(self):
        """Save pool to cache."""
        cache_data = {
            'feature_pool': self._feature_pool,
            'pool_indices': self._pool_indices,
            'metadata': self._pool_metadata,
            'config': {
                'dataset_name': self.dataset_name,
                'classes_def': self.classes_def,
                'pool_strategy': self.pool_strategy,
                'pool_k': self.pool_k,
                'label_filter': self.label_filter,
                'seed': self.seed,
                'dataset_path': self.dataset_path,
                'train_frac': self.train_frac,
                'surr_frac': self.surr_frac,
            }
        }
        
        torch.save(cache_data, self.cache_path)
        logger.info(f"Saved feature pool to {self.cache_path}")
        logger.info(f"Pool metadata: {self._pool_metadata}")
    
    @property
    def feature_pool(self) -> torch.Tensor:
        """Get the feature pool tensor."""
        return self._feature_pool
    
    @property
    def pool_indices(self) -> Optional[torch.Tensor]:
        """Get the indices of selected flows in the original dataset."""
        return self._pool_indices
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Get pool metadata."""
        return self._pool_metadata
    
    @property
    def size(self) -> int:
        """Get pool size."""
        return len(self._feature_pool) if self._feature_pool is not None else 0


# Singleton for global pool management
_GLOBAL_POOL_CACHE = {}


def get_global_feature_pool(
    dataset_name: str,
    classes_def: str = "binary",
    pool_strategy: str = "centroid",
    pool_k: Optional[int] = None,
    label_filter: int = 0,
    seed: int = 42,
    dataset_path: Optional[str] = None,
    train_frac: float = 0.5,
    surr_frac: float = 0.3,
    force_rebuild: bool = False,
) -> GlobalFeaturePool:
    """
    Get or create a global feature pool.
    
    This function maintains a cache of pools to avoid rebuilding multiple times
    in the same process.
    """
    cache_key = (
        f"{dataset_name}_{classes_def}_{pool_strategy}_{pool_k}_{label_filter}_{seed}_"
        f"{dataset_path}_{train_frac}_{surr_frac}"
    )
    
    if not force_rebuild and cache_key in _GLOBAL_POOL_CACHE:
        return _GLOBAL_POOL_CACHE[cache_key]
    
    pool = GlobalFeaturePool(
        dataset_name=dataset_name,
        classes_def=classes_def,
        pool_strategy=pool_strategy,
        pool_k=pool_k,
        label_filter=label_filter,
        seed=seed,
        dataset_path=dataset_path,
        train_frac=train_frac,
        surr_frac=surr_frac,
    )
    
    _GLOBAL_POOL_CACHE[cache_key] = pool
    return pool