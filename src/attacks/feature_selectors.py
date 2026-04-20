"""Feature pool construction and selection for attacks.

Pool construction strategies:
  - "all": use all flows (no filtering)
  - "random": randomly select k flows
  - "centroid": use the flow feature closest to the mean (returns single vector)
  - "cluster_all": cluster ALL flows (benign + malicious) and return k medoids

Feature selection during attack (from the pool):
  - "best": highest cosine similarity (same label only)
  - "worst_positive": lowest positive cosine similarity (same label only)
  - "random": uniform random over entire pool
  - "random_same_label": uniform random over same-label features only
"""

from typing import Optional, Union, Tuple, List
import torch
import numpy as np
from tqdm import tqdm

from src.utils.logger import setup_logging
setup_logging()
import logging
logger = logging.getLogger(__name__)


class FeaturePoolBuilder:
    """Build a feature pool from graph flows."""

    def __init__(self, graph, device: Union[str, torch.device] = "cpu"):
        self.graph = graph
        self.device = device

    def _get_mask(self, mask_name: Optional[str], indices_device) -> Optional[torch.Tensor]:
        """Helper to retrieve mask."""
        if not mask_name:
            return None
        
        mask = None
        try:
            if "flow" in self.graph.ntypes:
                mask = self.graph.nodes["flow"].data.get(mask_name)
            else:
                mask = self.graph.ndata.get(mask_name)
        except Exception:
            pass

        if mask is None:
            return None

        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.bool, device=self.device)
        
        if mask.dtype != torch.bool:
            mask = mask.bool()
        
        return mask.to(indices_device)

    def _find_best_k(
        self,
        X: np.ndarray,
        k_min: int = 5,
        k_max: int = 30,
        sample_size: int = 10_000,
        random_state: int = 42,
    ) -> int:
        """Find best k using silhouette score."""
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.metrics import silhouette_score

        logger.info(f"Searching best k between {k_min} and {k_max}...")
        
        best_k, best_score = k_min, -1.0
        for k in tqdm(range(k_min, k_max + 1), desc="Silhouette sweep"):
            km = MiniBatchKMeans(n_clusters=k, random_state=random_state, n_init=5, batch_size=4096)
            labels_km = km.fit_predict(X)
            score = silhouette_score(
                X, labels_km,
                sample_size=min(sample_size, len(X)),
                random_state=random_state,
            )
            if score > best_score:
                best_score, best_k = score, k

        logger.info(f"Best k={best_k} (silhouette={best_score:.4f})")
        return best_k

    def build_pool(
        self,
        strategy: str,
        k: Optional[int] = None,
        mask_name: Optional[str] = None,
        label_filter: Optional[int] = None,
        k_min: int = 5,
        k_max: int = 30,
        seed: int = 42
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[int]]]]:
        """Build feature pool.
        
        Args:
            strategy: "all", "random", "random_all", "centroid", "cluster_all"
            k: Number of features (for random/cluster_all). Ignored by "random_all".
            mask_name: Optional mask (e.g., 'test_mask', 'val_mask')
            label_filter: 0=benign, 1=malicious, None=all flows. Ignored by "random_all".
            k_min, k_max: Range for auto k selection (only for cluster_all when k=None)
        
        Returns:
            - For "centroid": returns tensor (single vector), None, None
            - For others: returns (pool_tensor, indices_tensor, labels_list)
        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        import random
        random.seed(seed)
        # Get features and labels
        if "flow" in self.graph.ntypes:
            features = self.graph.nodes["flow"].data["h"]
            labels = self.graph.nodes["flow"].data.get("label")
        else:
            features = self.graph.ndata.get("h") or self.graph.ndata.get("features")
            labels = self.graph.ndata.get("label")
        
        if features is None:
            raise RuntimeError("Graph has no features")

        # Start with all indices
        indices = torch.arange(features.shape[0], device=self.device)
        logger.info(f"Total flows: {len(indices)}")
        
        # Apply label filter if specified, except for random_all which must keep the full pool.
        effective_label_filter = None if strategy == "random_all" else label_filter
        if effective_label_filter is not None and labels is not None:
            if labels.device != indices.device:
                labels = labels.to(indices.device)
            is_selected = (labels == effective_label_filter)
            indices = indices[is_selected]
            logger.info(f"After label filter (label={effective_label_filter}): {len(indices)} flows")
        
        # Apply mask if specified
        if mask_name is not None:
            mask = self._get_mask(mask_name, indices.device)
            if mask is not None:
                indices = indices[mask[indices]]
                logger.info(f"After mask '{mask_name}': {len(indices)} flows")
        
        if len(indices) == 0:
            raise RuntimeError(f"No flows found: strategy={strategy}, mask={mask_name}, label_filter={effective_label_filter}")
        
        # Special case: centroid (return single vector, not a pool)
        if strategy == "centroid":
            features_cpu = features[indices.to(features.device)]
            centroid = features_cpu.mean(dim=0, keepdim=True)
            logger.info("Built centroid (single feature vector)")
            return centroid, None, None
        
        # For pool-based strategies
        if strategy == "all":
            selected_indices = indices
            logger.info(f"Using all {len(selected_indices)} flows")
            
        elif strategy == "random":
            k_actual = k if k is not None else min(100, len(indices))
            k_actual = min(k_actual, len(indices))
            perm = torch.randperm(len(indices), generator=torch.Generator().manual_seed(seed))
            selected_indices = indices[perm[:k_actual]]
            logger.info(f"Random selection: {len(selected_indices)} flows (k={k_actual})")

        elif strategy == "random_all":
            selected_indices = indices
            logger.info(f"Random-all selection: using all {len(selected_indices)} valid flows")
        
        elif strategy == "cluster_all":
            # Cluster ALL flows (benign + malicious) and return k medoids
            k_actual = k
            try:
                from sklearn.cluster import MiniBatchKMeans
                
                features_cpu = features[indices.to(features.device)].detach().cpu().numpy()
                n_samples = features_cpu.shape[0]
                
                # Auto-select k if not provided
                if k_actual is None:
                    # For cluster_all, use smaller k values (6-20 as requested in report)
                    if n_samples < 1000:
                        k_actual = max(3, n_samples // 100)
                    elif n_samples < 10000:
                        k_actual = max(5, n_samples // 500)
                    else:
                        k_actual = max(6, min(20, n_samples // 1000))
                    logger.info(f"Auto-selected k={k_actual}")
                else:
                    k_actual = min(k_actual, n_samples // 2)
                    logger.info(f"Using k={k_actual}")
                
                if n_samples < k_actual:
                    logger.warning(f"n_samples={n_samples} < k={k_actual}, using all samples")
                    selected_indices = indices
                else:
                    km = MiniBatchKMeans(
                        n_clusters=k_actual, random_state=42, n_init=10, batch_size=4096
                    ).fit(features_cpu)
                    centers = km.cluster_centers_
                    
                    selected_list = []
                    for i in range(k_actual):
                        cluster_mask = (km.labels_ == i)
                        if not np.any(cluster_mask):
                            continue
                        X_c = features_cpu[cluster_mask]
                        dists = np.linalg.norm(X_c - centers[i], axis=1)
                        best_loc = int(np.argmin(dists))
                        # Map back to global index
                        subset_pos = np.where(cluster_mask)[0][best_loc]
                        selected_list.append(indices[subset_pos])
                    
                    if not selected_list:
                        logger.warning("Clustering failed, falling back to random")
                        perm = torch.randperm(len(indices))
                        k_actual = min(k_actual, len(indices))
                        selected_indices = indices[perm[:k_actual]]
                    else:
                        selected_indices = torch.stack(selected_list)
                
                logger.info(f"Cluster_all: {len(selected_indices)} medoids")
                
            except ImportError:
                logger.warning("sklearn not found, falling back to random")
                k_actual = k if k is not None else min(100, len(indices))
                k_actual = min(k_actual, len(indices))
                perm = torch.randperm(len(indices))
                selected_indices = indices[perm[:k_actual]]
            except Exception as e:
                logger.warning(f"Clustering failed: {e}, falling back to random")
                k_actual = k if k is not None else min(100, len(indices))
                k_actual = min(k_actual, len(indices))
                perm = torch.randperm(len(indices))
                selected_indices = indices[perm[:k_actual]]
        
        else:
            raise ValueError(f"Unknown pool strategy: {strategy}")
        
        # Build pool tensor and return with indices and labels
        selected_indices = selected_indices.to(features.device)
        pool = features[selected_indices].to(self.device)
        indices_cpu = selected_indices.to('cpu').long()
        
        # Get labels for selected features (for same-label selection)
        if labels is not None:
            labels_selected = labels[selected_indices.to(labels.device)]
            pool_labels = labels_selected.cpu().tolist()
        else:
            pool_labels = None
        
        logger.info(f"Built pool: strategy='{strategy}', size={len(pool)}")
        
        return pool, indices_cpu, pool_labels


class FeatureSelector:
    """Select features from a pool during attack."""
    
    def __init__(self, feature_pool: torch.Tensor, pool_labels: Optional[List[int]] = None):
        self.feature_pool = feature_pool
        self.pool_labels = pool_labels  # List of labels for each feature in pool
    
    def select(
        self,
        gradient: Optional[torch.Tensor],
        target_label: int,
        selection_mode: str = "best"
    ) -> Tuple[int, torch.Tensor]:
        import random
        
        if selection_mode == "random":
            idx = random.randint(0, len(self.feature_pool) - 1)
            return idx, self.feature_pool[idx]
        
        elif selection_mode == "random_same_label":
            if self.pool_labels is None:
                logger.warning("pool_labels not available, falling back to random")
                idx = random.randint(0, len(self.feature_pool) - 1)
                return idx, self.feature_pool[idx]
            
            same_label_indices = [i for i, lbl in enumerate(self.pool_labels) if lbl == target_label]
            if not same_label_indices:
                logger.warning(f"No features with label {target_label} in pool, falling back to random")
                idx = random.randint(0, len(self.feature_pool) - 1)
            else:
                idx = random.choice(same_label_indices)
            return idx, self.feature_pool[idx]
        
        elif selection_mode == "best":
            if gradient is None:
                logger.warning("Gradient is None for best selection, falling back to random")
                idx = random.randint(0, len(self.feature_pool) - 1)
                return idx, self.feature_pool[idx]
            
            grad_norm = torch.norm(gradient)
            if grad_norm < 1e-8:
                logger.warning("Gradient near zero, falling back to random")
                idx = random.randint(0, len(self.feature_pool) - 1)
                return idx, self.feature_pool[idx]
            
            grad_unit = gradient / grad_norm
            pool_norms = torch.norm(self.feature_pool, dim=1, keepdim=True)
            pool_unit = self.feature_pool / (pool_norms + 1e-8)
            similarities = torch.matmul(pool_unit, grad_unit)
            
            if self.pool_labels is not None:
                same_label_mask = torch.tensor([lbl == target_label for lbl in self.pool_labels], 
                                                device=similarities.device)
                if same_label_mask.any():
                    similarities = similarities.clone()
                    similarities[~same_label_mask] = -float('inf')
            
            valid_mask = torch.isfinite(similarities)
            if not valid_mask.any():
                logger.warning("No valid features for best selection, falling back to random")
                idx = random.randint(0, len(self.feature_pool) - 1)
            else:
                similarities[~valid_mask] = -float('inf')
                idx = torch.argmax(similarities).item()
            
            return idx, self.feature_pool[idx]
        
        else:
            raise ValueError(f"Unknown selection_mode: {selection_mode}")