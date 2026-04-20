"""
Feature scaling utilities for graph datasets.

Provides inverse transformation capabilities to convert standardized features
back to original scale for interpretability and adversarial analysis.
"""

import torch
import logging
from typing import Optional, List

logger = logging.getLogger(__name__)


def inverse_transform_features(
    features_tensor: torch.Tensor,
    scaler,
    normalized_columns: Optional[List[str]] = None,
    feature_names: Optional[List[str]] = None
) -> torch.Tensor:
    """
    Convert standardized features back to original scale using saved scaler.
    
    This function enables interpretation of adversarial perturbations and model
    predictions in original feature space, which is critical for security analysis.
    
    Args:
        features_tensor: Standardized feature tensor (shape: [num_samples, num_features])
        scaler: Fitted StandardScaler object with learned mean_ and scale_ parameters
        normalized_columns: Names of columns that were normalized (optional, for logging)
        feature_names: All feature names in order (optional, for validation)
        
    Returns:
        torch.Tensor: Features in original scale with same shape and device as input
        
    Raises:
        ValueError: If scaler is None or incompatible with feature dimensions
        
    Rationale:
        Adversarial perturbations are computed in standardized space for numerical
        stability (unit variance ensures gradient scales are comparable across features).
        However, security analysts require original-scale values to interpret attack
        semantics and assess real-world impact.
        
        Examples of use cases:
        1. Attack analysis: "Attacker increased packet size by 500 bytes" 
           vs. uninformative "increased z-score by 0.3"
        2. Feature importance: SHAP/gradient values in original units for domain experts
        3. Adversarial example generation: Perturbations constrained to realistic ranges
        4. Cross-dataset comparison: Attack effectiveness measured in comparable units
        
    Example:
        >>> # Load dataset with scaler
        >>> dataset = CICIDSHeterogeneousGraph(path, "binary")
        >>> data = dataset.get(0)
        >>> 
        >>> # Extract features from attacked graph
        >>> attacked_features = attacked_graph.nodes['flow'].data['h']
        >>> 
        >>> # Convert to original scale for analysis
        >>> original_features = inverse_transform_features(
        ...     attacked_features,
        ...     data.scaler,
        ...     data.normalized_columns,
        ...     data.features_name
        ... )
        >>> 
        >>> # Compute perturbation magnitude in original units
        >>> perturbation = original_features - original_benign_features
        >>> print(f"Average packet size increase: {perturbation[:, packet_size_idx].mean():.2f} bytes")
    """
    if scaler is None:
        raise ValueError(
            "Scaler not available - graph was processed without saving scaler. "
            "Re-run graph processing with updated code to enable inverse transformation."
        )
    
    # Validate tensor dimensions
    if features_tensor.dim() not in [1, 2]:
        raise ValueError(
            f"Expected 1D or 2D tensor, got shape {features_tensor.shape}. "
            f"Features should be [num_samples, num_features] or [num_features]."
        )
    
    # Handle 1D input (single sample)
    squeeze_output = False
    if features_tensor.dim() == 1:
        features_tensor = features_tensor.unsqueeze(0)
        squeeze_output = True
    
    # Validate feature dimension matches scaler
    num_features = features_tensor.shape[1]
    expected_features = len(scaler.mean_)
    if num_features != expected_features:
        # Check if this is a partial feature set (only normalized columns)
        if normalized_columns is not None and num_features == len(normalized_columns):
            logger.debug(
                f"Input contains {num_features} normalized features "
                f"(subset of {expected_features} total features)"
            )
        else:
            raise ValueError(
                f"Feature dimension mismatch: tensor has {num_features} features, "
                f"scaler expects {expected_features}. Ensure features are in correct order."
            )
    
    # Convert tensor to numpy for sklearn compatibility
    device = features_tensor.device
    dtype = features_tensor.dtype
    features_np = features_tensor.cpu().detach().numpy()
    
    # Apply inverse transformation: X_original = X_standardized * scale_ + mean_
    # This reverses the standardization: X_standardized = (X_original - mean_) / scale_
    try:
        features_original = scaler.inverse_transform(features_np)
    except Exception as e:
        logger.error(f"Inverse transformation failed: {e}")
        logger.error(f"Scaler mean shape: {scaler.mean_.shape}, scale shape: {scaler.scale_.shape}")
        logger.error(f"Input shape: {features_np.shape}")
        raise
    
    # Convert back to tensor on original device with original dtype
    result = torch.tensor(features_original, dtype=dtype, device=device)
    
    # Log transformation statistics for validation
    if logger.isEnabledFor(logging.DEBUG):
        original_mean = result.mean().item()
        original_std = result.std().item()
        standardized_mean = features_tensor.mean().item()
        standardized_std = features_tensor.std().item()
        
        logger.debug(
            f"Inverse transform: standardized (mean={standardized_mean:.4f}, std={standardized_std:.4f}) "
            f"-> original (mean={original_mean:.4f}, std={original_std:.4f})"
        )
        
        # Verify transformation is correct (standardized values should have ~0 mean, ~1 std if from full dataset)
        if abs(standardized_mean) > 0.1 or abs(standardized_std - 1.0) > 0.2:
            logger.warning(
                f"Standardized features have unexpected statistics. "
                f"Expected ~N(0,1), got N({standardized_mean:.4f}, {standardized_std:.4f}). "
                f"This may indicate: (1) subset of data, (2) attacked features, or (3) data drift."
            )
    
    # Squeeze back to 1D if input was 1D
    if squeeze_output:
        result = result.squeeze(0)
    
    return result


def get_feature_perturbation_stats(
    original_features: torch.Tensor,
    perturbed_features: torch.Tensor,
    feature_names: Optional[List[str]] = None,
    top_k: int = 10
) -> dict:
    """
    Compute statistics on feature perturbations for adversarial analysis.
    
    Args:
        original_features: Original feature values (shape: [num_samples, num_features])
        perturbed_features: Perturbed feature values (shape: [num_samples, num_features])
        feature_names: Names of features for reporting (optional)
        top_k: Number of top perturbed features to report (default: 10)
        
    Returns:
        dict: Perturbation statistics including:
            - mean_perturbation: Average absolute change per feature
            - max_perturbation: Maximum absolute change per feature
            - top_features: Indices and names of most perturbed features
            - l2_norm: L2 norm of perturbation vector per sample
            - linf_norm: L-infinity norm per sample
            
    Example:
        >>> stats = get_feature_perturbation_stats(
        ...     benign_features_original_scale,
        ...     attacked_features_original_scale,
        ...     feature_names=dataset.features_name
        ... )
        >>> print(f"Most perturbed features: {stats['top_features']}")
        >>> print(f"Average L2 perturbation: {stats['l2_norm'].mean():.2f}")
    """
    # Compute perturbation (difference)
    perturbation = perturbed_features - original_features
    
    # Per-feature statistics
    mean_pert_per_feature = perturbation.abs().mean(dim=0)
    max_pert_per_feature = perturbation.abs().max(dim=0).values
    
    # Get top-k most perturbed features
    top_k = min(top_k, perturbation.shape[1])
    top_indices = mean_pert_per_feature.topk(top_k).indices
    
    top_features = []
    for idx in top_indices:
        feature_info = {
            'index': idx.item(),
            'mean_perturbation': mean_pert_per_feature[idx].item(),
            'max_perturbation': max_pert_per_feature[idx].item(),
        }
        if feature_names is not None and idx < len(feature_names):
            feature_info['name'] = feature_names[idx]
        top_features.append(feature_info)
    
    # Per-sample norms
    l2_norm = torch.norm(perturbation, p=2, dim=1)
    linf_norm = torch.norm(perturbation, p=float('inf'), dim=1)
    
    return {
        'mean_perturbation_per_feature': mean_pert_per_feature,
        'max_perturbation_per_feature': max_pert_per_feature,
        'top_features': top_features,
        'l2_norm': l2_norm,
        'linf_norm': linf_norm,
        'num_samples': perturbation.shape[0],
        'num_features': perturbation.shape[1]
    }
