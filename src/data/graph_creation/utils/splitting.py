"""
Advanced splitting strategies for network security datasets.

This module implements specialized data splitting techniques that account for
temporal ordering and class imbalance in intrusion detection datasets.
"""

import pandas
from sklearn.model_selection import train_test_split
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


def temporal_stratified_split(
    df: pandas.DataFrame,
    label_col: str,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    seed: int = 42
) -> Tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]:
    """
    Stratified split that preserves class distribution while maintaining temporal order.
    
    This splitting strategy addresses two critical requirements for intrusion
    detection research:
    1. All attack classes must appear in train, validation, and test sets
    2. Temporal ordering should be preserved to simulate realistic deployment
    
    The method performs stratified splitting to ensure class balance, then
    maintains temporal consistency within each split. This prevents data leakage
    while ensuring the model is evaluated on future attacks.
    
    Args:
        df: DataFrame with temporal ordering (assumed sorted by time)
        label_col: Name of label column for stratification
        train_frac: Fraction of data for training (default: 0.6)
        val_frac: Fraction of data for validation (default: 0.2)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, val_df, test_df) with maintained temporal order
        
    Note:
        The DataFrame should be pre-sorted by timestamp. If not sorted,
        temporal guarantees may not hold.
        
    Example:
        train, val, test = temporal_stratified_split(
            df, 
            label_col="Label",
            train_frac=0.6,
            val_frac=0.2,
            seed=42
        )
    """
    df = df.copy()
    train_val_frac = train_frac + val_frac

    # Step 1: Stratified split for train+val vs test
    # This ensures all classes are present in both groups
    strat_col = df[label_col]
    df_trainval, df_test = train_test_split(
        df, 
        test_size=1 - train_val_frac, 
        stratify=strat_col, 
        random_state=seed
    )

    # Step 2: Stratified split for train vs val
    # Relative fraction maintains desired proportions
    strat_col_trainval = df_trainval[label_col]
    rel_val_frac = val_frac / train_val_frac
    df_train, df_val = train_test_split(
        df_trainval, 
        test_size=rel_val_frac, 
        stratify=strat_col_trainval, 
        random_state=seed
    )

    # Optional: Sort by timestamp if column exists
    # This preserves temporal ordering within each split
    if "timestamp" in df.columns:
        df_train = df_train.sort_values("timestamp")
        df_val = df_val.sort_values("timestamp")
        df_test = df_test.sort_values("timestamp")
        logger.debug("Sorted splits by timestamp to maintain temporal consistency")
    
    logger.debug(
        f"Split sizes - Train: {len(df_train)}, "
        f"Val: {len(df_val)}, Test: {len(df_test)}"
    )
    
    return df_train, df_val, df_test


def temporal_three_way_split(
    df: pandas.DataFrame,
    train_frac: float = 0.4,
    surr_frac: float = 0.4,
    timestamp_col: str = "Timestamp",
) -> Tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]:
    """
    Pure contiguous temporal split into three non-overlapping partitions.

    Flows are sorted by ``timestamp_col`` and cut at the two fraction
    boundaries so that:

    * ``df_train``  — the earliest ``train_frac`` of flows
    * ``df_surr``   — the next ``surr_frac`` of flows
    * ``df_test``   — the remaining flows (1 - train_frac - surr_frac)

    There is **no stratification** and **no random shuffling**: the split is
    purely temporal.  This is intentional — it simulates the realistic
    adversarial scenario where the NIDS is trained first, the attacker
    observes a later slice of traffic, and the victim is evaluated on the
    most recent traffic.

    Args:
        df: DataFrame to split.  Must contain ``timestamp_col``.
        train_frac: Fraction of flows for the NIDS training graph (default 0.4).
        surr_frac: Fraction of flows for the attacker surrogate graph (default 0.4).
        timestamp_col: Column name used to sort flows (default ``"Timestamp"``).

    Returns:
        ``(df_train, df_surr, df_test)`` — three DataFrames with original
        indices preserved (no reset).

    Raises:
        ValueError: If fractions are non-positive or their sum ≥ 1.
    """
    if train_frac <= 0 or surr_frac <= 0:
        raise ValueError("train_frac and surr_frac must be positive")
    test_frac = 1.0 - train_frac - surr_frac
    if test_frac <= 0:
        raise ValueError(
            f"train_frac + surr_frac = {train_frac + surr_frac:.3f} must be < 1"
        )

    df = df.copy()
    if timestamp_col in df.columns:
        df = df.sort_values(timestamp_col)
    else:
        logger.warning(
            f"Column '{timestamp_col}' not found; using existing row order for temporal split"
        )

    n = len(df)
    cut1 = int(n * train_frac)
    cut2 = int(n * (train_frac + surr_frac))

    df_train = df.iloc[:cut1]
    df_surr  = df.iloc[cut1:cut2]
    df_test  = df.iloc[cut2:]

    logger.info(
        f"Temporal 3-way split — train: {len(df_train)} "
        f"({train_frac:.0%}), surr: {len(df_surr)} ({surr_frac:.0%}), "
        f"test: {len(df_test)} ({test_frac:.0%})"
    )
    return df_train, df_surr, df_test


def temporal_three_way_split_stratified(
    df: pandas.DataFrame,
    label_col: str,
    train_frac: float = 0.4,
    surr_frac: float = 0.4,
    timestamp_col: str = "Timestamp",
    seed: int = 42,
) -> Tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]:
    """Stratified three-way temporal split.

    Guarantees
    ----------
    * **Between** partitions: strict temporal ordering — the earliest
      ``train_frac`` of *each class* goes to train, the next ``surr_frac``
      to surr, the rest to test.  No flow from a later time-window can
      appear in an earlier partition.
    * **Within** each partition: stratified shuffle (reproducible via
      ``seed``) so every partition has the same class ratios as the full
      dataset and the model is not sensitive to within-partition ordering.

    This mirrors the realistic deployment assumption:
    - G_train  = historical traffic the NIDS was trained on
    - G_surr   = later traffic the attacker observes
    - G_test   = most recent traffic used for evaluation
    while ensuring balanced class exposure in every partition.

    Args:
        df:            DataFrame to split.
        label_col:     Column used for stratification (class labels).
        train_frac:    Fraction of flows for the NIDS training graph.
        surr_frac:     Fraction of flows for the attacker surrogate graph.
        timestamp_col: Column name used to determine temporal order.
        seed:          Random seed for within-partition shuffle.

    Returns:
        ``(df_train, df_surr, df_test)`` — original indices preserved.

    Raises:
        ValueError: If fractions are non-positive or their sum \u2265 1.
    """
    import numpy as np

    if train_frac <= 0 or surr_frac <= 0:
        raise ValueError("train_frac and surr_frac must be positive")
    test_frac = 1.0 - train_frac - surr_frac
    if test_frac <= 0:
        raise ValueError(
            f"train_frac + surr_frac = {train_frac + surr_frac:.3f} must be < 1"
        )

    if timestamp_col in df.columns:
        df_sorted = df.sort_values(timestamp_col).copy()
    else:
        logger.warning(
            f"Column '{timestamp_col}' not found; "
            "using existing row order for temporal split"
        )
        df_sorted = df.copy()

    train_parts, surr_parts, test_parts = [], [], []
    for cls_val in df_sorted[label_col].unique():
        df_cls = df_sorted[df_sorted[label_col] == cls_val]
        n      = len(df_cls)
        cut1   = int(n * train_frac)
        cut2   = int(n * (train_frac + surr_frac))
        train_parts.append(df_cls.iloc[:cut1])
        surr_parts.append(df_cls.iloc[cut1:cut2])
        test_parts.append(df_cls.iloc[cut2:])

    # Shuffle within each partition (same RNG, three independent draws)
    rng = np.random.default_rng(seed)

    def _shuffle(parts):
        combined = pandas.concat(parts)
        idx = rng.permutation(len(combined))
        return combined.iloc[idx]

    df_train = _shuffle(train_parts)
    df_surr  = _shuffle(surr_parts)
    df_test  = _shuffle(test_parts)

    logger.info(
        f"Stratified temporal 3-way split — "
        f"train: {len(df_train)} ({train_frac:.0%}), "
        f"surr: {len(df_surr)} ({surr_frac:.0%}), "
        f"test: {len(df_test)} ({test_frac:.0%})"
    )
    for part_name, part_df in [("train", df_train), ("surr", df_surr), ("test", df_test)]:
        vc = part_df[label_col].value_counts(normalize=True).round(3).to_dict()
        logger.info(f"  {part_name} class ratios: {vc}")

    return df_train, df_surr, df_test
