"""
Heterogeneous Graph Utilities Module.

Shared utility functions for heterogeneous graph creation
and processing. 
#! Refact, temporary module
"""

import logging

logger = logging.getLogger(__name__)


def log_memory(tag: str):
    """
    Log current memory usage for profiling and debugging.

   Issue of OOM on the CSE-CIC-IDS2018 dataset. Used to track memory consumption.

    Args:
        tag: Description of current processing stage for log identification.
    """
    import psutil
    
    # Get current process memory info
    mem = psutil.Process().memory_info().rss / (1024**3)
    
    # Log memory usage at DEBUG level
    logger.debug(f"[MEM] {tag}: {mem:.2f} GB")
