"""Device selection utilities."""

from typing import Tuple


def get_device(prefer_cuda: bool = True) -> Tuple[str, bool]:
    """Return device string and availability.

    If CUDA is available and prefer_cuda is True, returns ("cuda", True),
    otherwise returns ("cpu", False).
    """
    try:
        import torch

        if prefer_cuda and torch.cuda.is_available():
            return "cuda", True
    except Exception:
        pass

    return "cpu", False
