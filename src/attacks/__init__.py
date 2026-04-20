"""Attack implementations."""
from .targeted_evasion import TargetedEvasionAttack
from .random_evasion import RandomEvasionAttack

__all__ = [
    "TargetedEvasionAttack",
    "RandomEvasionAttack",
]
