"""Weight-sync backends and transport abstractions."""

from .base import WeightSyncDecision
from .train_backends import build_trainer_backend
from .transport import build_transport

__all__ = [
    "WeightSyncDecision",
    "build_trainer_backend",
    "build_transport",
]
