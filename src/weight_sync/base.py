from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class WeightSyncDecision:
    mode: str
    reason: str
    backend: str
    num_params: int
    max_nccl_params: int

    def asdict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "reason": self.reason,
            "backend": self.backend,
            "num_params": self.num_params,
            "max_nccl_params": self.max_nccl_params,
        }


class TrainerWeightSyncBackend(Protocol):
    name: str

    def plan(self) -> WeightSyncDecision:
        ...

    def init(self, **kwargs) -> dict[str, Any]:
        ...

    def prepare(self) -> dict[str, Any]:
        ...

    def transfer(self, **kwargs) -> dict[str, Any]:
        ...
