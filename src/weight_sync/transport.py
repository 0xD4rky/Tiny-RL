from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class TransferOp:
    src_param: str
    dst_param: str
    src_off: int
    dst_off: int
    nbytes: int
    pack: bool = False
    src_ptr: int = 0
    dst_ptr: int = 0
    dst_rkey: int = 0


class WeightTransferTransport(ABC):
    """Transport interface for inter-process weight transfer.

    Real RDMA backends should implement CUDA MR registration and async writes.
    """

    @abstractmethod
    def init_endpoint(self, **kwargs) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def register_memory_regions(self, regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def transfer(self, ops: list[TransferOp]) -> dict[str, Any]:
        raise NotImplementedError


class MockWeightTransferTransport(WeightTransferTransport):
    """Mock transport used to validate routing and correctness logic."""

    def __init__(self):
        self._initialized = False

    def init_endpoint(self, **kwargs) -> dict[str, Any]:
        self._initialized = True
        return {"status": "ok", "transport": "mock", "init": kwargs}

    def register_memory_regions(self, regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "ptr": int(r["ptr"]),
                "size": int(r["size"]),
                "rkey": idx + 1,
            }
            for idx, r in enumerate(regions)
            if int(r.get("size", 0)) > 0
        ]

    def transfer(self, ops: list[TransferOp]) -> dict[str, Any]:
        total_bytes = sum(op.nbytes for op in ops)
        return {
            "status": "ok",
            "transport": "mock",
            "num_ops": len(ops),
            "total_bytes": total_bytes,
        }


class NullRdmaTransport(WeightTransferTransport):
    """Placeholder for future ibverbs/libfabric backend."""

    def init_endpoint(self, **kwargs) -> dict[str, Any]:
        return {"status": "error", "reason": "rdma_transport_not_implemented", "init": kwargs}

    def register_memory_regions(self, regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return []

    def transfer(self, ops: list[TransferOp]) -> dict[str, Any]:
        return {
            "status": "error",
            "reason": "rdma_transport_not_implemented",
            "num_ops": len(ops),
        }


class AutoRdmaTransport(WeightTransferTransport):
    """Try ibverbs first, then libfabric, and use first successful backend."""

    def __init__(self):
        self._impl: WeightTransferTransport | None = None
        self._name: str = "rdma-auto"

    def init_endpoint(self, **kwargs) -> dict[str, Any]:
        if self._impl is not None:
            return self._impl.init_endpoint(**kwargs)

        errors: list[str] = []
        candidates: list[tuple[str, WeightTransferTransport]] = []
        try:
            from .rdma.ibverbs_layer import IbverbsTransport

            candidates.append(("ibverbs", IbverbsTransport()))
        except Exception as exc:
            errors.append(f"ibverbs import failed: {exc}")
        try:
            from .rdma.libfabric_layer import LibfabricTransport

            candidates.append(("libfabric", LibfabricTransport()))
        except Exception as exc:
            errors.append(f"libfabric import failed: {exc}")

        for name, impl in candidates:
            res = impl.init_endpoint(**kwargs)
            if res.get("status") == "ok":
                self._impl = impl
                self._name = name
                res.setdefault("transport", name)
                return res
            errors.append(f"{name}: {res.get('reason', 'init_failed')}")

        return {
            "status": "error",
            "transport": "rdma-auto",
            "reason": "; ".join(errors) if errors else "no_rdma_candidates",
        }

    def register_memory_regions(self, regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self._impl is None:
            return []
        return self._impl.register_memory_regions(regions)

    def transfer(self, ops: list[TransferOp]) -> dict[str, Any]:
        if self._impl is None:
            return {"status": "error", "transport": "rdma-auto", "reason": "transport_not_initialized"}
        return self._impl.transfer(ops)


def build_transport(name: str) -> WeightTransferTransport:
    name = (name or "mock").lower()
    if name == "mock":
        return MockWeightTransferTransport()
    if name == "ibverbs":
        from .rdma.ibverbs_layer import IbverbsTransport

        return IbverbsTransport()
    if name == "libfabric":
        from .rdma.libfabric_layer import LibfabricTransport

        return LibfabricTransport()
    if name in {"auto", "rdma-auto"}:
        return AutoRdmaTransport()
    if name == "rdma":
        # Backward-compatible alias for auto-selecting a real RDMA provider.
        return AutoRdmaTransport()
    raise ValueError(f"unknown transport backend: {name}")
