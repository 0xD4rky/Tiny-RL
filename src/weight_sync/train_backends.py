from __future__ import annotations

import os
from typing import Any

from .base import WeightSyncDecision
from .transport import TransferOp, build_transport


class DiskTrainerBackend:
    name = "disk"

    def __init__(self, server):
        self.server = server

    def plan(self) -> WeightSyncDecision:
        return WeightSyncDecision(
            mode="disk",
            reason="disk_backend_selected",
            backend=self.name,
            num_params=self.server.num_params,
            max_nccl_params=self.server.max_nccl_params,
        )

    def init(self, **kwargs) -> dict[str, Any]:
        return {"status": "ok", "mode": "disk", "reason": "disk_backend_selected"}

    def prepare(self) -> dict[str, Any]:
        return {"status": "ok", "mode": "disk", "reason": "disk_backend_selected"}

    def transfer(self, **kwargs) -> dict[str, Any]:
        return self.server.save_weights_to_disk()


class NcclTrainerBackend:
    name = "nccl"

    def __init__(self, server):
        self.server = server
        self._group = None

    def _is_available(self) -> tuple[bool, str]:
        try:
            from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferEngine  # noqa: F401
        except Exception as exc:
            return False, f"nccl_weight_transfer_unavailable: {exc}"
        return True, "ok"

    def plan(self) -> WeightSyncDecision:
        if self.server.num_params > self.server.max_nccl_params:
            return WeightSyncDecision(
                mode="disk",
                reason=(
                    f"num_params={self.server.num_params} exceeds "
                    f"max_nccl_params={self.server.max_nccl_params}"
                ),
                backend=self.name,
                num_params=self.server.num_params,
                max_nccl_params=self.server.max_nccl_params,
            )
        ok, reason = self._is_available()
        return WeightSyncDecision(
            mode="nccl" if ok else "disk",
            reason=reason,
            backend=self.name,
            num_params=self.server.num_params,
            max_nccl_params=self.server.max_nccl_params,
        )

    def init(self, **kwargs) -> dict[str, Any]:
        decision = self.plan()
        if decision.mode != "nccl":
            return {"status": "ok", "mode": "disk", "reason": decision.reason}

        master_address = kwargs["master_address"]
        master_port = int(kwargs["master_port"])
        world_size = int(kwargs["world_size"])

        if self.server.rank == 0:
            from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferEngine

            self._group = NCCLWeightTransferEngine.trainer_init(
                {
                    "master_address": master_address,
                    "master_port": master_port,
                    "world_size": world_size,
                }
            )

        self.server.broadcast_mode("nccl", "ok")
        return {"status": "ok", "mode": self.server.weight_sync_mode, "reason": self.server.weight_sync_reason}

    def prepare(self) -> dict[str, Any]:
        if self.server.weight_sync_mode != "nccl":
            return {"status": "ok", "mode": "disk", "reason": self.server.weight_sync_reason}

        names: list[str] = []
        dtype_names: list[str] = []
        shapes: list[list[int]] = []
        with self.server.summon_full_params(rank0_only=True) as full_params:
            if self.server.rank == 0:
                for name, t in full_params.items():
                    names.append(name)
                    dtype_names.append(str(t.dtype).replace("torch.", ""))
                    shapes.append(list(t.shape))

        self.server.dist_barrier()
        if self.server.rank == 0:
            return {
                "status": "ok",
                "mode": "nccl",
                "names": names,
                "dtype_names": dtype_names,
                "shapes": shapes,
                "packed": self.server.weight_sync_packed,
            }
        return {"status": "ok", "mode": "nccl"}

    def transfer(self, **kwargs) -> dict[str, Any]:
        if self.server.weight_sync_mode != "nccl":
            return {"status": "ok", "mode": "disk", "reason": self.server.weight_sync_reason}
        if self._group is None:
            raise RuntimeError("NCCL trainer transfer group is not initialized")

        packed = bool(kwargs.get("packed", self.server.weight_sync_packed))
        with self.server.summon_full_params(rank0_only=True) as full_params:
            if self.server.rank == 0:
                from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferEngine

                NCCLWeightTransferEngine.trainer_send_weights(
                    iterator=full_params.items(),
                    group=self._group,
                    packed=packed,
                )

        self.server.dist_barrier()
        return {"status": "ok", "mode": "nccl", "packed": packed}


class RdmaTrainerBackend:
    name = "rdma"

    def __init__(self, server):
        self.server = server
        tname = server.weight_sync_cfg.get("transport", "mock")
        self.transport = build_transport(tname)
        self.routing_table: list[TransferOp] = []

    def _prefer_nccl_single_node(self) -> bool:
        # For single 8xH100 (or any single-node setup), keep NCCL baseline.
        num_nodes = int(self.server.weight_sync_cfg.get("num_nodes", os.environ.get("NNODES", 1)))
        return num_nodes <= 1

    def plan(self) -> WeightSyncDecision:
        if self._prefer_nccl_single_node():
            return WeightSyncDecision(
                mode="nccl",
                reason="single_node_prefers_nccl",
                backend=self.name,
                num_params=self.server.num_params,
                max_nccl_params=self.server.max_nccl_params,
            )
        return WeightSyncDecision(
            mode="rdma",
            reason="rdma_backend_selected",
            backend=self.name,
            num_params=self.server.num_params,
            max_nccl_params=self.server.max_nccl_params,
        )

    def init(self, **kwargs) -> dict[str, Any]:
        decision = self.plan()
        if decision.mode != "rdma":
            return {"status": "ok", "mode": decision.mode, "reason": decision.reason}

        # Only rank 0 sets up the RDMA transport endpoint (TCP exchange +
        # ibverbs QP).  Other ranks just wait for the broadcast_mode.
        if self.server.rank == 0:
            init_kwargs = dict(self.server.weight_sync_cfg.get("transport_init", {}))
            init_kwargs.update(kwargs)
            init_kwargs.setdefault("role", "server")
            master_address = str(kwargs.get("master_address", init_kwargs.get("master_address", "127.0.0.1")))
            master_port = int(kwargs.get("master_port", init_kwargs.get("master_port", 6000)))
            if init_kwargs["role"] == "server":
                init_kwargs.setdefault("listen_host", "0.0.0.0")
                init_kwargs.setdefault("listen_port", master_port)
            else:
                init_kwargs.setdefault("peer_host", master_address)
                init_kwargs.setdefault("peer_port", master_port)
            init_kwargs.setdefault("tcp_timeout_s", 300)
            res = self.transport.init_endpoint(**init_kwargs)
            if res.get("status") != "ok":
                self.server.broadcast_mode("disk", f"rdma_init_failed: {res.get('reason', 'unknown')}")
                return {
                    "status": "ok",
                    "mode": self.server.weight_sync_mode,
                    "reason": self.server.weight_sync_reason,
                    "transport": res,
                }
            self.server.broadcast_mode("rdma", "ok")
        else:
            self.server.broadcast_mode("rdma", "ok")

        return {"status": "ok", "mode": self.server.weight_sync_mode}

    def prepare(self) -> dict[str, Any]:
        # Keep preparation cheap for now: metadata-only route setup.
        if self.server.weight_sync_mode != "rdma":
            return {"status": "ok", "mode": self.server.weight_sync_mode, "reason": self.server.weight_sync_reason}

        routes: list[dict[str, Any]] = []
        with self.server.summon_full_params(rank0_only=True) as full_params:
            if self.server.rank == 0:
                for name, t in full_params.items():
                    routes.append(
                        {
                            "src_param": name,
                            "dst_param": name,
                            "src_off": 0,
                            "dst_off": 0,
                            "shape": list(t.shape),
                            "dtype": str(t.dtype).replace("torch.", ""),
                            "nbytes": int(t.numel() * t.element_size()),
                            "pack": False,
                        }
                    )
        self.server.dist_barrier()
        if self.server.rank == 0:
            return {"status": "ok", "mode": "rdma", "routes": routes}
        return {"status": "ok", "mode": "rdma"}

    def transfer(self, **kwargs) -> dict[str, Any]:
        if self.server.weight_sync_mode != "rdma":
            return {"status": "ok", "mode": self.server.weight_sync_mode, "reason": self.server.weight_sync_reason}

        ops = kwargs.get("ops", [])
        res: dict[str, Any] = {"status": "ok", "mode": "rdma", "rank": self.server.rank}
        # All ranks enter summon_full_params (collective all-gather).
        # Rank 0 builds transfer ops and executes the RDMA transfer while
        # the full tensors are still alive in GPU memory.
        with self.server.summon_full_params(rank0_only=True) as full_params:
            if self.server.rank == 0:
                param_ptrs: dict[str, int] = {}
                for name, t in full_params.items():
                    param_ptrs[name] = int(t.data_ptr())

                parsed_ops: list[TransferOp] = []
                for op in ops:
                    if isinstance(op, TransferOp):
                        p = op
                    else:
                        src_param = str(op.get("src_param", ""))
                        src_base = int(param_ptrs.get(src_param, 0))
                        src_off = int(op.get("src_off", 0))
                        p = TransferOp(
                            src_param=src_param,
                            dst_param=str(op.get("dst_param", src_param)),
                            src_off=src_off,
                            dst_off=int(op.get("dst_off", 0)),
                            nbytes=int(op.get("nbytes", 0)),
                            pack=bool(op.get("pack", False)),
                            src_ptr=int(op.get("src_ptr", src_base + src_off if src_base else 0)),
                            dst_ptr=int(op.get("dst_ptr", 0)),
                            dst_rkey=int(op.get("dst_rkey", 0)),
                        )
                    if p.nbytes > 0 and p.src_ptr > 0 and p.dst_ptr > 0 and p.dst_rkey > 0:
                        parsed_ops.append(p)

                res = self.transport.transfer(parsed_ops)
        self.server.dist_barrier()
        if res.get("status") != "ok":
            disk_res = self.server.save_weights_to_disk()
            return {
                "status": "ok",
                "mode": "rdma",
                "transport": res,
                "fallback": "disk",
                "disk": disk_res,
            }
        return {"status": "ok", "mode": "rdma", "transport": res}


def build_trainer_backend(server, backend_name: str):
    backend_name = (backend_name or "disk").lower()
    if backend_name == "disk":
        return DiskTrainerBackend(server)
    if backend_name == "nccl":
        return NcclTrainerBackend(server)
    if backend_name == "rdma":
        return RdmaTrainerBackend(server)
    raise ValueError(f"unknown trainer weight sync backend: {backend_name}")
