"""Rollout server that owns vLLM GPUs and exposes rollout + weight-sync endpoints."""

from __future__ import annotations

import argparse
import logging
import time
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, Request

from utils import (
    load_config,
    load_model,
    create_vllm_engine,
    destroy_vllm_engine,
    vllm_rollout,
    group_advantages,
    init_rng,
)
from weight_sync.transport import build_transport

log = logging.getLogger(__name__)
app = FastAPI()


def _resolve_engine_model(engine):
    candidates = [
        ["llm_engine", "model_executor", "driver_worker", "model_runner", "model"],
        ["llm_engine", "model_executor", "driver_worker", "worker", "model_runner", "model"],
        ["engine", "model"],
    ]
    for path in candidates:
        cur = engine
        ok = True
        for part in path:
            if not hasattr(cur, part):
                ok = False
                break
            cur = getattr(cur, part)
        if ok and hasattr(cur, "named_parameters"):
            return cur
    return None


def _collect_param_metadata_local(engine) -> list[dict[str, Any]]:
    model = _resolve_engine_model(engine)
    if model is None:
        return []
    out: list[dict[str, Any]] = []
    for name, p in model.named_parameters():
        if not isinstance(p, torch.Tensor):
            continue
        if not p.is_cuda:
            continue
        out.append(
            {
                "name": name,
                "ptr": int(p.data_ptr()),
                "nbytes": int(p.numel() * p.element_size()),
                "dtype": str(p.dtype).replace("torch.", ""),
                "shape": list(p.shape),
            }
        )
    return out


def _collect_memory_regions_local() -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    if not torch.cuda.is_available():
        return regions
    try:
        snapshot = torch.cuda.memory_snapshot()
    except Exception:
        return regions
    for seg in snapshot:
        size = int(seg.get("total_size", 0))
        ptr = int(seg.get("address", 0))
        if ptr and size > 0:
            regions.append({"ptr": ptr, "size": size})
    return regions


class RolloutServer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.mcfg = cfg["model"]
        self.rcfg = cfg["rollout"]
        self.scfg = cfg.get("server", {})
        self.wcfg = self.scfg.get("weight_sync", {})
        self.num_infer_gpus = len(cfg.get("gpu_split", {}).get("inference", [0]))
        self.transport = build_transport(self.wcfg.get("transport", "mock"))
        self.last_synced_step = -1

        _, self.tokenizer = load_model(
            self.mcfg["name"],
            trust_remote_code=self.mcfg["trust_remote_code"],
            bf16=self.mcfg["bf16"],
            device_map="cpu",
        )
        self.engine = create_vllm_engine(
            self.mcfg["name"],
            self.rcfg["gpu_memory_utilization"],
            self.rcfg["max_length"],
            tensor_parallel_size=self.num_infer_gpus,
        )
        self.pad_id = self.tokenizer.eos_token_id

    def generate(self, questions: list[str], answers: list[str]) -> dict[str, Any]:
        t0 = time.perf_counter()
        results = vllm_rollout(
            self.engine,
            self.tokenizer,
            questions,
            answers,
            group_size=self.rcfg["group_size"],
            max_length=self.rcfg["max_length"],
            temperature=self.rcfg["temperature"],
            top_p=self.rcfg["top_p"],
        )
        rollout_time = max(time.perf_counter() - t0, 1e-6)

        rollout_batches: list[dict[str, Any]] = []
        all_rewards: list[float] = []
        total_generated_tokens = 0
        for seq_ids, returns, act_mask, vllm_lp, _completions in results:
            attn_mask = seq_ids != self.pad_id
            adv = group_advantages(returns)
            all_rewards.extend(returns.squeeze(-1).tolist())
            total_generated_tokens += int(act_mask.sum())
            rollout_batches.append(
                {
                    "sequences": seq_ids.tolist(),
                    "returns": returns.tolist(),
                    "advantages": adv.tolist(),
                    "action_mask": act_mask.tolist(),
                    "attention_mask": attn_mask.tolist(),
                    "action_log_probs": vllm_lp.tolist(),
                }
            )

        reward_mean = sum(all_rewards) / max(len(all_rewards), 1)
        accuracy = sum(1 for r in all_rewards if r >= 0.8) / max(len(all_rewards), 1)
        return {
            "status": "ok",
            "rollout_batches": rollout_batches,
            "reward_mean": reward_mean,
            "accuracy": accuracy,
            "rollout_tokens_per_sec": total_generated_tokens / rollout_time,
        }

    def init_weight_sync(self, mode: str, init_info: dict[str, Any]) -> dict[str, Any]:
        if mode == "nccl":
            llm_engine = getattr(self.engine, "llm_engine", self.engine)
            try:
                llm_engine.init_weight_transfer_engine({"init_info": init_info})
            except TypeError:
                llm_engine.init_weight_transfer_engine(init_info=init_info)
            return {"status": "ok", "mode": "nccl"}

        if mode == "rdma":
            merged = dict(self.wcfg.get("transport_init", {}))
            merged.update(init_info)
            merged.setdefault("role", "client")
            master_address = str(init_info.get("master_address", merged.get("master_address", "0.0.0.0")))
            master_port = int(init_info.get("master_port", merged.get("master_port", 6000)))
            if merged["role"] == "client":
                merged.setdefault("peer_host", master_address)
                merged.setdefault("peer_port", master_port)
            else:
                merged.setdefault("listen_host", "0.0.0.0")
                merged.setdefault("listen_port", master_port)
            return self.transport.init_endpoint(**merged)

        return {"status": "ok", "mode": "disk"}

    def update_weights_nccl(self, update_info: dict[str, Any]) -> dict[str, Any]:
        llm_engine = getattr(self.engine, "llm_engine", self.engine)
        try:
            llm_engine.update_weights({"update_info": update_info})
        except TypeError:
            llm_engine.update_weights(update_info=update_info)
        return {"status": "ok", "mode": "nccl"}

    def reload_from_disk(self, model_path: str) -> dict[str, Any]:
        destroy_vllm_engine(self.engine)
        self.engine = create_vllm_engine(
            model_path,
            self.rcfg["gpu_memory_utilization"],
            self.rcfg["max_length"],
            tensor_parallel_size=self.num_infer_gpus,
        )
        return {"status": "ok", "mode": "disk", "model_path": model_path}

    def get_param_metadata(self) -> dict[str, Any]:
        data = [{"rank": 0, "params": _collect_param_metadata_local(self.engine)}]
        return {"status": "ok", "workers": data}

    def get_memory_regions(self) -> dict[str, Any]:
        data = [{"rank": 0, "regions": _collect_memory_regions_local()}]
        return {"status": "ok", "workers": data}

    def register_mrs(self, regions: list[dict[str, Any]]) -> dict[str, Any]:
        mrs = self.transport.register_memory_regions(regions)
        return {"status": "ok", "mrs": mrs}

    def apply_rdma_routes(self, routes: list[dict[str, Any]], step: int) -> dict[str, Any]:
        # Phase-2 mock path: validate route format and acknowledge sync version.
        total_bytes = sum(int(r.get("nbytes", 0)) for r in routes)
        self.last_synced_step = max(self.last_synced_step, int(step))
        return {
            "status": "ok",
            "mode": "rdma",
            "num_routes": len(routes),
            "total_bytes": total_bytes,
            "ack_step": self.last_synced_step,
        }


server: RolloutServer | None = None


@app.get("/health")
async def ep_health():
    return {"status": "ok", "synced_step": server.last_synced_step if server else -1}


@app.post("/generate")
async def ep_generate(request: Request):
    data = await request.json()
    return server.generate(data["questions"], data["answers"])


@app.post("/init_weight_sync")
async def ep_init_weight_sync(request: Request):
    data = await request.json()
    return server.init_weight_sync(mode=data["mode"], init_info=data.get("init_info", {}))


@app.post("/update_weights_nccl")
async def ep_update_weights_nccl(request: Request):
    data = await request.json()
    return server.update_weights_nccl(update_info=data["update_info"])


@app.post("/reload_from_disk")
async def ep_reload_from_disk(request: Request):
    data = await request.json()
    return server.reload_from_disk(model_path=data["model_path"])


@app.post("/get_param_metadata")
async def ep_get_param_metadata():
    return server.get_param_metadata()


@app.post("/get_memory_regions")
async def ep_get_memory_regions():
    return server.get_memory_regions()


@app.post("/register_mrs")
async def ep_register_mrs(request: Request):
    data = await request.json()
    return server.register_mrs(regions=data.get("regions", []))


@app.post("/apply_rdma_routes")
async def ep_apply_rdma_routes(request: Request):
    data = await request.json()
    return server.apply_rdma_routes(routes=data.get("routes", []), step=int(data.get("step", -1)))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--port", type=int, default=7000)
    args = parser.parse_args()

    cfg = load_config(args.config)
    init_rng(cfg["training"]["seed"])
    server = RolloutServer(cfg)

    log.info("Rollout server listening on port %d", args.port)
    uvicorn.run(app, host="0.0.0.0", port=args.port)
