"""
FSDP training server — one FastAPI instance per GPU rank.

Hosts the policy model (FSDP, trainable) and reference model (FSDP, frozen).
The controller sends rollout data via HTTP; all ranks must receive their
requests at the same time so FSDP collectives stay in sync.

Usage:
    torchrun --nproc_per_node=N train_server.py --config config.yaml --port 5000

    Each rank serves on port = base_port + local_rank.
"""

import argparse
import logging
import os
from pathlib import Path
import time
from contextlib import contextmanager
from typing import Iterator

import torch
import torch.distributed as dist
import uvicorn
from fastapi import FastAPI, Request
from safetensors.torch import save_file
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from transformers import AutoConfig

from loss import approx_kl_divergence, build_loss
from replay_buffer import Experience, ReplayBuffer, join_experience_batch
from utils import load_config, load_model, sequences_log_probs, init_rng
from weight_sync.train_backends import build_trainer_backend

log = logging.getLogger(__name__)
app = FastAPI()


# ── GPU peak TFLOPS (BF16 tensor core) ───────────────────────────────

def _gpu_peak_tflops(device: torch.device) -> float | None:
    name = torch.cuda.get_device_properties(device).name.lower()
    if "h200"  in name: return 1979.0
    if "h100"  in name: return 989.0
    if "a100"  in name: return 312.0
    if "a6000" in name: return 154.0
    if "4090"  in name: return 165.0
    return None


# ── FSDP helper ──────────────────────────────────────────────────────

def _wrap_fsdp(model):
    # Use the model's _no_split_modules to find decoder-layer classes and
    # apply FSDP2 wrapping at those boundaries before wrapping the root module.
    layer_classes = set()
    if hasattr(model, "_no_split_modules"):
        for cls_name in model._no_split_modules:
            for module in model.modules():
                if type(module).__name__ == cls_name:
                    layer_classes.add(type(module))
                    break

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        output_dtype=torch.bfloat16,
    )
    for module in model.modules():
        if type(module) in layer_classes:
            fully_shard(module, mp_policy=mp_policy)
    fully_shard(model, mp_policy=mp_policy)
    return model, "fsdp2"


# ── TrainServer ──────────────────────────────────────────────────────

class TrainServer:

    def __init__(self, cfg: dict):
        self.cfg = cfg
        mcfg, tcfg = cfg["model"], cfg["training"]
        scfg = cfg.get("server", {})
        wcfg = scfg.get("weight_sync", {})
        self.rank = dist.get_rank()
        self.device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        self.weight_sync_cfg = wcfg
        max_nccl_params = wcfg.get("max_nccl_params", 70_000_000_000)
        try:
            self.max_nccl_params = int(max_nccl_params)
        except (TypeError, ValueError):
            self.max_nccl_params = int(float(max_nccl_params))
        self.weight_sync_packed = bool(wcfg.get("packed", True))

        # policy model (trainable, FSDP2)
        model, self.tokenizer = load_model(
            mcfg["name"], trust_remote_code=mcfg["trust_remote_code"],
            bf16=mcfg["bf16"], device_map=None,
        )
        model.to(self.device)
        self.num_params = sum(p.numel() for p in model.parameters())
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        self.model, self.fsdp_impl = _wrap_fsdp(model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(tcfg["lr"]))
        self.objective = build_loss(cfg["loss"])

        # reference model (frozen, FSDP2)
        ref, _ = load_model(
            mcfg["name"], trust_remote_code=mcfg["trust_remote_code"],
            bf16=mcfg["bf16"], device_map=None,
        )
        ref.to(self.device)
        self.ref_model, _ = _wrap_fsdp(ref)
        self.ref_model.eval()

        self.pad_id = self.tokenizer.eos_token_id
        self.replay = ReplayBuffer()
        self._sync_dir_ready = False
        self.weight_sync_mode = "disk"
        self.weight_sync_reason = "not_initialized"
        self.requested_weight_sync_backend = wcfg.get("backend", "nccl").lower()
        self.backends = {
            "disk": build_trainer_backend(self, "disk"),
            "nccl": build_trainer_backend(self, "nccl"),
            "rdma": build_trainer_backend(self, "rdma"),
        }
        self.active_weight_sync_backend = self.backends.get(
            self.requested_weight_sync_backend, self.backends["disk"]
        )
        log.info(
            "Rank %d ready on %s (wrap=%s, weight_sync=%s)",
            self.rank,
            self.device,
            self.fsdp_impl,
            self.requested_weight_sync_backend,
        )

    def iter_model_named_parameters(self) -> Iterator[tuple[str, torch.Tensor]]:
        module = self.model.module if hasattr(self.model, "module") else self.model
        return module.named_parameters()

    @contextmanager
    def summon_full_params(self, rank0_only: bool = True):
        """FSDP2: temporarily swap DTensor params with full (all-gathered) tensors."""
        from torch.distributed._tensor import DTensor
        swapped: list[tuple[torch.nn.Parameter, torch.Tensor]] = []
        try:
            for param in self.model.parameters():
                if isinstance(param.data, DTensor):
                    swapped.append((param, param.data))
                    param.data = param.data.full_tensor()
            yield
        finally:
            for param, orig in swapped:
                param.data = orig

    def dist_barrier(self):
        dist.barrier()

    def broadcast_mode(self, mode: str, reason: str):
        payload = [mode, reason]
        dist.broadcast_object_list(payload, src=0)
        self.weight_sync_mode, self.weight_sync_reason = payload

    # ── /create_online_dataset ───────────────────────────────────────

    def create_online_dataset(self, rollout_batches: list[dict]):
        """Buffer rollout data for training.

        action_log_probs come directly from vLLM (computed at generation time
        under the same policy that produced the tokens), so we only need one
        forward pass here — the reference model for KL.
        """
        self.replay.clear()
        with torch.no_grad():
            for d in rollout_batches:
                seq  = torch.tensor(d["sequences"],        dtype=torch.long,  device=self.device)
                attn = torch.tensor(d["attention_mask"],   dtype=torch.bool,  device=self.device)
                act  = torch.tensor(d["action_mask"],      dtype=torch.bool,  device=self.device)
                lp   = torch.tensor(d["action_log_probs"], dtype=torch.float, device=self.device)
                ref_lp = sequences_log_probs(self.ref_model, seq, attn)
                kl = approx_kl_divergence(lp, ref_lp, act)
                self.replay.append(Experience(
                    sequences=seq, action_log_probs=lp, ref_log_probs=ref_lp,
                    returns=torch.tensor(d["returns"],    dtype=torch.float, device=self.device),
                    advantages=torch.tensor(d["advantages"], dtype=torch.float, device=self.device),
                    action_mask=act, attention_mask=attn, kl=kl,
                ).to(torch.device("cpu")))
        torch.cuda.empty_cache()
        return {"status": "ok", "buffer_size": len(self.replay)}

    # ── /train_1_iter ────────────────────────────────────────────────

    def train_1_iter(self) -> dict:
        tcfg = self.cfg["training"]
        if len(self.replay) < tcfg["train_batch_size"]:
            return {"status": "skipped"}

        loader = DataLoader(
            self.replay, batch_size=tcfg["train_batch_size"],
            shuffle=False, drop_last=True, collate_fn=join_experience_batch,
        )
        grad_acc = tcfg.get("grad_acc_steps", 1)
        metrics: dict = {}
        total_tokens = 0

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        for epoch in range(tcfg["epochs_per_step"]):
            self.model.train()
            self.optimizer.zero_grad()
            for micro, exp in enumerate(loader):
                exp = exp.to(self.device)
                total_tokens += int(exp.attention_mask.sum())
                lp = sequences_log_probs(self.model, exp.sequences, exp.attention_mask)
                loss, kl = self.objective(log_probs=lp, experience=exp)
                (loss / grad_acc).backward()
                if (micro + 1) % grad_acc == 0:
                    gn = clip_grad_norm_(self.model.parameters(), tcfg["max_norm"])
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    metrics = {"loss": loss.item(), "kl": kl.item(), "grad_norm": gn.item()}

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        # throughput
        tokens_per_sec = total_tokens / elapsed

        # MFU: 6*N*T flops (2 fwd + 4 bwd) across all train GPUs
        world_size = dist.get_world_size()
        peak_tflops = _gpu_peak_tflops(self.device)
        if peak_tflops and elapsed > 0:
            actual_tflops = 6 * self.num_params * total_tokens / elapsed / 1e12
            mfu = actual_tflops / (world_size * peak_tflops)
        else:
            mfu = None

        metrics.update({"tokens_per_sec": tokens_per_sec, "mfu": mfu})
        return {"status": "ok", "metrics": metrics}

    # ── /save_weights ────────────────────────────────────────────────

    def save_weights_to_disk(self) -> dict:
        """Gather full parameters on rank0 and write them to shared directory."""
        sd: dict[str, torch.Tensor] = {}
        with self.summon_full_params(rank0_only=True):
            if self.rank == 0:
                for name, param in self.iter_model_named_parameters():
                    sd[name] = param.data.cpu().clone()

        if self.rank == 0:
            sync_dir = Path(self.cfg.get("server", {}).get("weights_dir", ".vllm_weights"))
            sync_dir.mkdir(exist_ok=True)
            if not self._sync_dir_ready:
                mcfg = self.cfg["model"]
                AutoConfig.from_pretrained(
                    mcfg["name"], trust_remote_code=mcfg["trust_remote_code"],
                ).save_pretrained(sync_dir)
                self.tokenizer.save_pretrained(sync_dir)
                self._sync_dir_ready = True
            save_file(sd, str(sync_dir / "model.safetensors"))
            log.info("Weights saved to %s", sync_dir)

        self.dist_barrier()
        return {"status": "ok", "mode": "disk"}

    def save_weights(self) -> dict:
        return self.save_weights_to_disk()

    def plan_weight_sync(self) -> dict:
        requested_backend = self.backends.get(
            self.requested_weight_sync_backend, self.backends["disk"]
        )
        decision = requested_backend.plan()
        self.active_weight_sync_backend = self.backends.get(decision.mode, self.backends["disk"])
        self.weight_sync_mode = decision.mode
        self.weight_sync_reason = decision.reason
        payload = decision.asdict()
        payload["status"] = "ok"
        return payload

    def init_weight_sync(self, master_address: str, master_port: int, world_size: int) -> dict:
        result = self.active_weight_sync_backend.init(
            master_address=master_address,
            master_port=master_port,
            world_size=world_size,
        )
        if "mode" in result:
            self.weight_sync_mode = result["mode"]
        if "reason" in result:
            self.weight_sync_reason = result["reason"]
        return result

    def prepare_weight_sync(self) -> dict:
        return self.active_weight_sync_backend.prepare()

    def broadcast_weights(self, packed: bool | None = None) -> dict:
        return self.active_weight_sync_backend.transfer(packed=packed)

    def transfer_weights(self, ops: list[dict], packed: bool | None = None) -> dict:
        return self.active_weight_sync_backend.transfer(ops=ops, packed=packed)


# ── FastAPI endpoints ────────────────────────────────────────────────

server: TrainServer | None = None


@app.post("/create_online_dataset")
async def ep_create_dataset(request: Request):
    data = await request.json()
    return server.create_online_dataset(data["rollout_batches"])


@app.post("/train_1_iter")
async def ep_train():
    return server.train_1_iter()


@app.post("/save_weights")
async def ep_save_weights():
    return server.save_weights()


@app.post("/plan_weight_sync")
async def ep_plan_weight_sync():
    return server.plan_weight_sync()


@app.post("/init_weight_sync")
async def ep_init_weight_sync(request: Request):
    data = await request.json()
    return server.init_weight_sync(
        master_address=data["master_address"],
        master_port=int(data["master_port"]),
        world_size=int(data["world_size"]),
    )


@app.post("/prepare_weight_sync")
async def ep_prepare_weight_sync():
    return server.prepare_weight_sync()


@app.post("/broadcast_weights")
async def ep_broadcast_weights(request: Request):
    try:
        data = await request.json()
    except Exception:
        data = {}
    return server.broadcast_weights(packed=data.get("packed"))


@app.post("/transfer_weights")
async def ep_transfer_weights(request: Request):
    data = await request.json()
    return server.transfer_weights(
        ops=data.get("ops", []),
        packed=data.get("packed"),
    )


@app.get("/health")
async def ep_health():
    return {"status": "ok", "rank": server.rank if server else -1}


# ── main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    cfg = load_config(args.config)
    init_rng(cfg["training"]["seed"])
    server = TrainServer(cfg)

    # Rank 0: start wandb for system GPU monitoring (grouped with controller run)
    if dist.get_rank() == 0:
        import wandb
        run_name = cfg.get("wandb", {}).get("run_name", "run")
        wandb.init(
            project=cfg.get("wandb", {}).get("project", "RL"),
            name=f"{run_name}-train",
            group=run_name,
            job_type="train",
            config=cfg,
        )

    port = args.port + int(os.environ["LOCAL_RANK"])
    log.info("Train server rank %d on port %d", dist.get_rank(), port)
    uvicorn.run(app, host="0.0.0.0", port=port)
