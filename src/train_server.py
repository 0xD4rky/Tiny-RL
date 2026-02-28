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
import functools
import logging
import os
from pathlib import Path
import time
from typing import Iterator

import torch
import torch.distributed as dist
import uvicorn
from fastapi import FastAPI, Request
from safetensors.torch import save_file
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from transformers import AutoConfig

from loss import approx_kl_divergence, build_loss
from replay_buffer import Experience, ReplayBuffer, join_experience_batch
from utils import load_config, load_model, sequences_log_probs, init_rng

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
    # Use transformer_auto_wrap_policy based on the model's _no_split_modules.
    # This wraps at decoder-layer boundaries and avoids FSDP sharding lm_head
    # and embed_tokens into separate units (which causes size mismatches).
    layer_classes = set()
    if hasattr(model, "_no_split_modules"):
        for cls_name in model._no_split_modules:
            for module in model.modules():
                if type(module).__name__ == cls_name:
                    layer_classes.add(type(module))
                    break

    wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=layer_classes,
    )

    return FSDP(
        model,
        auto_wrap_policy=wrap_policy,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device(),
        use_orig_params=True,
    )


# ── TrainServer ──────────────────────────────────────────────────────

class TrainServer:

    def __init__(self, cfg: dict):
        self.cfg = cfg
        mcfg, tcfg = cfg["model"], cfg["training"]
        scfg = cfg.get("server", {})
        wcfg = scfg.get("weight_sync", {})
        self.rank = dist.get_rank()
        self.device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))

        # policy model (trainable, FSDP)
        model, self.tokenizer = load_model(
            mcfg["name"], trust_remote_code=mcfg["trust_remote_code"],
            bf16=mcfg["bf16"], device_map=None,
        )
        model.to(self.device)
        self.num_params = sum(p.numel() for p in model.parameters())
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        self.model = _wrap_fsdp(model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(tcfg["lr"]))
        self.objective = build_loss(cfg["loss"])

        # reference model (frozen, FSDP)
        ref, _ = load_model(
            mcfg["name"], trust_remote_code=mcfg["trust_remote_code"],
            bf16=mcfg["bf16"], device_map=None,
        )
        ref.to(self.device)
        self.ref_model = _wrap_fsdp(ref)
        self.ref_model.eval()

        self.pad_id = self.tokenizer.eos_token_id
        self.replay = ReplayBuffer()
        self._sync_dir_ready = False
        self._weight_sync_mode = "disk"
        self._weight_sync_reason = "not_initialized"
        self._weight_sync_group = None
        self._weight_sync_backend = wcfg.get("backend", "nccl")
        self._weight_sync_packed = bool(wcfg.get("packed", True))
        self._max_nccl_params = int(wcfg.get("max_nccl_params", 70_000_000_000))
        log.info("Rank %d ready on %s", self.rank, self.device)

    def _iter_model_named_parameters(self) -> Iterator[tuple[str, torch.Tensor]]:
        module = self.model.module if hasattr(self.model, "module") else self.model
        return module.named_parameters()

    def _plan_weight_sync_mode(self) -> tuple[str, str]:
        if self._weight_sync_backend != "nccl":
            return "disk", f"backend={self._weight_sync_backend}"
        if self.num_params > self._max_nccl_params:
            return (
                "disk",
                f"num_params={self.num_params} exceeds max_nccl_params={self._max_nccl_params}",
            )
        try:
            from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferEngine
        except Exception as exc:
            return "disk", f"nccl_weight_transfer_unavailable: {exc}"
        if NCCLWeightTransferEngine is None:
            return "disk", "nccl_weight_transfer_unavailable"
        return "nccl", "ok"

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

    def save_weights(self) -> dict:
        """Gather full state-dict on rank 0 and write to shared directory."""
        policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, policy):
            sd = self.model.state_dict()

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

        dist.barrier()
        return {"status": "ok"}

    def plan_weight_sync(self) -> dict:
        mode, reason = self._plan_weight_sync_mode()
        self._weight_sync_mode = mode
        self._weight_sync_reason = reason
        return {
            "status": "ok",
            "mode": mode,
            "reason": reason,
            "num_params": self.num_params,
            "max_nccl_params": self._max_nccl_params,
        }

    def init_weight_sync(self, master_address: str, master_port: int, world_size: int) -> dict:
        mode, reason = self._plan_weight_sync_mode()
        if self.rank == 0 and mode == "nccl":
            from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferEngine

            self._weight_sync_group = NCCLWeightTransferEngine.trainer_init(
                dict(
                    master_address=master_address,
                    master_port=int(master_port),
                    world_size=int(world_size),
                )
            )
        elif self.rank == 0:
            self._weight_sync_group = None

        payload = [mode, reason]
        dist.broadcast_object_list(payload, src=0)
        self._weight_sync_mode, self._weight_sync_reason = payload
        log.info(
            "Weight sync mode on rank %d: %s (%s)",
            self.rank,
            self._weight_sync_mode,
            self._weight_sync_reason,
        )
        return {"status": "ok", "mode": self._weight_sync_mode, "reason": self._weight_sync_reason}

    def prepare_weight_sync(self) -> dict:
        if self._weight_sync_mode != "nccl":
            return {"status": "ok", "mode": "disk", "reason": self._weight_sync_reason}

        names: list[str] = []
        dtype_names: list[str] = []
        shapes: list[list[int]] = []
        with FSDP.summon_full_params(self.model, recurse=True, writeback=False, rank0_only=True):
            if self.rank == 0:
                for name, param in self._iter_model_named_parameters():
                    names.append(name)
                    dtype_names.append(str(param.dtype).replace("torch.", ""))
                    shapes.append(list(param.shape))

        dist.barrier()
        if self.rank == 0:
            return {
                "status": "ok",
                "mode": "nccl",
                "names": names,
                "dtype_names": dtype_names,
                "shapes": shapes,
                "packed": self._weight_sync_packed,
            }
        return {"status": "ok", "mode": "nccl"}

    def broadcast_weights(self, packed: bool | None = None) -> dict:
        if self._weight_sync_mode != "nccl":
            return {"status": "ok", "mode": "disk", "reason": self._weight_sync_reason}

        if self._weight_sync_group is None:
            raise RuntimeError("weight sync group is not initialized")

        packed = self._weight_sync_packed if packed is None else bool(packed)
        with FSDP.summon_full_params(self.model, recurse=True, writeback=False, rank0_only=True):
            if self.rank == 0:
                from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferEngine

                NCCLWeightTransferEngine.trainer_send_weights(
                    iterator=self._iter_model_named_parameters(),
                    group=self._weight_sync_group,
                    packed=packed,
                )

        dist.barrier()
        return {"status": "ok", "mode": "nccl", "packed": packed}


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

    port = args.port + int(os.environ["LOCAL_RANK"])
    log.info("Train server rank %d on port %d", dist.get_rank(), port)
    uvicorn.run(app, host="0.0.0.0", port=port)
