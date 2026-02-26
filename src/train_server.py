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
        self.rank = dist.get_rank()
        self.device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))

        # policy model (trainable, FSDP)
        model, self.tokenizer = load_model(
            mcfg["name"], trust_remote_code=mcfg["trust_remote_code"],
            bf16=mcfg["bf16"], device_map=None,
        )
        model.to(self.device)
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
        log.info("Rank %d ready on %s", self.rank, self.device)

    # ── /create_online_dataset ───────────────────────────────────────

    def create_online_dataset(self, rollout_batches: list[dict]):
        """Compute policy + ref log-probs and buffer for training."""
        self.replay.clear()
        self.model.eval()
        with torch.no_grad():
            for d in rollout_batches:
                seq = torch.tensor(d["sequences"], dtype=torch.long, device=self.device)
                attn = torch.tensor(d["attention_mask"], dtype=torch.bool, device=self.device)
                act = torch.tensor(d["action_mask"], dtype=torch.bool, device=self.device)
                lp = sequences_log_probs(self.model, seq, attn)
                ref_lp = sequences_log_probs(self.ref_model, seq, attn)
                kl = approx_kl_divergence(lp, ref_lp, act)
                self.replay.append(Experience(
                    sequences=seq, action_log_probs=lp, ref_log_probs=ref_lp,
                    returns=torch.tensor(d["returns"], dtype=torch.float, device=self.device),
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

        for epoch in range(tcfg["epochs_per_step"]):
            self.model.train()
            self.optimizer.zero_grad()
            for micro, exp in enumerate(loader):
                exp = exp.to(self.device)
                lp = sequences_log_probs(self.model, exp.sequences, exp.attention_mask)
                loss, kl = self.objective(log_probs=lp, experience=exp)
                (loss / grad_acc).backward()
                if (micro + 1) % grad_acc == 0:
                    gn = clip_grad_norm_(self.model.parameters(), tcfg["max_norm"])
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    metrics = {"loss": loss.item(), "kl": kl.item(), "grad_norm": gn.item()}
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
