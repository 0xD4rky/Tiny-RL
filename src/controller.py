"""
Asyncio controller — coordinates rollout generation and distributed training.

Talks to train servers and rollout server via HTTP. Follows the
compose-rl controller pattern:

    asyncio loop:
        1. generate rollouts  (vLLM rollout server)
        2. send data to train servers  (/create_online_dataset)
        3. trigger training  (/train_1_iter)
        4. sync weights  (NCCL in-memory, with disk fallback)

Usage (preferred — single command):
    python launch.py --config config.yaml
"""

import argparse
import asyncio
import logging
from pathlib import Path
import time

import wandb
from torch.utils.data import DataLoader

from utils import (
    load_config,
    read_prompts,
    init_rng,
)
from weight_sync.qwen_router import build_qwen_routing_table
from rollout_engine import RolloutEngine
from train_engine import TrainEngine

log = logging.getLogger(__name__)


async def run(cfg: dict):
    tcfg = cfg["training"]
    rcfg = cfg["rollout"]
    dcfg = cfg["dataset"]
    scfg = cfg.get("server", {})

    init_rng(tcfg["seed"])

    # ── train engine (HTTP client to FSDP servers) ───────────────────
    train_engine = TrainEngine(
        base_url=scfg.get("train_url", "http://localhost"),
        base_port=scfg.get("train_port", 5000),
        num_ranks=cfg.get("fsdp", {}).get("num_gpus", 1),
    )
    rollout_engine = RolloutEngine(
        base_url=scfg.get("rollout_url", "http://localhost"),
        port=scfg.get("rollout_port", 7000),
    )
    log.info("Waiting for train and rollout servers ...")
    await train_engine.wait_for_ready()
    await rollout_engine.wait_for_ready()

    # ── data ─────────────────────────────────────────────────────────
    prompts = read_prompts(dcfg["name"], max_rows=dcfg.get("max_rows"))
    loader = DataLoader(prompts, batch_size=rcfg["rollouts_per_step"],
                        shuffle=True, drop_last=True)

    run_name = cfg["wandb"]["run_name"]
    wandb.init(
        project=cfg["wandb"]["project"],
        name=f"{run_name}-rollout",
        group=run_name,
        job_type="rollout",
        config=cfg,
    )
    checkpoint_dir = Path(tcfg["checkpoint_path"]) / run_name
    sync_interval = rcfg.get("sync_interval", 2)
    weights_dir = scfg.get("weights_dir", ".vllm_weights")
    wsync_cfg = scfg.get("weight_sync", {})
    n_inference_gpus = len(cfg.get("gpu_split", {}).get("inference", [0]))
    sync_mode = "disk"
    sync_reason = "not_initialized"
    update_info: dict | None = None
    rdma_routes: list[dict] = []
    weight_sync_packed = bool(wsync_cfg.get("packed", True))

    # ── initialize weight sync path ───────────────────────────────────
    plan_results = await train_engine.plan_weight_sync()
    modes = {p.get("mode") for p in plan_results}
    reasons = {p.get("reason") for p in plan_results}
    if len(modes) != 1 or len(reasons) != 1:
        raise RuntimeError(f"inconsistent train-rank weight-sync plan: {plan_results}")
    plan = plan_results[0]
    sync_mode = plan.get("mode", "disk")
    sync_reason = plan.get("reason", "unknown")
    if sync_mode == "nccl":
        master_address = wsync_cfg.get("master_address", "127.0.0.1")
        master_port = int(wsync_cfg.get("master_port", scfg.get("train_port", 5000) + 1000))
        world_size = n_inference_gpus + 1  # trainer rank0 + all inference TP ranks
        init_info = {
            "master_address": master_address,
            "master_port": master_port,
            "world_size": world_size,
            "rank_offset": 1,
        }
        try:
            rollout_init, train_init = await asyncio.gather(
                rollout_engine.init_weight_sync("nccl", init_info),
                train_engine.init_weight_sync(master_address, master_port, world_size),
            )
            if rollout_init.get("status") != "ok":
                raise RuntimeError(f"rollout nccl init failed: {rollout_init}")
            if train_init[0].get("mode") != "nccl":
                sync_mode = "disk"
                sync_reason = train_init[0].get("reason", "trainer_init_returned_disk")
            else:
                prep_results = await train_engine.prepare_weight_sync()
                prep = prep_results[0]
                if prep.get("mode") != "nccl":
                    sync_mode = "disk"
                    sync_reason = prep.get("reason", "prepare_weight_sync_returned_disk")
                else:
                    weight_sync_packed = bool(prep.get("packed", weight_sync_packed))
                    update_info = {
                        "names": prep["names"],
                        "dtype_names": prep["dtype_names"],
                        "shapes": prep["shapes"],
                        "packed": weight_sync_packed,
                    }
        except Exception as exc:
            sync_mode = "disk"
            sync_reason = f"nccl_init_failed: {exc}"
            log.warning("Weight sync fallback to disk: %s", sync_reason)
    elif sync_mode == "rdma":
        try:
            # Initialize transport endpoints and precompute routes once.
            init_payload = {
                "master_address": wsync_cfg.get("master_address", "127.0.0.1"),
                "master_port": int(wsync_cfg.get("master_port", 6000)),
            }
            rollout_init, train_init = await asyncio.gather(
                rollout_engine.init_weight_sync("rdma", init_payload),
                train_engine.init_weight_sync(
                    init_payload["master_address"],
                    init_payload["master_port"],
                    0,
                ),
            )
            if rollout_init.get("status") != "ok":
                raise RuntimeError(f"rollout rdma init failed: {rollout_init}")
            if train_init[0].get("mode") != "rdma":
                sync_mode = train_init[0].get("mode", "disk")
                sync_reason = train_init[0].get("reason", "rdma_not_selected")
            else:
                rollout_meta, rollout_regions, train_prep = await asyncio.gather(
                    rollout_engine.get_param_metadata(),
                    rollout_engine.get_memory_regions(),
                    train_engine.prepare_weight_sync(),
                )
                if train_prep[0].get("mode") != "rdma":
                    sync_mode = train_prep[0].get("mode", "disk")
                    sync_reason = train_prep[0].get("reason", "rdma_prepare_failed")
                else:
                    mrs = []
                    for worker in rollout_regions.get("workers", []):
                        reg = await rollout_engine.register_mrs(worker.get("regions", []))
                        mrs.extend(reg.get("mrs", []))
                    rdma_routes = build_qwen_routing_table(
                        trainer_routes=train_prep[0].get("routes", []),
                        rollout_meta=rollout_meta,
                        rollout_mrs=mrs,
                    )
                    log.info(
                        "Prepared RDMA routes: %d entries, %d remote MRs",
                        len(rdma_routes),
                        len(mrs),
                    )
        except Exception as exc:
            sync_mode = "disk"
            sync_reason = f"rdma_init_failed: {exc}"
            log.warning("Weight sync fallback to disk: %s", sync_reason)

    log.info("Weight sync mode: %s (%s)", sync_mode, sync_reason)

    log.info("%d prompts, %d steps, group_size=%d",
             len(prompts), len(loader), rcfg["group_size"])

    # ── main loop ────────────────────────────────────────────────────
    for k, batch in enumerate(loader):
        t_step = time.perf_counter()
        questions = list(batch["problem"])
        answers = list(batch["answer"])

        # 1) generate rollouts from rollout server
        t_generate = time.perf_counter()
        rollout_resp = await rollout_engine.generate(questions, answers)
        generate_time = time.perf_counter() - t_generate
        rollout_batches = rollout_resp["rollout_batches"]
        rollout_tps = float(rollout_resp["rollout_tokens_per_sec"])
        reward_mean = float(rollout_resp["reward_mean"])
        accuracy = float(rollout_resp["accuracy"])
        log.info(
            "Step %d: reward=%.3f acc=%.1f%% | rollout %.0f tok/s (generate %.2fs)",
            k,
            reward_mean,
            accuracy * 100,
            rollout_tps,
            generate_time,
        )

        # 2) send to train servers  (all ranks receive the same data)
        t_push = time.perf_counter()
        await train_engine.create_online_dataset(rollout_batches)
        push_time = time.perf_counter() - t_push

        # 3) train
        t_train = time.perf_counter()
        train_results = await train_engine.train_1_iter()
        train_time = time.perf_counter() - t_train
        metrics = train_results[0].get("metrics", {})

        if metrics:
            wandb.log({
                "rollout/reward_mean": reward_mean,
                "rollout/accuracy": accuracy,
                "rollout/tokens_per_sec": rollout_tps,
                "timing/generate_time_s": generate_time,
                "timing/dataset_push_time_s": push_time,
                "timing/train_rpc_time_s": train_time,
                "train/loss": metrics["loss"],
                "train/kl": metrics["kl"],
                "train/grad_norm": metrics["grad_norm"],
                "train/tokens_per_sec": metrics.get("tokens_per_sec"),
                "train/mfu": metrics.get("mfu"),
            }, step=k)
        else:
            wandb.log(
                {
                    "timing/generate_time_s": generate_time,
                    "timing/dataset_push_time_s": push_time,
                    "timing/train_rpc_time_s": train_time,
                },
                step=k,
            )

        # 4) sync weights
        if (k + 1) % sync_interval == 0:
            t_sync = time.perf_counter()
            if sync_mode == "nccl" and update_info is not None:
                try:
                    rollout_sync, train_sync = await asyncio.gather(
                        rollout_engine.update_weights_nccl(update_info),
                        train_engine.broadcast_weights(packed=weight_sync_packed),
                    )
                    if rollout_sync.get("status") != "ok" or train_sync[0].get("status") != "ok":
                        raise RuntimeError(f"nccl sync failed: rollout={rollout_sync} train={train_sync}")
                    sync_time = time.perf_counter() - t_sync
                    log.info("Rollout server updated from trainer via NCCL (%.1fs)", sync_time)
                except Exception as exc:
                    sync_mode = "disk"
                    sync_reason = f"nccl_sync_failed: {exc}"
                    log.warning("Switching to disk weight sync: %s", sync_reason)
                    await train_engine.save_weights()
                    await rollout_engine.reload_from_disk(weights_dir)
                    sync_time = time.perf_counter() - t_sync
                    log.info("Rollout server reloaded from %s (%.1fs)", weights_dir, sync_time)
            elif sync_mode == "rdma":
                try:
                    # Controller-ACK based boundary: both transfer sides must ack this step.
                    rollout_ack, train_ack = await asyncio.gather(
                        rollout_engine.apply_rdma_routes(rdma_routes, step=k),
                        train_engine.transfer_weights(ops=rdma_routes, packed=weight_sync_packed),
                    )
                    if rollout_ack.get("status") != "ok" or train_ack[0].get("status") != "ok":
                        raise RuntimeError(f"rdma sync failed: rollout={rollout_ack} train={train_ack}")
                    if train_ack[0].get("fallback") == "disk":
                        await rollout_engine.reload_from_disk(weights_dir)
                    sync_time = time.perf_counter() - t_sync
                    log.info("RDMA sync acked at step %d (%.1fs)", k, sync_time)
                except Exception as exc:
                    sync_mode = "disk"
                    sync_reason = f"rdma_sync_failed: {exc}"
                    log.warning("Switching to disk weight sync: %s", sync_reason)
                    await train_engine.save_weights()
                    await rollout_engine.reload_from_disk(weights_dir)
                    sync_time = time.perf_counter() - t_sync
                    log.info("Rollout server reloaded from %s (%.1fs)", weights_dir, sync_time)
            else:
                await train_engine.save_weights()
                await rollout_engine.reload_from_disk(weights_dir)
                sync_time = time.perf_counter() - t_sync
                log.info("Rollout server reloaded from %s (%.1fs)", weights_dir, sync_time)
        else:
            sync_time = 0.0

        step_total = time.perf_counter() - t_step
        log.info(
            "Step %d timing: total %.2fs | generate %.2fs | push %.2fs | train_rpc %.2fs | sync %.2fs",
            k,
            step_total,
            generate_time,
            push_time,
            train_time,
            sync_time,
        )
        wandb.log(
            {
                "timing/step_total_s": step_total,
                "timing/generate_time_s": generate_time,
                "timing/dataset_push_time_s": push_time,
                "timing/train_rpc_time_s": train_time,
                "timing/sync_time_s": sync_time,
            },
            step=k,
        )

        # 5) checkpoint
        if (k + 1) % tcfg["checkpoint_interval"] == 0:
            save_path = checkpoint_dir / f"step_{k + 1}"
            save_path.mkdir(parents=True, exist_ok=True)
            log.info("Checkpoint: %s (weights in train server)", save_path)

    wandb.finish()
    log.info("Training complete.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    asyncio.run(run(cfg))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    main()
