"""
Asyncio controller — coordinates rollout generation and distributed training.

Talks to FSDP train servers via HTTP through the TrainEngine.  Follows the
compose-rl controller pattern:

    asyncio loop:
        1. generate rollouts  (vLLM, local)
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
    load_model,
    create_vllm_engine,
    destroy_vllm_engine,
    vllm_rollout,
    group_advantages,
    read_prompts,
    init_rng,
)
from train_engine import TrainEngine

log = logging.getLogger(__name__)


async def run(cfg: dict):
    tcfg = cfg["training"]
    rcfg = cfg["rollout"]
    mcfg = cfg["model"]
    dcfg = cfg["dataset"]
    scfg = cfg.get("server", {})

    init_rng(tcfg["seed"])

    # ── train engine (HTTP client to FSDP servers) ───────────────────
    train_engine = TrainEngine(
        base_url=scfg.get("train_url", "http://localhost"),
        base_port=scfg.get("train_port", 5000),
        num_ranks=cfg.get("fsdp", {}).get("num_gpus", 1),
    )
    log.info("Waiting for train servers ...")
    await train_engine.wait_for_ready()

    # ── vLLM (local, for rollout generation) ─────────────────────────
    n_inference_gpus = len(cfg.get("gpu_split", {}).get("inference", [0]))
    log.info("Creating vLLM engine on %d GPU(s) ...", n_inference_gpus)
    _, tokenizer = load_model(mcfg["name"], trust_remote_code=mcfg["trust_remote_code"],
                              bf16=mcfg["bf16"], device_map="cpu")
    engine = create_vllm_engine(
        mcfg["name"], rcfg["gpu_memory_utilization"], rcfg["max_length"],
        tensor_parallel_size=n_inference_gpus,
    )
    pad_id = tokenizer.eos_token_id

    # ── data ─────────────────────────────────────────────────────────
    prompts = read_prompts(dcfg["name"], max_rows=dcfg.get("max_rows"))
    loader = DataLoader(prompts, batch_size=rcfg["rollouts_per_step"],
                        shuffle=True, drop_last=True)

    run_name = cfg["wandb"]["run_name"]
    wandb.init(project=cfg["wandb"]["project"], name=run_name, config=cfg)
    checkpoint_dir = Path(tcfg["checkpoint_path"]) / run_name
    sync_interval = rcfg.get("sync_interval", 2)
    weights_dir = scfg.get("weights_dir", ".vllm_weights")
    wsync_cfg = scfg.get("weight_sync", {})
    sync_mode = "disk"
    sync_reason = "not_initialized"
    weight_update_info: dict | None = None
    weight_sync_packed = bool(wsync_cfg.get("packed", True))

    async def _reload_engine_from_disk():
        nonlocal engine
        await train_engine.save_weights()
        destroy_vllm_engine(engine)
        engine = create_vllm_engine(
            weights_dir,
            rcfg["gpu_memory_utilization"],
            rcfg["max_length"],
            tensor_parallel_size=n_inference_gpus,
        )

    # ── initialize weight sync path ───────────────────────────────────
    plan_results = await train_engine.plan_weight_sync()
    plan = plan_results[0]
    sync_mode = plan.get("mode", "disk")
    sync_reason = plan.get("reason", "unknown")
    if sync_mode == "nccl":
        if not hasattr(engine, "init_weight_transfer_engine") or not hasattr(engine, "update_weights"):
            sync_mode = "disk"
            sync_reason = "vllm_api_missing_weight_transfer"
            log.warning("Weight sync fallback to disk: %s", sync_reason)
        else:
            master_address = wsync_cfg.get("master_address", "127.0.0.1")
            master_port = int(wsync_cfg.get("master_port", scfg.get("train_port", 5000) + 1000))
            world_size = n_inference_gpus + 1  # trainer rank0 + all inference TP ranks
            init_info = {
                "master_address": master_address,
                "master_port": master_port,
                "world_size": world_size,
                "rank_offset": 1,
            }

            def _init_vllm_weight_transfer():
                try:
                    engine.init_weight_transfer_engine({"init_info": init_info})
                except TypeError:
                    engine.init_weight_transfer_engine(init_info=init_info)

            try:
                _, init_results = await asyncio.gather(
                    asyncio.to_thread(_init_vllm_weight_transfer),
                    train_engine.init_weight_sync(master_address, master_port, world_size),
                )
                init_mode = init_results[0].get("mode", "disk")
                if init_mode != "nccl":
                    sync_mode = "disk"
                    sync_reason = init_results[0].get("reason", "trainer_init_returned_disk")
                else:
                    prep_results = await train_engine.prepare_weight_sync()
                    prep = prep_results[0]
                    if prep.get("mode") != "nccl":
                        sync_mode = "disk"
                        sync_reason = prep.get("reason", "prepare_weight_sync_returned_disk")
                    else:
                        weight_sync_packed = bool(prep.get("packed", weight_sync_packed))
                        weight_update_info = {
                            "update_info": {
                                "names": prep["names"],
                                "dtype_names": prep["dtype_names"],
                                "shapes": prep["shapes"],
                                "packed": weight_sync_packed,
                            }
                        }

                        def _update_vllm_weights():
                            try:
                                engine.update_weights(weight_update_info)
                            except TypeError:
                                engine.update_weights(update_info=weight_update_info["update_info"])
            except Exception as exc:
                sync_mode = "disk"
                sync_reason = f"nccl_init_failed: {exc}"
                log.warning("Weight sync fallback to disk: %s", sync_reason)

    log.info("Weight sync mode: %s (%s)", sync_mode, sync_reason)

    log.info("%d prompts, %d steps, group_size=%d",
             len(prompts), len(loader), rcfg["group_size"])

    # ── main loop ────────────────────────────────────────────────────
    for k, batch in enumerate(loader):
        questions = list(batch["problem"])
        answers = list(batch["answer"])

        # 1) generate rollouts with vLLM
        t_rollout = time.perf_counter()
        results = vllm_rollout(
            engine, tokenizer, questions, answers,
            group_size=rcfg["group_size"], max_length=rcfg["max_length"],
            temperature=rcfg["temperature"], top_p=rcfg["top_p"],
        )
        rollout_time = time.perf_counter() - t_rollout

        # package as JSON-serialisable dicts
        rollout_batches: list[dict] = []
        all_rewards: list[float] = []
        total_generated_tokens = 0
        for seq_ids, returns, act_mask, vllm_lp, _completions in results:
            attn_mask = seq_ids != pad_id
            adv = group_advantages(returns)
            all_rewards.extend(returns.squeeze(-1).tolist())
            total_generated_tokens += int(act_mask.sum())
            rollout_batches.append({
                "sequences": seq_ids.tolist(),
                "returns": returns.tolist(),
                "advantages": adv.tolist(),
                "action_mask": act_mask.tolist(),
                "attention_mask": attn_mask.tolist(),
                "action_log_probs": vllm_lp.tolist(),
            })

        rollout_tps = total_generated_tokens / rollout_time
        reward_mean = sum(all_rewards) / len(all_rewards)
        accuracy = sum(1 for r in all_rewards if r >= 0.8) / len(all_rewards)
        log.info("Step %d: reward=%.3f acc=%.1f%% | rollout %.0f tok/s",
                 k, reward_mean, accuracy * 100, rollout_tps)

        # 2) send to train servers  (all ranks receive the same data)
        await train_engine.create_online_dataset(rollout_batches)

        # 3) train
        train_results = await train_engine.train_1_iter()
        metrics = train_results[0].get("metrics", {})

        if metrics:
            wandb.log({
                "rollout/reward_mean": reward_mean,
                "rollout/accuracy": accuracy,
                "rollout/tokens_per_sec": rollout_tps,
                "train/loss": metrics["loss"],
                "train/kl": metrics["kl"],
                "train/grad_norm": metrics["grad_norm"],
                "train/tokens_per_sec": metrics.get("tokens_per_sec"),
                "train/mfu": metrics.get("mfu"),
            }, step=k)

        # 4) sync weights
        if (k + 1) % sync_interval == 0:
            t_sync = time.perf_counter()
            if sync_mode == "nccl" and weight_update_info is not None:
                try:
                    await asyncio.gather(
                        asyncio.to_thread(_update_vllm_weights),
                        train_engine.broadcast_weights(packed=weight_sync_packed),
                    )
                    sync_time = time.perf_counter() - t_sync
                    log.info("vLLM updated from trainer via NCCL (%.1fs)", sync_time)
                except Exception as exc:
                    sync_mode = "disk"
                    sync_reason = f"nccl_sync_failed: {exc}"
                    log.warning("Switching to disk weight sync: %s", sync_reason)
                    await _reload_engine_from_disk()
                    sync_time = time.perf_counter() - t_sync
                    log.info("vLLM reloaded from %s (%.1fs)", weights_dir, sync_time)
            else:
                await _reload_engine_from_disk()
                sync_time = time.perf_counter() - t_sync
                log.info("vLLM reloaded from %s (%.1fs)", weights_dir, sync_time)
            wandb.log({"sync/weight_sync_time": sync_time}, step=k)

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
