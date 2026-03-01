"""
Single entry point for the full RL training pipeline.

Multi-node aware: uses NODE_RANK to assign roles.
  - Node 0: FSDP training (all local GPUs)
  - Node 1: vLLM inference + asyncio controller

Single-node: launches both on the same machine with a GPU split.

Usage:
    python launch.py --config config.yaml
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import yaml

log = logging.getLogger(__name__)

def _resolve_config(cfg: dict, master_addr: str) -> dict:
    """Replace __MASTER_ADDR__ placeholders throughout the config."""
    raw = json.dumps(cfg)
    raw = raw.replace("__MASTER_ADDR__", master_addr)
    return json.loads(raw)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config_path = str(Path(args.config).resolve())
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    node_rank = int(os.environ.get("NODE_RANK", 0))
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    # MosaicML doesn't set NNODES; read from config, then env, then default.
    num_nodes = int(
        cfg.get("server", {}).get("weight_sync", {}).get(
            "num_nodes", os.environ.get("NNODES", 1)
        )
    )

    # Resolve placeholders (e.g. __MASTER_ADDR__ → actual hostname)
    cfg = _resolve_config(cfg, master_addr)

    gpu_split = cfg.get("gpu_split", {})
    train_gpus = gpu_split.get("train", [0])
    inference_gpus = gpu_split.get("inference", [1])

    scfg = cfg.get("server", {})
    train_port = scfg.get("train_port", 5000)
    rollout_port = scfg.get("rollout_port", 7000)

    src_dir = Path(__file__).resolve().parent

    # Ensure controller sees correct number of train ranks
    cfg.setdefault("fsdp", {})["num_gpus"] = len(train_gpus)

    # Write resolved config for subprocesses
    resolved_config = Path("/tmp/resolved_config.yaml")
    with open(resolved_config, "w") as f:
        yaml.safe_dump(cfg, f)
    resolved_path = str(resolved_config)

    if num_nodes >= 2:
        # ── Multi-node: dedicated roles per node ───────────────────────
        if node_rank == 0:
            # Node 0: training only
            log.info("Node 0: launching FSDP training on GPUs %s", train_gpus)
            train_env = os.environ.copy()
            train_env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in train_gpus)
            cmd = [
                sys.executable, "-m", "torch.distributed.run",
                f"--nproc_per_node={len(train_gpus)}",
                str(src_dir / "train_server.py"),
                "--config", resolved_path,
                "--port", str(train_port),
            ]
            proc = subprocess.Popen(cmd, env=train_env)
            try:
                proc.wait()
            except KeyboardInterrupt:
                proc.terminate()
                proc.wait(timeout=15)

        elif node_rank == 1:
            # Node 1: inference + controller
            log.info("Node 1: launching rollout on GPUs %s + controller", inference_gpus)

            rollout_env = os.environ.copy()
            rollout_env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in inference_gpus)
            rollout_cmd = [
                sys.executable,
                str(src_dir / "rollout_server.py"),
                "--config", resolved_path,
                "--port", str(rollout_port),
            ]
            rollout_proc = subprocess.Popen(rollout_cmd, env=rollout_env)

            # Controller runs CPU-only
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            try:
                from controller import run as run_controller
                asyncio.run(run_controller(cfg))
            except KeyboardInterrupt:
                log.info("Interrupted.")
            finally:
                if rollout_proc.poll() is None:
                    log.info("Terminating rollout server ...")
                    rollout_proc.terminate()
                    try:
                        rollout_proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        rollout_proc.kill()
                log.info("Done.")
        else:
            log.info("Node %d: no role assigned, idling.", node_rank)

    else:
        # ── Single-node: split GPUs between training and inference ─────
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

        train_env = os.environ.copy()
        train_env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in train_gpus)
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            f"--nproc_per_node={len(train_gpus)}",
            str(src_dir / "train_server.py"),
            "--config", resolved_path,
            "--port", str(train_port),
        ]

        rollout_env = os.environ.copy()
        rollout_env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in inference_gpus)
        rollout_cmd = [
            sys.executable,
            str(src_dir / "rollout_server.py"),
            "--config", resolved_path,
            "--port", str(rollout_port),
        ]

        log.info("Launching train servers on GPUs %s ...", train_gpus)
        train_proc = subprocess.Popen(cmd, env=train_env)
        log.info("Launching rollout server on GPUs %s ...", inference_gpus)
        rollout_proc = subprocess.Popen(rollout_cmd, env=rollout_env)

        try:
            from controller import run as run_controller
            asyncio.run(run_controller(cfg))
        except KeyboardInterrupt:
            log.info("Interrupted.")
        finally:
            procs = [(rollout_proc, "rollout server"), (train_proc, "train servers")]
            for proc, label in procs:
                if proc.poll() is None:
                    log.info("Terminating %s ...", label)
                    proc.terminate()
                    try:
                        proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        proc.kill()
            log.info("Done.")


if __name__ == "__main__":
    main()
