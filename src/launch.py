"""
Single entry point for the full RL training pipeline.

gpu_split uses global GPU indices across the entire cluster:

  gpu_split:
    train: [0,1,2,3,4,5,6,7]               # node 0 trains
    inference: [8,9,10,11,12,13,14,15]      # node 1 infers

  gpu_split:
    train: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]         # nodes 0-1 train
    inference: [16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]  # nodes 2-3 infer

Each node maps global indices to local GPUs and launches the
appropriate processes.

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

SRC_DIR = Path(__file__).resolve().parent


def _resolve_config(cfg: dict, master_addr: str) -> dict:
    raw = json.dumps(cfg)
    raw = raw.replace("__MASTER_ADDR__", master_addr)
    return json.loads(raw)


def _write_resolved_config(cfg: dict) -> str:
    path = Path("/tmp/resolved_config.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)
    return str(path)


def _global_to_local(global_gpus: list[int], node_rank: int, gpus_per_node: int) -> list[int]:
    """Return local GPU indices (0-based) that belong to this node."""
    lo = node_rank * gpus_per_node
    hi = lo + gpus_per_node
    return sorted(g - lo for g in global_gpus if lo <= g < hi)


def _make_env(gpus: list[int]) -> dict:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
    return env


def _launch_train(cfg_path: str, gpus: list[int], port: int) -> subprocess.Popen:
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        f"--nproc_per_node={len(gpus)}",
        str(SRC_DIR / "train_server.py"),
        "--config", cfg_path,
        "--port", str(port),
    ]
    log.info("Launching FSDP training on local GPUs %s", gpus)
    return subprocess.Popen(cmd, env=_make_env(gpus))


def _launch_rollout(cfg_path: str, gpus: list[int], port: int) -> subprocess.Popen:
    cmd = [
        sys.executable,
        str(SRC_DIR / "rollout_server.py"),
        "--config", cfg_path,
        "--port", str(port),
    ]
    log.info("Launching rollout server on local GPUs %s", gpus)
    return subprocess.Popen(cmd, env=_make_env(gpus))


def _terminate(procs: list[tuple[subprocess.Popen, str]]):
    for proc, label in procs:
        if proc.poll() is None:
            log.info("Terminating %s ...", label)
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(Path(args.config).resolve()) as f:
        cfg = yaml.safe_load(f)

    node_rank = int(os.environ.get("NODE_RANK", 0))
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    gpus_per_node = int(os.environ.get("LOCAL_WORLD_SIZE", 8))

    cfg = _resolve_config(cfg, master_addr)

    gpu_split = cfg.get("gpu_split", {})
    train_gpus = _global_to_local(gpu_split.get("train", []), node_rank, gpus_per_node)
    inference_gpus = _global_to_local(gpu_split.get("inference", []), node_rank, gpus_per_node)

    if not train_gpus and not inference_gpus:
        log.info("Node %d: no GPUs assigned, exiting.", node_rank)
        return

    scfg = cfg.get("server", {})
    train_port = scfg.get("train_port", 5000)
    rollout_port = scfg.get("rollout_port", 7000)
    cfg.setdefault("fsdp", {})["num_gpus"] = len(gpu_split.get("train", []))
    cfg_path = _write_resolved_config(cfg)

    log.info("Node %d: local train=%s local inference=%s", node_rank, train_gpus, inference_gpus)

    procs: list[tuple[subprocess.Popen, str]] = []
    try:
        if train_gpus:
            procs.append((_launch_train(cfg_path, train_gpus, train_port), "train servers"))

        if inference_gpus:
            procs.append((_launch_rollout(cfg_path, inference_gpus, rollout_port), "rollout server"))
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            from controller import run as run_controller
            asyncio.run(run_controller(cfg))
        else:
            procs[0][0].wait()
    except KeyboardInterrupt:
        log.info("Interrupted.")
    finally:
        _terminate(procs)
        log.info("Done.")


if __name__ == "__main__":
    main()
