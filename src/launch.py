"""
Single entry point for the full RL training pipeline.

Splits GPUs between training (FSDP) and inference (vLLM), launches
train servers as a subprocess, then runs the asyncio controller.

Usage:
    python launch.py --config config.yaml
"""

import argparse
import asyncio
import logging
import os
import subprocess
import sys
from pathlib import Path

import yaml

log = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config_path = str(Path(args.config).resolve())
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    gpu_split = cfg.get("gpu_split", {})
    train_gpus = gpu_split.get("train", [0])
    inference_gpus = gpu_split.get("inference", [1])

    scfg = cfg.get("server", {})
    port = scfg.get("train_port", 5000)

    # Set inference GPUs for this process (controller + vLLM)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in inference_gpus)

    # Ensure controller sees correct number of train ranks
    cfg.setdefault("fsdp", {})["num_gpus"] = len(train_gpus)

    # ── launch FSDP train servers via torchrun ─────────────────────────
    src_dir = Path(__file__).resolve().parent
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        f"--nproc_per_node={len(train_gpus)}",
        str(src_dir / "train_server.py"),
        "--config", config_path,
        "--port", str(port),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in train_gpus)

    log.info("Launching train servers on GPUs %s ...", train_gpus)
    train_proc = subprocess.Popen(cmd, env=env)

    try:
        # Import controller *after* CUDA_VISIBLE_DEVICES is set so vLLM
        # only sees the inference GPUs.
        from controller import run as run_controller

        asyncio.run(run_controller(cfg))
    except KeyboardInterrupt:
        log.info("Interrupted.")
    finally:
        if train_proc.poll() is None:
            log.info("Terminating train servers ...")
            train_proc.terminate()
            try:
                train_proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                train_proc.kill()
        log.info("Done.")


if __name__ == "__main__":
    main()
