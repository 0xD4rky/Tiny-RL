# Tiny-RL

This repo is inspired from https://github.com/open-thought/tiny-grpo and has support for different RL variants to train your llms on Math tasks.

## Setup

```bash
uv sync
```

## Usage

```bash
cd src
uv run train.py --config config.yaml
```

Config lives in `src/config.yaml`. Key settings:

- `model.name` — base model (default: `Qwen/Qwen3-1.7B`)
- `loss.name` — algorithm: `grpo`, `dapo`, or `reinforce_pp`
- `rollout.group_size` — completions per question
- `training.lr` — learning rate

Checkpoints save to `./output`. 
Metrics logged to wandb.

## Structure

```
src/
  train.py          # training loop + vllm rollouts
  loss.py           # grpo, dapo, reinforce++ losses
  rewards.py        # math answer extraction + reward model
  replay_buffer.py  # experience storage + batching
  config.yaml       # all hyperparameters
```
