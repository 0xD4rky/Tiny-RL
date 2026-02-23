from collections.abc import Callable
import argparse
import logging
import os
from pathlib import Path
import random
import re
from typing import Any, Optional

import yaml
import wandb
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    PreTrainedModel,
)
from datasets import load_dataset

os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
from vllm import LLM, SamplingParams
logging.getLogger("vllm").setLevel(logging.WARNING)

import gc

from loss import approx_kl_divergence, build_loss
from replay_buffer import ReplayBuffer, Experience, join_experience_batch
from rewards import Rewards, extract_boxed_answer


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_model(
    model_name_or_path: str,
    trust_remote_code: bool = False,
    bf16: bool = True,
    device_map=None,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=trust_remote_code,
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16 if bf16 else "auto",
        device_map=device_map,
    )
    return model, tokenizer


system_prompt = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.

Before answering, the Assistant thinks through the problem deeply and carefully inside <think> </think> tags. This thinking process should be thorough and exploratory — the Assistant should break the problem down, consider multiple angles or approaches, reason step by step, catch and correct any mistakes along the way, and reflect on whether the reasoning holds up before committing to an answer. The thinking should read like genuine, effortful problem-solving, not a summary.

Once the reasoning is complete, the Assistant provides a clear and concise answer inside <answer> </answer> tags.

The format is always:
<think> deep reasoning process here </think>
<answer> answer here </answer>
"""


def format_prompt(tokenizer: PreTrainedTokenizer, question: str) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def read_prompts(
    dataset_name: str,
    max_rows: Optional[int] = None,
) -> list:
    dataset = load_dataset(dataset_name, split="train")
    rows = dataset.to_list()
    for r in rows:
        r["answer"] = extract_boxed_answer(r["solution"])
    rows = [r for r in rows if r["answer"]]
    return rows


def create_vllm_engine(
    model_path: str,
    gpu_memory_utilization: float = 0.3,
    max_model_len: int = 2048,
) -> LLM:
    return LLM(
        model=model_path,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        enforce_eager=True,
    )


def destroy_vllm_engine(engine: LLM):
    del engine
    gc.collect()
    torch.cuda.empty_cache()


def sync_vllm_weights(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    engine: LLM,
    gpu_memory_utilization: float = 0.4,
) -> LLM:
    destroy_vllm_engine(engine)
    sync_dir = Path(".vllm_weights")
    sync_dir.mkdir(exist_ok=True)
    model.save_pretrained(sync_dir)
    tokenizer.save_pretrained(sync_dir)
    return create_vllm_engine(str(sync_dir), gpu_memory_utilization)


def vllm_rollout(
    engine: LLM,
    tokenizer: PreTrainedTokenizer,
    questions: list[str],
    answers: list[str],
    group_size: int,
    max_length: int,
    temperature: float,
    top_p: float,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]]:
    prompts = [format_prompt(tokenizer, q) for q in questions]

    outputs = engine.generate(
        prompts,
        [
            SamplingParams(
                n=group_size,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max(max_length - len(tokenizer.encode(p)), 1),
            )
            for p in prompts
        ],
    )

    pad_token_id = tokenizer.eos_token_id
    results = []

    for output, oracle in zip(outputs, answers):
        prompt_ids = list(output.prompt_token_ids)
        prompt_len = len(prompt_ids)

        scorer = Rewards()
        seqs, completions, rewards = [], [], []
        for comp in output.outputs:
            full_ids = prompt_ids + list(comp.token_ids)
            seqs.append(full_ids)
            completions.append(comp.text)
            rewards.append(scorer.score_completion(comp.text, oracle))

        max_seq_len = max(len(s) for s in seqs)
        sequence_ids = torch.full((group_size, max_seq_len), pad_token_id, dtype=torch.long)
        action_mask = torch.zeros(group_size, max_seq_len - 1, dtype=torch.bool)

        for i, seq in enumerate(seqs):
            sequence_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            action_mask[i, prompt_len - 1 : len(seq) - 1] = True

        returns = torch.tensor(rewards, dtype=torch.float).unsqueeze(1)
        results.append((sequence_ids, returns, action_mask, completions))

    return results


def init_rng(seed: int) -> torch.Generator:
    random.seed(seed)
    return torch.manual_seed(seed)


def group_advantages(returns: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (returns - returns.mean()) / (returns.std() + eps)


def sequence_log_probs_from_logits(
    logits: torch.Tensor, output_ids: torch.Tensor
) -> torch.Tensor:
    return -F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        output_ids.reshape(-1),
        reduction="none",
    ).reshape(output_ids.shape)


def sequences_log_probs(
    model: PreTrainedModel,
    sequence_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    chunk_size: int = 4,
) -> torch.Tensor:
    all_log_probs = []
    for i in range(0, sequence_ids.shape[0], chunk_size):
        chunk_ids = sequence_ids[i : i + chunk_size]
        chunk_mask = attention_mask[i : i + chunk_size]
        position_ids = chunk_mask.long().cumsum(dim=-1) - 1
        position_ids.masked_fill_(mask=(chunk_mask == 0), value=1)
        output = model.forward(
            input_ids=chunk_ids,
            attention_mask=chunk_mask,
            position_ids=position_ids,
            use_cache=False,
        )
        logits = output["logits"]
        log_probs = sequence_log_probs_from_logits(
            logits=logits[:, :-1].to(torch.float32),
            output_ids=chunk_ids[:, 1:],
        )
        all_log_probs.append(log_probs)
        del output, logits
    return torch.cat(all_log_probs, dim=0)

def main(cfg: dict):
    tcfg, mcfg, rcfg, dcfg = cfg["training"], cfg["model"], cfg["rollout"], cfg["dataset"]

    device = torch.device("cuda", tcfg["device_index"])
    cpu_device = torch.device("cpu")
    init_rng(tcfg["seed"])

    reference_model, _ = load_model(
        mcfg["name"], trust_remote_code=mcfg["trust_remote_code"],
        bf16=mcfg["bf16"], device_map=device,
    )
    model, tokenizer = load_model(
        mcfg["name"], trust_remote_code=mcfg["trust_remote_code"],
        bf16=mcfg["bf16"], device_map=device,
    )
    optimizer = optim.Adam(model.parameters(), lr=float(tcfg["lr"]))

    reference_model.eval()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    vllm_engine = create_vllm_engine(mcfg["name"], rcfg["gpu_memory_utilization"])
    pad_token_id = tokenizer.eos_token_id

    prompts = read_prompts(dcfg["name"], max_rows=dcfg.get("max_rows"))
    prompt_loader = DataLoader(
        prompts,
        batch_size=rcfg["rollouts_per_step"],
        shuffle=True,
        drop_last=True,
        pin_memory=False,
    )

    replay_buffer = ReplayBuffer()
    objective = build_loss(cfg["loss"])

    run_name = cfg["wandb"]["run_name"]
    wandb.init(project=cfg["wandb"]["project"], name=run_name, config=cfg)

    checkpoint_dir = Path(tcfg["checkpoint_path"]) / run_name if tcfg.get("checkpoint_path") else None
    checkpoint_interval = tcfg["checkpoint_interval"]
    total_steps = len(prompt_loader)

    print(f"training: {len(prompts)} prompts, {total_steps} steps, "
          f"group_size={rcfg['group_size']} | run={run_name}")

    for k, prompt_batch in enumerate(prompt_loader):
        replay_buffer.clear()
        questions = prompt_batch["problem"]
        answers = prompt_batch["answer"]

        rollout_results = vllm_rollout(
            vllm_engine, tokenizer, questions, answers,
            group_size=rcfg["group_size"], max_length=rcfg["max_length"],
            temperature=rcfg["temperature"], top_p=rcfg["top_p"],
        )

        all_rewards = []

        model.eval()
        with torch.no_grad():
            for i, (seq_ids, returns, act_mask, completions) in enumerate(rollout_results):
                seq_ids = seq_ids.to(device)
                returns = returns.to(device)
                act_mask = act_mask.to(device)

                all_rewards.extend(returns.squeeze(-1).tolist())

                advantages = group_advantages(returns)

                print(
                    f"  rollout q='{questions[i][:50]}', a='{answers[i]}', "
                    f"returns={returns.sum().item():.2f}, "
                    f"buf={i+1}/{len(questions)}, seq={seq_ids.shape}"
                )
                if i == 0:
                    print(f"[sample completion]\n{completions[0]}\n  [/sample]")

                attention_mask = seq_ids != pad_token_id
                log_probs = sequences_log_probs(model, seq_ids, attention_mask)
                ref_log_probs = sequences_log_probs(reference_model, seq_ids, attention_mask)
                kl = approx_kl_divergence(log_probs, ref_log_probs, act_mask)

                experience = Experience(
                    sequences=seq_ids,
                    action_log_probs=log_probs,
                    ref_log_probs=ref_log_probs,
                    returns=returns,
                    advantages=advantages,
                    attention_mask=attention_mask,
                    action_mask=act_mask,
                    kl=kl,
                )
                replay_buffer.append(experience.to(cpu_device))

        torch.cuda.empty_cache()

        reward_mean = sum(all_rewards) / len(all_rewards)
        accuracy = sum(1 for r in all_rewards if r >= 0.8) / len(all_rewards)
        print(f"returns of step {k}: reward={reward_mean:.3f}, acc={accuracy:.1%}")

        if len(replay_buffer) < tcfg["train_batch_size"]:
            print(f"skipping training: not enough signal ({len(replay_buffer)} experiences)")
            continue

        experience_sampler = DataLoader(
            replay_buffer,
            batch_size=tcfg["train_batch_size"],
            shuffle=True,
            drop_last=True,
            collate_fn=join_experience_batch,
        )

        grad_acc_steps = tcfg.get("grad_acc_steps", 1)

        for epoch in range(tcfg["epochs_per_step"]):
            model.train()
            optimizer.zero_grad()
            for micro_step, exp in enumerate(experience_sampler):
                exp = exp.to(device)

                log_probs = sequences_log_probs(
                    model, sequence_ids=exp.sequences, attention_mask=exp.attention_mask
                )
                loss, kl = objective(log_probs=log_probs, experience=exp)
                loss = loss / grad_acc_steps

                if not loss.isfinite():
                    print(f"  loss not finite, skipping: {loss.item()}")
                    continue

                loss.backward()

                if (micro_step + 1) % grad_acc_steps == 0:
                    grad_norm = clip_grad_norm_(model.parameters(), max_norm=tcfg["max_norm"])
                    optimizer.step()
                    optimizer.zero_grad()
                    print(f"  {epoch}: kl={kl.item():.4f}, grad_norm={grad_norm.item():.4f}, loss={loss.item():.4f}")

        wandb.log({
            "rollout/reward_mean": reward_mean,
            "rollout/accuracy": accuracy,
            "train/loss": loss.item(),
            "train/kl": kl.item(),
            "train/grad_norm": grad_norm.item(),
        }, step=k)

        sync_interval = rcfg.get("sync_interval", 1)
        if (k + 1) % sync_interval == 0:
            vllm_engine = sync_vllm_weights(
                model, tokenizer, vllm_engine, rcfg["gpu_memory_utilization"],
            )

        if checkpoint_dir and checkpoint_interval and (k + 1) % checkpoint_interval == 0:
            save_path = checkpoint_dir / f"step_{k+1}"
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"  checkpoint saved: {save_path}")

    if checkpoint_dir:
        save_path = checkpoint_dir / "final"
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"saved final: {save_path}")

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(load_config(args.config))
