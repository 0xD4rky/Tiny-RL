import logging
import os
import random
from typing import Optional

import yaml
import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    PreTrainedModel,
)
from datasets import load_dataset

os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
logging.getLogger("vllm").setLevel(logging.WARNING)

import gc

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
    tensor_parallel_size: int = 1,
) -> "LLM":
    from vllm import LLM

    return LLM(
        model=model_path,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
        enforce_eager=True,
    )


def destroy_vllm_engine(engine: "LLM"):
    del engine
    gc.collect()
    torch.cuda.empty_cache()


def vllm_rollout(
    engine: "LLM",
    tokenizer: PreTrainedTokenizer,
    questions: list[str],
    answers: list[str],
    group_size: int,
    max_length: int,
    temperature: float,
    top_p: float,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str]]]:
    from vllm import SamplingParams

    prompts = [format_prompt(tokenizer, q) for q in questions]

    outputs = engine.generate(
        prompts,
        [
            SamplingParams(
                n=group_size,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max(max_length - len(tokenizer.encode(p)), 1),
                logprobs=1,  # capture log prob of each sampled token
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
        seqs, completions, rewards, gen_log_probs = [], [], [], []
        for comp in output.outputs:
            full_ids = prompt_ids + list(comp.token_ids)
            seqs.append(full_ids)
            completions.append(comp.text)
            rewards.append(scorer.score_completion(comp.text, oracle))
            # comp.logprobs: list[dict[token_id -> Logprob]], one entry per generated token
            gen_log_probs.append([
                lp_dict[tok_id].logprob
                for tok_id, lp_dict in zip(comp.token_ids, comp.logprobs)
            ])

        max_seq_len = max(len(s) for s in seqs)
        sequence_ids = torch.full((group_size, max_seq_len), pad_token_id, dtype=torch.long)
        action_mask = torch.zeros(group_size, max_seq_len - 1, dtype=torch.bool)
        action_log_probs = torch.zeros(group_size, max_seq_len - 1, dtype=torch.float32)

        for i, (seq, lps) in enumerate(zip(seqs, gen_log_probs)):
            sequence_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            action_mask[i, prompt_len - 1 : len(seq) - 1] = True
            # generated tokens occupy positions [prompt_len-1 : prompt_len-1+gen_len]
            # in the (seq_len - 1) log-prob space
            gen_len = len(lps)
            action_log_probs[i, prompt_len - 1 : prompt_len - 1 + gen_len] = torch.tensor(lps)

        returns = torch.tensor(rewards, dtype=torch.float).unsqueeze(1)
        results.append((sequence_ids, returns, action_mask, action_log_probs, completions))

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
        output = model(
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
