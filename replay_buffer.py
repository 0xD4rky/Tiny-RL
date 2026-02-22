from dataclasses import dataclass, fields
from typing import Optional, Self, List

import torch
import torch.nn.functional as F


def zero_pad_sequences(
    sequences: List[torch.Tensor], padding_side: str = "left"
) -> torch.Tensor:
    assert padding_side in ["left", "right"]
    max_len = max(seq.size(-1) for seq in sequences)
    padded = []
    for seq in sequences:
        pad_len = max_len - seq.size(-1)
        pad = (pad_len, 0) if padding_side == "left" else (0, pad_len)
        padded.append(F.pad(seq, pad))
    return torch.stack(padded, dim=0)


@dataclass
class Experience:
    sequences: torch.Tensor
    action_log_probs: torch.Tensor
    ref_log_probs: torch.Tensor
    action_mask: torch.Tensor
    returns: Optional[torch.Tensor] = None
    advantages: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None
    kl: Optional[torch.Tensor] = None

    def to(self, device: torch.device) -> Self:
        members = {}
        for field in fields(self):
            v = getattr(self, field.name)
            if isinstance(v, torch.Tensor):
                v = v.to(device=device)
            members[field.name] = v
        return Experience(**members)


EXPERIENCE_KEYS = (
    "sequences",
    "action_log_probs",
    "ref_log_probs",
    "action_mask",
    "returns",
    "advantages",
    "attention_mask",
    "kl",
)


def split_experience_batch(experience: Experience) -> list[Experience]:
    batch_size = experience.sequences.size(0)
    batch_data = [{} for _ in range(batch_size)]
    for key in EXPERIENCE_KEYS:
        value = getattr(experience, key)
        if value is None:
            vals = [None] * batch_size
        else:
            vals = torch.unbind(value)
        for i, v in enumerate(vals):
            batch_data[i][key] = v
    return [Experience(**data) for data in batch_data]


def join_experience_batch(items: list[Experience]) -> Experience:
    batch_data = {}
    for key in EXPERIENCE_KEYS:
        vals = [getattr(item, key) for item in items]
        if all(v is not None for v in vals):
            batch_data[key] = zero_pad_sequences(vals, "left")
        else:
            batch_data[key] = None
    return Experience(**batch_data)


class ReplayBuffer:
    def __init__(self, limit: int = 0) -> None:
        self.limit = limit
        self.items: list[Experience] = []

    def append(self, experience: Experience) -> None:
        items = split_experience_batch(experience)
        self.items.extend(items)
        if self.limit > 0:
            samples_to_remove = len(self.items) - self.limit
            if samples_to_remove > 0:
                self.items = self.items[samples_to_remove:]

    def clear(self) -> None:
        self.items.clear()

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Experience:
        return self.items[idx]

