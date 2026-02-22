import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict

from replay_buffer import Experience


def approx_kl_divergence(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    action_mask: Optional[torch.Tensor],
) -> torch.Tensor:

    log_ratio = ref_log_probs.float() - log_probs.float()
    if action_mask is not None:
        log_ratio = log_ratio * action_mask

    return log_ratio.exp() - log_ratio - 1

def masked_mean(
    tensor: torch.Tensor,
    mask: Optional[torch.Tensor],
    dim: int = None,
) -> torch.Tensor:
    if mask is None:
        return tensor.mean(axis=dim)
    return (tensor * mask).sum(axis=dim) / mask.sum(axis=dim)


class GRPOLoss(nn.Module):
    """
    grpo actor loss
    """

    def __init__(self, clip_eps: float, kl_beta: float) -> None:
        super().__init__()
        self.clip_eps = clip_eps
        self.kl_beta = kl_beta
    
    def forward(
        self,
        log_probs: torch.Tensor,
        experience: Experience,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        old_log_probs = experience.action_log_probs
        ref_log_probs = experience.ref_log_probs
        advantages = experience.advantages
        action_mask = experience.action_mask

        kl = approx_kl_divergence(
            log_probs=log_probs,
            ref_log_probs=ref_log_probs,
            action_mask=action_mask,
        )

        ratio = (log_probs - old_log_probs).exp() 
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)

        loss = -torch.min(ratio * advantages, clipped_ratio * advantages) + self.kl_beta * kl
        loss = masked_mean(loss, action_mask, dim=-1).mean()

        return loss, kl.mean()


class DAPOLoss(nn.Module):
    """
    dapo actor loss
    """

    def __init__(self, clip_low: float = 0.2, clip_high: float = 0.28) -> None:
        super().__init__()
        self.clip_low = clip_low
        self.clip_high = clip_high

    def forward(
        self,
        log_probs: torch.Tensor,
        experience: Experience,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        old_log_probs = experience.action_log_probs
        ref_log_probs = experience.ref_log_probs
        advantages = experience.advantages
        action_mask = experience.action_mask

        kl = approx_kl_divergence(
            log_probs=log_probs,
            ref_log_probs=ref_log_probs,
            action_mask=action_mask,
        )

        ratio = (log_probs - old_log_probs).exp()
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_low, 1 + self.clip_high)

        surrogate = torch.min(ratio * advantages, clipped_ratio * advantages)

        if action_mask is not None:
            loss = -(surrogate * action_mask).sum() / action_mask.sum()
        else:
            loss = -surrogate.mean()

        return loss, kl.mean()


class ReinforcePPLoss(nn.Module):
    """
    REINFORCE++ loss
    """

    def __init__(self, clip_eps: float = 0.2, kl_beta: float = 0.04) -> None:
        super().__init__()
        self.clip_eps = clip_eps
        self.kl_beta = kl_beta

    def forward(
        self,
        log_probs: torch.Tensor,
        experience: Experience,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        old_log_probs = experience.action_log_probs
        ref_log_probs = experience.ref_log_probs
        advantages = experience.advantages
        action_mask = experience.action_mask

        kl = approx_kl_divergence(
            log_probs=log_probs,
            ref_log_probs=ref_log_probs,
            action_mask=action_mask,
        )

        shaped_advantages = advantages - self.kl_beta * kl

        ratio = (log_probs - old_log_probs).exp()
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)

        loss = -torch.min(ratio * shaped_advantages, clipped_ratio * shaped_advantages)
        loss = masked_mean(loss, action_mask, dim=-1).mean()

        return loss, kl.mean()


LOSS_REGISTRY: Dict[str, type] = {
    "grpo": GRPOLoss,
    "dapo": DAPOLoss,
    "reinforce_pp": ReinforcePPLoss,
}

LOSS_ARGS: Dict[str, list[str]] = {
    "grpo": ["clip_eps", "kl_beta"],
    "dapo": ["clip_low", "clip_high"],
    "reinforce_pp": ["clip_eps", "kl_beta"],
}


def build_loss(cfg: Dict) -> nn.Module:
    name = cfg["name"]
    kwargs = {k: cfg[k] for k in LOSS_ARGS[name]}
    return LOSS_REGISTRY[name](**kwargs)
