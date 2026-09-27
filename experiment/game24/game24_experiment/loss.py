"""Exact per-token clipped objective from the local method documents."""

from __future__ import annotations

import torch


def decision_objective(
    current_logp: torch.Tensor, old_logp: torch.Tensor,
    reference_logp: torch.Tensor, advantage: float,
    *, epsilon: float, kappa: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if current_logp.ndim != 1 or current_logp.numel() == 0:
        raise ValueError("Each decision needs at least one generated token")
    if old_logp.shape != current_logp.shape or reference_logp.shape != current_logp.shape:
        raise ValueError("Token probability arrays must align")
    ratio = torch.exp(current_logp - old_logp)
    unclipped = ratio * advantage
    clipped = ratio.clamp(1 - epsilon, 1 + epsilon) * advantage
    policy = torch.minimum(unclipped, clipped)
    log_ref_over_current = reference_logp - current_logp
    kl = torch.exp(log_ref_over_current) - log_ref_over_current - 1
    objective = (policy - kappa * kl).mean()
    return objective, {
        "ratio": ratio.detach(),
        "policy": policy.detach(),
        "kl": kl.detach(),
        "clipped": (clipped < unclipped).detach(),
        "ratio_outside_interval": ((ratio < 1 - epsilon) | (ratio > 1 + epsilon)).detach(),
    }
