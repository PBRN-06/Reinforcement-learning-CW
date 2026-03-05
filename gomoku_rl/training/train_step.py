"""PPO-style training step: policy loss (clipped), value loss, entropy bonus."""

import torch
import torch.nn.functional as F
import numpy as np

from gomoku_rl.config import (
    GAMMA,
    GAE_LAMBDA,
    PPO_CLIP,
    VALUE_COEF,
    ENTROPY_COEF,
    PPO_EPOCHS,
)
from gomoku_rl.models.policy_net import GomokuPolicyNet


def compute_returns_and_advantages(
    rewards: list[float],
    values: list[float],
    dones: list[bool],
    gamma: float = GAMMA,
    gae_lambda: float = GAE_LAMBDA,
    last_value: float = 0.0,
    last_done: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute returns and advantages using GAE.
    rewards, values, dones: lists of length T.
    Returns: returns (T,), advantages (T,).
    """
    rewards = np.array(rewards, dtype=np.float32)
    values = np.array(values, dtype=np.float32)
    dones = np.array(dones, dtype=np.float32)
    T = len(rewards)
    returns = np.zeros(T, dtype=np.float32)
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0.0
    next_value = last_value
    next_done = 1.0 if last_done else 0.0
    for t in reversed(range(T)):
        next_non_terminal = 1.0 - next_done
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        gae = delta + gamma * gae_lambda * next_non_terminal * gae
        advantages[t] = gae
        returns[t] = gae + values[t]
        next_value = values[t]
        next_done = dones[t]
    return torch.from_numpy(returns), torch.from_numpy(advantages)


def train_step(
    policy_net: GomokuPolicyNet,
    optimizer: torch.optim.Optimizer,
    states: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
    legal_masks: torch.Tensor,
    value_coef: float = VALUE_COEF,
    entropy_coef: float = ENTROPY_COEF,
    clip_eps: float = PPO_CLIP,
    device: torch.device | None = None,
) -> dict[str, float]:
    """
    One or more PPO epochs over the provided batch.
    states: (N, 3, 9, 9), actions: (N,) flat indices, old_log_probs: (N,),
    returns: (N,), advantages: (N,), legal_masks: (N, 81).
    Returns dict with loss components for logging.
    """
    if device is None:
        device = next(policy_net.parameters()).device
    states = states.to(device)
    actions = actions.to(device)
    old_log_probs = old_log_probs.to(device)
    returns = returns.to(device)
    advantages = advantages.to(device)
    legal_masks = legal_masks.to(device)

    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    n_epochs = 0
    for _ in range(PPO_EPOCHS):
        action_probs, logits, values = policy_net(states, legal_masks)
        values = values.squeeze(-1)
        dist = torch.distributions.Categorical(probs=action_probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(values, returns)

        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 0.5)
        optimizer.step()

        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        total_entropy += entropy.item()
        n_epochs += 1

    return {
        "policy_loss": total_policy_loss / n_epochs,
        "value_loss": total_value_loss / n_epochs,
        "entropy": total_entropy / n_epochs,
    }
