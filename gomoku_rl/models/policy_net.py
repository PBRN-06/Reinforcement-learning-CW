"""CNN-based Actor-Critic policy network for 9x9 Gomoku."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gomoku_rl.config import (
    BOARD_SIZE,
    STATE_CHANNELS,
    ACTION_SIZE,
    CONV_FILTERS,
    CONV_KERNEL,
    CONV_PADDING,
    LINEAR_HIDDEN,
    USE_BATCH_NORM,
)


class GomokuPolicyNet(nn.Module):
    """
    3-layer CNN backbone + Actor (policy) and Critic (value) heads.
    Input: (batch, 3, 9, 9). Output: action_probs (81), value scalar.
    """

    def __init__(
        self,
        in_channels: int = STATE_CHANNELS,
        board_size: int = BOARD_SIZE,
        action_size: int = ACTION_SIZE,
        conv_filters: list[int] | None = None,
        linear_hidden: int = LINEAR_HIDDEN,
        use_batch_norm: bool = USE_BATCH_NORM,
    ):
        super().__init__()
        self.board_size = board_size
        self.action_size = action_size
        conv_filters = conv_filters or CONV_FILTERS
        layers = []
        c_in = in_channels
        for c_out in conv_filters:
            layers.append(
                nn.Conv2d(c_in, c_out, CONV_KERNEL, padding=CONV_PADDING)
            )
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(c_out))
            layers.append(nn.ReLU(inplace=True))
            c_in = c_out
        self.backbone = nn.Sequential(*layers)
        self.feature_size = c_in * board_size * board_size
        self.fc = nn.Linear(self.feature_size, linear_hidden)
        self.actor_head = nn.Linear(linear_hidden, action_size)
        self.critic_head = nn.Linear(linear_hidden, 1)

    def forward(
        self,
        x: torch.Tensor,
        legal_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: (B, 3, 9, 9)
        legal_mask: (B, 81) True for legal actions. If None, no masking.
        Returns: action_probs (B, 81), log_probs (B, 81) for selected actions later, value (B, 1)
        """
        B = x.shape[0]
        features = self.backbone(x)
        features = features.view(B, -1)
        features = F.relu(self.fc(features))
        logits = self.actor_head(features)
        if legal_mask is not None:
            logits = logits.masked_fill(~legal_mask, -1e9)
        action_probs = F.softmax(logits, dim=-1)
        value = self.critic_head(features)
        return action_probs, logits, value

    def get_action_and_log_prob(
        self,
        x: torch.Tensor,
        legal_mask: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample (or argmax) action and return action index, log_prob, value.
        Returns: action_idx (B,), log_prob (B,), value (B, 1), action_probs (B, 81)
        """
        action_probs, logits, value = self.forward(x, legal_mask)
        dist = torch.distributions.Categorical(probs=action_probs)
        if deterministic:
            action_idx = action_probs.argmax(dim=-1)
        else:
            action_idx = dist.sample()
        log_prob = dist.log_prob(action_idx)
        return action_idx, log_prob, value, action_probs

    def predict(
        self,
        board_state: np.ndarray | torch.Tensor,
        legal_actions: list[tuple[int, int]] | np.ndarray | None = None,
        deterministic: bool = True,
        device: torch.device | None = None,
    ) -> tuple[int, int]:
        """
        Standard API: predict(board_state) -> (x, y).
        board_state: (3,9,9) or (9,9). If (9,9), converted to 3-channel.
        legal_actions: list of (row,col) or None (use all empty cells).
        Returns (row, col) in 0..8.
        """
        if device is None:
            device = next(self.parameters()).device
        self.eval()
        with torch.no_grad():
            if isinstance(board_state, np.ndarray):
                if board_state.ndim == 2:
                    # Repo convention: 0=empty, 1=P1, 2=P2. Legacy: 0, 1, -1.
                    if (board_state == 2).any() or (board_state.max() <= 2 and board_state.min() >= 0):
                        p1 = (board_state == 1).astype(np.float32)
                        p2 = (board_state == 2).astype(np.float32)
                        empty = (board_state == 0).astype(np.float32)
                    else:
                        p1 = (board_state == 1).astype(np.float32)
                        p2 = (board_state == -1).astype(np.float32)
                        empty = (board_state == 0).astype(np.float32)
                    board_state = np.stack([p1, p2, empty], axis=0)
                x = torch.from_numpy(board_state).float().unsqueeze(0).to(device)
            else:
                x = board_state.float().unsqueeze(0).to(device)
            if legal_actions is None:
                # Infer from state: empty channel (index 2) > 0.5
                if x.shape[1] >= 3:
                    empty = x[0, 2].view(-1) > 0.5
                    legal_mask = empty.unsqueeze(0)
                else:
                    legal_mask = torch.ones(1, self.action_size, dtype=torch.bool, device=device)
            else:
                if isinstance(legal_actions, np.ndarray):
                    indices = legal_actions
                else:
                    indices = [r * self.board_size + c for r, c in legal_actions]
                legal_mask = torch.zeros(1, self.action_size, dtype=torch.bool, device=device)
                legal_mask[0, indices] = True
            _, log_prob, _, action_probs = self.get_action_and_log_prob(
                x, legal_mask, deterministic=deterministic
            )
            action_idx = action_probs.argmax(dim=-1).item() if deterministic else torch.distributions.Categorical(probs=action_probs).sample().item()
        row = action_idx // self.board_size
        col = action_idx % self.board_size
        return (int(row), int(col))
