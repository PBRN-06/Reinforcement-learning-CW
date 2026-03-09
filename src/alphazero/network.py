"""
AlphaZero Neural Network Architecture

ResNet-based policy-value network for Gomoku.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.alphazero.utils import encode_board_state, create_valid_move_mask


class ResidualBlock(nn.Module):
    """Residual block with two conv layers and skip connection."""

    def __init__(self, num_filters: int):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Skip connection
        out = F.relu(out)
        return out


class AlphaZeroNetwork(nn.Module):
    """
    AlphaZero neural network with ResNet backbone and dual heads.

    Architecture:
        Input: (batch, 3, board_size, board_size)
        ↓
        ConvBlock + ResBlocks
        ↓
        Policy Head (81 action logits) + Value Head (scalar [-1, 1])
    """

    def __init__(self, num_res_blocks: int = 6, num_filters: int = 64, board_size: int = 9):
        super().__init__()
        self.board_size = board_size
        self.num_actions = board_size * board_size

        # Initial convolution block
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU()
        )

        # Residual tower
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_filters) for _ in range(num_res_blocks)
        ])

        # Policy head
        self.policy_conv = nn.Conv2d(num_filters, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, self.num_actions)

        # Value head
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 64)
        self.value_fc2 = nn.Linear(64, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: (batch, 3, board_size, board_size) tensor

        Returns:
            policy_logits: (batch, num_actions) raw logits
            value: (batch, 1) value estimate in [-1, 1]
        """
        # Shared representation
        out = self.conv_block(x)
        for res_block in self.res_blocks:
            out = res_block(out)

        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(out)))
        policy = policy.view(policy.size(0), -1)  # Flatten
        policy_logits = self.policy_fc(policy)

        # Value head
        value = F.relu(self.value_bn(self.value_conv(out)))
        value = value.view(value.size(0), -1)  # Flatten
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))  # Output in [-1, 1]

        return policy_logits, value

    @torch.no_grad()
    def predict(self, board_state: np.ndarray):
        """
        Single state inference for MCTS.

        Args:
            board_state: (board_size, board_size) numpy array

        Returns:
            policy_probs: (num_actions,) numpy array of action probabilities
            value: float in [-1, 1]
        """
        # Encode board state
        encoded = encode_board_state(board_state, self.board_size)
        state_tensor = torch.FloatTensor(encoded).unsqueeze(0)  # Add batch dim

        # Move to same device as model
        device = next(self.parameters()).device
        state_tensor = state_tensor.to(device)

        # Forward pass
        policy_logits, value = self.forward(state_tensor)

        # Get valid move mask
        valid_moves = create_valid_move_mask(board_state)

        # Apply mask to policy (set invalid moves to -inf)
        policy_logits = policy_logits.cpu().numpy()[0]
        policy_logits = policy_logits - (1 - valid_moves) * 1e9

        # Softmax to get probabilities
        policy_probs = self._softmax(policy_logits)

        # Extract value
        value_scalar = value.cpu().numpy()[0, 0]

        return policy_probs, float(value_scalar)

    @torch.no_grad()
    def predict_batch(self, board_states: list):
        """
        Batched inference for multiple states (for parallel MCTS).

        Args:
            board_states: List of (board_size, board_size) numpy arrays

        Returns:
            policy_probs_batch: List of (num_actions,) numpy arrays
            values_batch: List of floats in [-1, 1]
        """
        if len(board_states) == 0:
            return [], []

        # Encode all board states
        encoded_states = []
        valid_masks = []

        for board_state in board_states:
            encoded = encode_board_state(board_state, self.board_size)
            encoded_states.append(encoded)
            valid_masks.append(create_valid_move_mask(board_state))

        # Stack into batch tensor
        state_tensor = torch.FloatTensor(np.stack(encoded_states, axis=0))

        # Move to device
        device = next(self.parameters()).device
        state_tensor = state_tensor.to(device)

        # Forward pass (batched)
        policy_logits, values = self.forward(state_tensor)

        # Move to CPU for processing
        policy_logits = policy_logits.cpu().numpy()
        values = values.cpu().numpy()

        # Apply masks and compute probabilities for each state
        policy_probs_batch = []
        values_batch = []

        for i in range(len(board_states)):
            # Apply valid move mask
            masked_logits = policy_logits[i] - (1 - valid_masks[i]) * 1e9

            # Softmax to get probabilities
            policy_probs = self._softmax(masked_logits)
            policy_probs_batch.append(policy_probs)

            # Extract value
            values_batch.append(float(values[i, 0]))

        return policy_probs_batch, values_batch

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        x_max = np.max(x)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x)

    def save(self, path: str):
        """Save model state dict."""
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        """Load model state dict."""
        self.load_state_dict(torch.load(path))
