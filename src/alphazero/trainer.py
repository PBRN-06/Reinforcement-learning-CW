"""
AlphaZero Trainer

Main training loop that coordinates self-play, network updates, and checkpointing.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from src.alphazero.network import AlphaZeroNetwork
from src.alphazero.self_play import SelfPlayManager
from src.data.replay_buffer import ReplayBuffer
from src.alphazero.utils import encode_board_state


class AlphaZeroTrainer:
    """Orchestrates the AlphaZero training process."""

    def __init__(self, config):
        """
        Initialize trainer.

        Args:
            config: AlphaZeroConfig instance
        """
        self.config = config

        # Create network
        self.network = AlphaZeroNetwork(
            num_res_blocks=config.num_res_blocks,
            num_filters=config.num_filters,
            board_size=config.board_size
        )

        # Move to device
        self.device = torch.device(config.device)
        self.network.to(self.device)

        # Create optimizer
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Create replay buffer
        self.replay_buffer = ReplayBuffer(max_size=config.replay_buffer_size)

        # Training state
        self.iteration = 0
        self.total_games = 0

        # Create directories
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)

    def train(self):
        """Main training loop."""
        print(f"\n{'=' * 60}")
        print(f"Starting AlphaZero Training")
        print(f"Device: {self.device}")
        print(f"Network: {self.config.num_res_blocks} ResBlocks, {self.config.num_filters} filters")
        print(f"MCTS: {self.config.num_simulations} simulations per move")
        print(f"{'=' * 60}\n")

        for iteration in range(self.config.num_iterations):
            self.iteration = iteration + 1

            print(f"\n{'=' * 60}")
            print(f"Iteration {self.iteration}/{self.config.num_iterations}")
            print(f"{'=' * 60}")

            # Step 1: Self-play
            print(f"\n[1/3] Generating {self.config.games_per_iteration} self-play games...")
            self_play_manager = SelfPlayManager(self.network, self.config)
            examples = self_play_manager.generate_games(self.config.games_per_iteration)

            # Step 2: Add to replay buffer
            print(f"\n[2/3] Adding {len(examples)} examples to replay buffer...")
            self.replay_buffer.add_examples(examples)
            self.total_games += self.config.games_per_iteration
            print(f"  Replay buffer size: {len(self.replay_buffer)} examples")

            # Step 3: Train network
            print(f"\n[3/3] Training network for {self.config.epochs_per_iteration} epochs...")
            train_metrics = self.train_network()

            print(f"\n{'=' * 60}")
            print(f"Iteration {self.iteration} Summary:")
            print(f"  Total games played: {self.total_games}")
            print(f"  Policy loss: {train_metrics['policy_loss']:.4f}")
            print(f"  Value loss: {train_metrics['value_loss']:.4f}")
            print(f"  Total loss: {train_metrics['total_loss']:.4f}")
            print(f"{'=' * 60}")

            # Step 4: Save checkpoint
            if self.iteration % self.config.checkpoint_freq == 0:
                self.save_checkpoint()

        print(f"\n{'=' * 60}")
        print(f"Training Complete!")
        print(f"{'=' * 60}\n")

    def train_network(self):
        """
        Train network on replay buffer for multiple epochs.

        Returns:
            Dict with training metrics
        """
        self.network.train()

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_loss = 0.0
        num_batches = 0

        for epoch in range(self.config.epochs_per_iteration):
            # Create DataLoader
            data_loader = self.replay_buffer.get_data_loader(
                batch_size=self.config.batch_size,
                shuffle=True
            )

            for batch in data_loader:
                states, policies, outcomes = batch

                # Encode states to 3-channel format
                state_tensors = self._encode_batch(states)
                state_tensors = state_tensors.to(self.device)
                policy_targets = policies.to(self.device)
                value_targets = outcomes.to(self.device)

                # Forward pass
                policy_logits, value_pred = self.network(state_tensors)

                # Compute losses
                policy_loss = self._policy_loss(policy_logits, policy_targets)
                value_loss = self._value_loss(value_pred, value_targets)
                loss = policy_loss + value_loss

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(),
                    self.config.grad_clip
                )

                self.optimizer.step()

                # Track metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_loss += loss.item()
                num_batches += 1

        # Average metrics
        return {
            'policy_loss': total_policy_loss / max(num_batches, 1),
            'value_loss': total_value_loss / max(num_batches, 1),
            'total_loss': total_loss / max(num_batches, 1)
        }

    def _policy_loss(self, logits, targets):
        """
        Cross-entropy loss between MCTS policy and network policy.

        Args:
            logits: (batch, num_actions) network policy logits
            targets: (batch, num_actions) MCTS policy distribution

        Returns:
            Scalar loss
        """
        # Log-softmax on logits, then negative log-likelihood
        log_probs = F.log_softmax(logits, dim=1)
        loss = -torch.sum(targets * log_probs, dim=1).mean()
        return loss

    def _value_loss(self, pred, target):
        """
        Mean squared error between predicted and actual outcome.

        Args:
            pred: (batch, 1) predicted values
            target: (batch, 1) actual outcomes

        Returns:
            Scalar loss
        """
        return F.mse_loss(pred, target)

    def _encode_batch(self, states):
        """
        Encode batch of board states to 3-channel tensors.

        Args:
            states: (batch, board_size, board_size) tensor

        Returns:
            (batch, 3, board_size, board_size) tensor
        """
        batch_size = states.shape[0]
        board_size = states.shape[1]
        encoded = torch.zeros(batch_size, 3, board_size, board_size, dtype=torch.float32)

        states_np = states.numpy() if isinstance(states, torch.Tensor) else states

        for i in range(batch_size):
            encoded[i] = torch.from_numpy(encode_board_state(states_np[i], board_size))

        return encoded

    def save_checkpoint(self):
        """Save training checkpoint."""
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            f"checkpoint_{self.iteration}.pt"
        )

        torch.save({
            'iteration': self.iteration,
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_games': self.total_games,
            'config': self.config
        }, checkpoint_path)

        print(f"\n  Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, path: str):
        """
        Load training checkpoint.

        Args:
            path: Path to checkpoint file
        """
        print(f"\nLoading checkpoint: {path}")
        checkpoint = torch.load(path, map_location=self.device)

        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.iteration = checkpoint['iteration']
        self.total_games = checkpoint['total_games']

        print(f"  Resumed from iteration {self.iteration}")
        print(f"  Total games played: {self.total_games}")
