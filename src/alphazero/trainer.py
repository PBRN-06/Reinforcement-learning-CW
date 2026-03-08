"""
AlphaZero Trainer

Main training loop that coordinates self-play, network updates, and checkpointing.
"""

import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from src.alphazero.network import AlphaZeroNetwork
from src.alphazero.self_play import SelfPlayManager
from src.data.replay_buffer import ReplayBuffer
from src.alphazero.utils import encode_board_state, get_legal_moves

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


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

        # Create learning rate scheduler (reduces LR when loss plateaus)
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     self.optimizer,
        #     mode='min',           # Minimize loss
        #     factor=0.5,           # Reduce LR by 50% when triggered
        #     patience=10,          # Wait 10 iterations before reducing
        #     min_lr=1e-6          # Don't go below this LR
        # )
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=[150,300,500],
            gamma=0.1
        )

        # Create replay buffer
        self.replay_buffer = ReplayBuffer(max_size=config.replay_buffer_size)

        # Training state
        self.iteration = 0
        self.total_games = 0

        # Create directories
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)

        # Training metrics log
        self.metrics_log_path = os.path.join(config.log_dir, "training_metrics.json")
        self.metrics_history = []

    def _get_random_moves_count(self, iteration: int) -> int:
        """
        Calculate number of random opening moves based on training iteration.
        Implements adaptive decay schedule for faster early training.

        Args:
            iteration: Current training iteration

        Returns:
            Number of random moves to use at game start
        """
        max_random = self.config.random_opening_moves
        max_iterations = self.config.num_iterations

        # Adaptive schedule: decay random moves as training progresses
        if iteration <= max_iterations * 0.3:  # First 30% of training
            return max_random
        elif iteration <= max_iterations * 0.6:  # Next 30% (30-60%)
            return max(max_random // 2, 0)
        elif iteration <= max_iterations * 0.8:  # Next 20% (60-80%)
            return max(max_random // 5, 0)
        else:  # Final 20%
            return 0

    def train(self):
        """Main training loop."""
        print(f"\n{'=' * 60}")
        print(f"Starting AlphaZero Training")
        print(f"Device: {self.device}")
        print(f"Network: {self.config.num_res_blocks} ResBlocks, {self.config.num_filters} filters")
        print(f"MCTS: {self.config.num_simulations} simulations per move")

        # Show MCTS batching info
        if hasattr(self.config, 'mcts_batch_size') and self.config.mcts_batch_size > 1:
            print(f"MCTS Batching: Enabled (batch_size={self.config.mcts_batch_size}) - GPU optimized")
        else:
            print(f"MCTS Batching: Disabled (sequential evaluation)")

        print(f"Generation: Every {self.config.generation_frequency} iteration(s)")
        print(f"{'=' * 60}\n")

        # Start from current iteration (supports resuming from checkpoint)
        start_iteration = self.iteration
        for iteration in range(start_iteration, self.config.num_iterations):
            self.iteration = iteration + 1

            print(f"\n{'=' * 60}")
            print(f"Iteration {self.iteration}/{self.config.num_iterations}")
            print(f"{'=' * 60}")

            # Step 1: Self-play (conditional based on generation_frequency)
            # Always generate if buffer is empty (e.g., after resuming from checkpoint)
            should_generate = (((self.config.generation_frequency == 1) or (self.iteration % self.config.generation_frequency == 1)) or (self.iteration == 1) 
                               or self.replay_buffer.is_empty())

            if should_generate:
                # Calculate random opening moves for this iteration
                num_random_moves = self._get_random_moves_count(self.iteration)
                if num_random_moves > 0:
                    print(f"\n[1/3] Generating {self.config.games_per_iteration} self-play games...")
                    print(f"      Using {num_random_moves} random opening moves for speedup")
                else:
                    print(f"\n[1/3] Generating {self.config.games_per_iteration} self-play games...")
                    print(f"      Using full MCTS from move 1 (no random openings)")

                self_play_manager = SelfPlayManager(self.network, self.config)
                examples = self_play_manager.generate_games(
                    self.config.games_per_iteration,
                    current_iteration=self.iteration
                )

                # Step 2: Add to replay buffer
                print(f"\n[2/3] Adding {len(examples)} examples to replay buffer...")
                self.replay_buffer.add_examples(examples)
                self.total_games += self.config.games_per_iteration
                print(f"  Replay buffer size: {len(self.replay_buffer)} examples")
            else:
                print(f"\n[1/3] Skipping game generation (frequency={self.config.generation_frequency})")
                print(f"      Training on existing replay buffer ({len(self.replay_buffer)} examples)")
                print(f"\n[2/3] Replay buffer unchanged")

            # Step 3: Train network
            print(f"\n[3/3] Training network for {self.config.epochs_per_iteration} epochs...")
            train_metrics = self.train_network()

            # Step 3.5: Update learning rate scheduler based on loss
            self.scheduler.step(train_metrics['total_loss'])

            print(f"\n{'=' * 60}")
            print(f"Iteration {self.iteration} Summary:")
            print(f"  Total games played: {self.total_games}")
            print(f"  Policy loss: {train_metrics['policy_loss']:.4f}")
            print(f"  Value loss: {train_metrics['value_loss']:.4f}")
            print(f"  Total loss: {train_metrics['total_loss']:.4f}")
            print(f"  Network entropy: {train_metrics['net_entropy']:.4f}")
            print(f"  Target entropy: {train_metrics['target_entropy']:.4f}")
            print(f"{'=' * 60}")

            # Step 4: Log metrics
            self._log_metrics(train_metrics)

            # Step 5: Save checkpoint
            if self.__should_save_checkpoint():
                self.save_checkpoint()

        print(f"\n{'=' * 60}")
        print(f"Training Complete!")
        print(f"{'=' * 60}\n")
        
    def __should_save_checkpoint(self) -> bool:
        if self.iteration < 20:
            return True          # every iteration early on
        elif self.iteration < 50:
            return self.iteration % 3 == 0
        elif self.iteration < 100:
            return self.iteration % 5 == 0
        else:
            return self.iteration % 10 == 0

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
        total_net_entropy = 0.0
        total_target_entropy = 0.0
        num_batches = 0

        for epoch in range(self.config.epochs_per_iteration):
            # Create DataLoader
            data_loader = self.replay_buffer.get_data_loader(
                batch_size=self.config.batch_size,
                shuffle=True
            )

            # Wrap with tqdm if available
            if TQDM_AVAILABLE:
                data_loader = tqdm(data_loader, desc=f"  Epoch {epoch+1}/{self.config.epochs_per_iteration}", leave=False)

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
                mask = self.__batch_mask(policy_logits, states)
                policy_loss = self._policy_loss(policy_logits, policy_targets, mask)
                value_loss = self._value_loss(value_pred, value_targets)
                net_entropy, target_entropy = self._entropy(policy_logits, policy_targets, mask)
                # Weight value loss to balance gradient magnitudes (policy ~2.8, value ~0.08)
                # This ensures value head gets sufficient gradient signal
                loss = policy_loss + 0.5 * value_loss

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
                total_net_entropy += net_entropy.item()
                total_target_entropy += target_entropy.item()
                num_batches += 1

                # Update progress bar with loss
                if TQDM_AVAILABLE and hasattr(data_loader, 'set_postfix'):
                    data_loader.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'p_loss': f'{policy_loss.item():.4f}',
                        'v_loss': f'{value_loss.item():.4f}'
                    })

        # Average metrics
        return {
            'policy_loss': total_policy_loss / max(num_batches, 1),
            'value_loss': total_value_loss / max(num_batches, 1),
            'total_loss': total_loss / max(num_batches, 1),
            'net_entropy': total_net_entropy / max(num_batches, 1),
            'target_entropy': total_target_entropy / max(num_batches, 1),
        }

    def __batch_mask(self, logits, states):
        mask = torch.zeros_like(logits, dtype=torch.bool)
        for i,state in enumerate(states):
            legal = get_legal_moves(state)
            mask[i, legal] = True

        return mask

    def _policy_loss(self, logits, targets, batch_mask):
        """
        Cross-entropy loss between MCTS policy and network policy.

        Args:
            logits: (batch, num_actions) network policy logits
            targets: (batch, num_actions) MCTS policy distribution
            batch_mask: logits mask for invalid states

        Returns:
            Scalar loss
        """
        masked_logits = logits.clone()
        masked_logits[~batch_mask] = -float('inf')

        # Fix -inf * 0 = nan by zeroing out -inf positions in log_probs
        log_probs = F.log_softmax(masked_logits, dim=1)
        log_probs = torch.where(
            torch.isfinite(log_probs),
            log_probs,
            torch.zeros_like(log_probs)
        )

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
    
    def _entropy(self, logits, targets, batch_mask):
        """
        Entropy of current loss

        Args:
            logits: (batch, num_actions) network policy logits
            targets: (batch, num_actions) MCTS policy distribution
            batch_mask: logits mask for invalid states

        Returns:
            (Scalar,Scalar) network entropy, target entropy
        """
        masked_logits = logits.clone()
        masked_logits[~batch_mask] = -float('inf')

        with torch.no_grad():
            predicted_probs = F.softmax(masked_logits, dim=1)

            masked_logits = logits.clone()
            masked_logits[~batch_mask] = -float('inf')

            network_entropy = -torch.sum(predicted_probs * torch.log(predicted_probs + 1e-8), dim=1).mean()

            target_entropy = -torch.sum(targets * torch.log(targets + 1e-8), dim=1).mean()

            return network_entropy,target_entropy

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

    def _log_metrics(self, train_metrics):
        """
        Log training metrics to JSON file.

        Args:
            train_metrics: Dict with policy_loss, value_loss, total_loss
        """
        metrics_entry = {
            'iteration': self.iteration,
            'total_games': self.total_games,
            'policy_loss': train_metrics['policy_loss'],
            'value_loss': train_metrics['value_loss'],
            'total_loss': train_metrics['total_loss'],
            'network_entropy': train_metrics['net_entropy'],
            'target_entropy': train_metrics['target_entropy']
        }

        self.metrics_history.append(metrics_entry)

        # Save to JSON file (overwrite each time)
        with open(self.metrics_log_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)

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
            'scheduler_state_dict': self.scheduler.state_dict(),
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
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.iteration = checkpoint['iteration']
        self.total_games = checkpoint['total_games']

        # Load scheduler state if available (backward compatibility)
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f"  Loaded scheduler state")
        else:
            print(f"  No scheduler state in checkpoint (old checkpoint format)")

        # Load existing metrics history if available
        if os.path.exists(self.metrics_log_path):
            with open(self.metrics_log_path, 'r') as f:
                self.metrics_history = json.load(f)
            print(f"  Loaded {len(self.metrics_history)} metric entries from log")

            # Remove any metrics after the resumed checkpoint to avoid duplicates
            self.metrics_history = [m for m in self.metrics_history if m['iteration'] <= self.iteration]
            print(f"  Keeping {len(self.metrics_history)} metrics up to iteration {self.iteration}")

            # Save filtered metrics immediately to disk
            with open(self.metrics_log_path, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)
            print(f"  Saved cleaned metrics to {self.metrics_log_path}")

        print(f"  Resumed from iteration {self.iteration}")
        print(f"  Total games played: {self.total_games}")
