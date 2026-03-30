"""
Self-Play for AlphaZero Training Data Generation

Generates games via self-play using MCTS for training the neural network.
"""

import numpy as np
import torch
from src.alphazero.mcts import MCTS
from src.alphazero.mcts_batched import BatchedMCTS
from src.alphazero.utils import (
    get_canonical_board, check_terminal, augment_example,
    get_legal_moves, apply_action, find_latest_checkpoint, find_checkpoint_pool
)
from src.rewards import calculate_reward
from src.alphazero.network import AlphaZeroNetwork

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


class SelfPlayWorker:
    """Worker for generating self-play games."""

    def __init__(self, network, config, opponent_network=None):
        """
        Initialize self-play worker.

        Args:
            network: Neural network for MCTS (current player)
            config: AlphaZeroConfig instance
            opponent_network: Optional opponent network (for vs past checkpoint games)
        """
        self.network = network
        self.opponent_network = opponent_network
        self.config = config

        # Create MCTS for main network
        if hasattr(config, 'mcts_batch_size') and config.mcts_batch_size > 1:
            self.mcts = BatchedMCTS(
                network,
                num_simulations=config.num_simulations,
                batch_size=config.mcts_batch_size,
                c_puct=config.c_puct,
                dirichlet_alpha=config.dirichlet_alpha,
                dirichlet_epsilon=config.dirichlet_epsilon
            )
            self.using_batched_mcts = True
        else:
            # Fallback to standard MCTS
            self.mcts = MCTS(
                network,
                num_simulations=config.num_simulations,
                c_puct=config.c_puct,
                dirichlet_alpha=config.dirichlet_alpha,
                dirichlet_epsilon=config.dirichlet_epsilon
            )
            self.using_batched_mcts = False

        # Create separate MCTS for opponent if provided
        if opponent_network is not None:
            if hasattr(config, 'mcts_batch_size') and config.mcts_batch_size > 1:
                self.opponent_mcts = BatchedMCTS(
                    opponent_network,
                    num_simulations=config.num_simulations,
                    batch_size=config.mcts_batch_size,
                    c_puct=config.c_puct,
                    dirichlet_alpha=config.dirichlet_alpha,
                    dirichlet_epsilon=config.dirichlet_epsilon
                )
            else:
                self.opponent_mcts = MCTS(
                    opponent_network,
                    num_simulations=config.num_simulations,
                    c_puct=config.c_puct,
                    dirichlet_alpha=config.dirichlet_alpha,
                    dirichlet_epsilon=config.dirichlet_epsilon
                )
        else:
            self.opponent_mcts = None

        self.board_size = config.board_size

    def _get_temperature(self, move_count: int, iteration: int) -> float:
        """
        Calculate temperature for move sampling with adaptive decay.
        Higher temperature early in training for more exploration.

        Args:
            move_count: Current move number in the game
            iteration: Current training iteration

        Returns:
            Temperature value for move sampling
        """
        max_iterations = self.config.num_iterations
        threshold = self.config.temperature_threshold

        # Get temperature bounds (with decay for high temp)
        temp_high = self.config.temperature_high if hasattr(self.config, 'temperature_high') else 1.0
        temp_low = self.config.temperature_low if hasattr(self.config, 'temperature_low') else 0.1

        # Decay high temperature over training (more exploration early)
        if iteration <= max_iterations * 0.5:  # First half: full exploration
            high_temp = temp_high
        elif iteration <= max_iterations * 0.75:  # Third quarter: reduce
            high_temp = temp_high * 0.7
        else:  # Final quarter: minimal
            high_temp = temp_high * 0.5

        # Return temperature based on move count
        if move_count < threshold:
            return high_temp  # Exploration phase
        else:
            return temp_low  # Exploitation phase

    def play_game(self, current_iteration: int = 1):
        """
        Play one complete game and return training examples.

        Args:
            current_iteration: Current training iteration (for adaptive random moves)

        Returns:
            List of training examples with augmentation
        """
        # Initialize game state
        board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        current_player = 1
        examples = []
        move_count = 0
        max_moves = self.board_size * self.board_size

        while move_count < max_moves:
            # Normal MCTS-guided move
            # Get canonical board (current player always sees themselves as 1)
            canonical_board = get_canonical_board(board, current_player)

            # Determine temperature with adaptive decay
            temperature = self._get_temperature(move_count, current_iteration)

            # Choose which MCTS to use based on current player
            # Player 1 always uses main network
            # Player 2 uses opponent network if available, otherwise uses main (self-play)
            if current_player == 2 and self.opponent_mcts is not None:
                # Play against past checkpoint
                mcts_policy, _ = self.opponent_mcts.search(
                    canonical_board,
                    current_player=1,  # Always 1 in canonical form
                    add_noise=False,   # Stable target
                    temperature=0.1    #  "
                )
            else:
                # Self-play or player 1 move
                mcts_policy, _ = self.mcts.search(
                    canonical_board,
                    current_player=1,  # Always 1 in canonical form
                    add_noise=True,
                    temperature=temperature
                )

            # Store training example (before making move), ignoring checkpoint agents since those are fixed
            is_checkpoint_opponent_move = (current_player == 2 and self.opponent_mcts is not None)

            if not is_checkpoint_opponent_move:
                examples.append({
                    'state': canonical_board.copy(),
                    'policy': mcts_policy.copy(),
                    'player': current_player
                })

            # Sample action from MCTS policy
            legal_moves = get_legal_moves(board)
            if len(legal_moves) == 0:
                break  # Board full (draw)

            # Normalize policy over legal moves only
            legal_policy = mcts_policy[legal_moves]
            if np.sum(legal_policy) > 0:
                legal_policy = legal_policy / np.sum(legal_policy)
                action = np.random.choice(legal_moves, p=legal_policy)
            else:
                # Fallback to uniform random
                action = np.random.choice(legal_moves)

            # Apply action
            board = apply_action(board, action, current_player)
            move_count += 1

            # Check for terminal state
            winner = check_terminal(board, self.board_size)
            if winner is not None:
                break

            # Switch player
            current_player = 3 - current_player  # 1->2, 2->1

        # Determine final outcome
        winner = check_terminal(board, self.board_size)

        # Assign outcomes to all examples
        for example in examples:
            # Base terminal outcome (win/loss/draw)
            if winner == 0:  # Draw
                base_outcome = 0.0
            elif winner == example['player']:  # Win
                base_outcome = 1.0
            else:  # Loss
                base_outcome = -1.0

            # Add shaped reward if enabled (provides dense intermediate feedback)
            if self.config.reward_shaping_weight > 0:
                # Calculate reward based on sequences (2→1, 3→5, 4→50)
                # Opponent penalties are asymmetric: (2→-3, 3→-10, 4→-80)
                shaped_reward = calculate_reward(
                    example['state'],
                    player=1,  # Already in canonical form (player is always 1)
                    board_size=self.board_size
                )
                # Normalize to [-1, 1] range using tanh
                # Typical mid-game: ~5-20 → normalized 0.05-0.20
                # Strong position: ~50 → normalized ~0.46
                # Opponent threat: ~-80 → normalized ~-0.66
                normalized_shaped = np.tanh(shaped_reward / 100.0)

                # Combine: base outcome (dominant) + shaped component (guidance)
                example['outcome'] = base_outcome + self.config.reward_shaping_weight * normalized_shaped
            else:
                # No reward shaping (original AlphaZero behavior)
                example['outcome'] = base_outcome

        # Apply data augmentation (8-fold symmetry)
        augmented_examples = []
        for example in examples:
            augmented = augment_example(
                example['state'],
                example['policy'],
                example['outcome']
            )
            augmented_examples.extend(augmented)

        return augmented_examples


class RandomOpponentWorker:
    """Worker for generating games against a random opponent (no MCTS)."""

    def __init__(self, network, config):
        """
        Initialize random opponent worker.

        Args:
            network: Neural network for MCTS (current player only)
            config: AlphaZeroConfig instance
        """
        self.network = network
        self.config = config

        # Create MCTS for main network only
        if hasattr(config, 'mcts_batch_size') and config.mcts_batch_size > 1:
            self.mcts = BatchedMCTS(
                network,
                num_simulations=config.num_simulations,
                batch_size=config.mcts_batch_size,
                c_puct=config.c_puct,
                dirichlet_alpha=config.dirichlet_alpha,
                dirichlet_epsilon=config.dirichlet_epsilon
            )
        else:
            self.mcts = MCTS(
                network,
                num_simulations=config.num_simulations,
                c_puct=config.c_puct,
                dirichlet_alpha=config.dirichlet_alpha,
                dirichlet_epsilon=config.dirichlet_epsilon
            )

        self.board_size = config.board_size

    def _get_temperature(self, move_count: int, iteration: int) -> float:
        """Calculate temperature for move sampling (same as SelfPlayWorker)."""
        max_iterations = self.config.num_iterations
        threshold = self.config.temperature_threshold

        temp_high = self.config.temperature_high if hasattr(self.config, 'temperature_high') else 1.0
        temp_low = self.config.temperature_low if hasattr(self.config, 'temperature_low') else 0.1

        if iteration <= max_iterations * 0.5:
            high_temp = temp_high
        elif iteration <= max_iterations * 0.75:
            high_temp = temp_high * 0.7
        else:
            high_temp = temp_high * 0.5

        if move_count < threshold:
            return high_temp
        else:
            return temp_low

    def play_game(self, current_iteration: int = 1):
        """
        Play one game against random opponent (player 2 plays randomly).

        Args:
            current_iteration: Current training iteration

        Returns:
            List of training examples with augmentation
        """
        board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        current_player = 1
        examples = []
        move_count = 0
        max_moves = self.board_size * self.board_size

        while move_count < max_moves:
            legal_moves = get_legal_moves(board)
            if len(legal_moves) == 0:
                break

            if current_player == 1:
                # Player 1 uses MCTS
                canonical_board = get_canonical_board(board, current_player)
                temperature = self._get_temperature(move_count, current_iteration)

                mcts_policy, _ = self.mcts.search(
                    canonical_board,
                    current_player=1,
                    add_noise=True,
                    temperature=temperature
                )

                # Store training example
                examples.append({
                    'state': canonical_board.copy(),
                    'policy': mcts_policy.copy(),
                    'player': current_player
                })

                # Sample action from MCTS policy
                legal_policy = mcts_policy[legal_moves]
                if np.sum(legal_policy) > 0:
                    legal_policy = legal_policy / np.sum(legal_policy)
                    action = np.random.choice(legal_moves, p=legal_policy)
                else:
                    action = np.random.choice(legal_moves)
            else:
                # Player 2 plays randomly (no training data)
                action = np.random.choice(legal_moves)

            # Apply action
            board = apply_action(board, action, current_player)
            move_count += 1

            # Check for terminal state
            winner = check_terminal(board, self.board_size)
            if winner is not None:
                break

            # Switch player
            current_player = 3 - current_player

        # Assign outcomes (same as SelfPlayWorker)
        winner = check_terminal(board, self.board_size)

        for example in examples:
            if winner == 0:
                base_outcome = 0.0
            elif winner == example['player']:
                base_outcome = 1.0
            else:
                base_outcome = -1.0

            if self.config.reward_shaping_weight > 0:
                shaped_reward = calculate_reward(
                    example['state'],
                    player=1,
                    board_size=self.board_size
                )
                normalized_shaped = np.tanh(shaped_reward / 100.0)
                example['outcome'] = base_outcome + self.config.reward_shaping_weight * normalized_shaped
            else:
                example['outcome'] = base_outcome

        # Apply data augmentation
        augmented_examples = []
        for example in examples:
            augmented = augment_example(
                example['state'],
                example['policy'],
                example['outcome']
            )
            augmented_examples.extend(augmented)

        return augmented_examples


class SelfPlayManager:
    """Manages parallel self-play game generation."""

    def __init__(self, network, config):
        """
        Initialize self-play manager.

        Args:
            network: Neural network for MCTS
            config: AlphaZeroConfig instance
        """
        self.network = network
        self.config = config
        self.opponent_networks = []  # Pool of opponent networks
        self.checkpoint_pool_paths = []  # Current pool of checkpoint paths

    def _load_checkpoint_pool(self, checkpoint_paths: list):
        """
        Load a pool of opponent networks from checkpoints.

        Args:
            checkpoint_paths: List of paths to checkpoint files
        """
        print(f"  Loading checkpoint pool ({len(checkpoint_paths)} checkpoints)...")

        device = torch.device(self.config.device)
        self.opponent_networks = []

        for checkpoint_path in checkpoint_paths:
            # Create opponent network with same architecture
            opponent_network = AlphaZeroNetwork(
                num_res_blocks=self.config.num_res_blocks,
                num_filters=self.config.num_filters,
                board_size=self.config.board_size
            )

            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            opponent_network.load_state_dict(checkpoint['model_state_dict'])
            opponent_network.to(device)
            opponent_network.eval()

            self.opponent_networks.append(opponent_network)

        print(f"  Loaded {len(self.opponent_networks)} opponents into pool")

    def _sample_opponent_type(self, pool_size: int) -> str:
        """
        Sample opponent type using adaptive rolloff based on pool size.

        Args:
            pool_size: Number of checkpoints in the pool

        Returns:
            'self_play', 'pool', or 'random'
        """
        roll = np.random.random()

        if pool_size == 0:
            # No checkpoints yet - pure self-play or random
            return 'self_play' if roll < 0.7 else 'random'

        elif pool_size < 5:
            # Small pool - lean more on self-play, use what we have
            if roll < 0.4:
                return 'self_play'
            elif roll < 0.7:
                return 'pool'
            else:
                return 'random'

        elif pool_size < 15:
            # Medium pool - standard distribution
            if roll < 0.2:
                return 'self_play'
            elif roll < 0.8:
                return 'pool'
            else:
                return 'random'

        else:
            # Full pool (>=15) - no random opponents, they produce unrealistic positions
            if roll < 0.2:
                return 'self_play'
            else:
                return 'pool'

    def generate_games(self, num_games: int, current_iteration: int = 1):
        """
        Generate games with adaptive opponent selection based on pool size.

        Opponent distribution:
        - No checkpoints: 70% self-play, 30% random
        - Small pool (<5): 40% self-play, 30% pool, 30% random
        - Full pool (≥5): 20% self-play, 60% pool, 20% random

        Args:
            num_games: Number of games to generate
            current_iteration: Current training iteration (for adaptive random moves)

        Returns:
            List of all training examples from all games
        """
        all_examples = []

        # Load checkpoint pool
        pool_paths = find_checkpoint_pool(
            self.config.checkpoint_dir,
            current_iteration,
            pool_size_max=self.config.pool_size_max
        )

        # Only reload if pool has changed
        if pool_paths != self.checkpoint_pool_paths:
            if pool_paths:
                self._load_checkpoint_pool(pool_paths)
            self.checkpoint_pool_paths = pool_paths

        pool_size = len(self.opponent_networks)

        # Create workers
        self_play_worker = SelfPlayWorker(self.network, self.config)
        random_worker = RandomOpponentWorker(self.network, self.config)

        # Pre-create pool workers (one for each opponent in pool)
        pool_workers = []
        if pool_size > 0:
            for opponent_net in self.opponent_networks:
                pool_workers.append(
                    SelfPlayWorker(self.network, self.config, opponent_network=opponent_net)
                )

        # Determine distribution message based on pool size
        if pool_size == 0:
            dist_msg = "70% self-play, 30% random (no checkpoints)"
        elif pool_size < 5:
            dist_msg = f"40% self-play, 30% pool ({pool_size} ckpts), 30% random"
        elif pool_size < 15:
            dist_msg = f"20% self-play, 60% pool ({pool_size} ckpts), 20% random"
        else:
            dist_msg = f"20% self-play, 80% pool ({pool_size} ckpts), 0% random"

        print(f"  Opponent distribution: {dist_msg}")

        # Generate games using adaptive sampling
        game_counts = {'self_play': 0, 'pool': 0, 'random': 0}

        # Use tqdm if available, otherwise fall back to basic logging
        if TQDM_AVAILABLE:
            pbar = tqdm(range(num_games), desc="  Self-play", unit="game")
            for game_idx in pbar:
                # Sample opponent type using adaptive rolloff
                game_type = self._sample_opponent_type(pool_size)
                game_counts[game_type] += 1

                if game_type == 'pool':
                    # Randomly select from pool
                    worker = np.random.choice(pool_workers)
                elif game_type == 'self_play':
                    worker = self_play_worker
                else:  # random
                    worker = random_worker

                examples = worker.play_game(current_iteration)
                all_examples.extend(examples)
                pbar.set_postfix({"examples": len(all_examples)})
            pbar.close()
        else:
            # Fallback: Detect if running in Jupyter/Colab
            try:
                from IPython import get_ipython
                in_notebook = get_ipython() is not None
            except:
                in_notebook = False

            import sys

            for game_idx in range(num_games):
                if in_notebook:
                    if (game_idx + 1) % 10 == 0 or (game_idx + 1) == num_games:
                        print(f"  Generating game {game_idx + 1}/{num_games}...", flush=True)
                        sys.stdout.flush()
                else:
                    print(f"  Generating game {game_idx + 1}/{num_games}...", end='\r')

                # Sample opponent type using adaptive rolloff
                game_type = self._sample_opponent_type(pool_size)
                game_counts[game_type] += 1

                if game_type == 'pool':
                    worker = np.random.choice(pool_workers)
                elif game_type == 'self_play':
                    worker = self_play_worker
                else:  # random
                    worker = random_worker

                examples = worker.play_game(current_iteration)
                all_examples.extend(examples)

            if not in_notebook:
                print()

        # Log actual game distribution
        print(f"  Generated {num_games} games: {game_counts['self_play']} self-play, "
              f"{game_counts['pool']} pool, {game_counts['random']} random ({len(all_examples)} examples)", flush=True)

        return all_examples
