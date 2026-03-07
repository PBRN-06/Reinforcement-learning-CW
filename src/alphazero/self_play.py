"""
Self-Play for AlphaZero Training Data Generation

Generates games via self-play using MCTS for training the neural network.
"""

import numpy as np
from src.alphazero.mcts import MCTS
from src.alphazero.mcts_batched import BatchedMCTS
from src.alphazero.utils import (
    get_canonical_board, check_terminal, augment_example,
    get_legal_moves, apply_action
)
from src.rewards import calculate_reward

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


class SelfPlayWorker:
    """Worker for generating self-play games."""

    def __init__(self, network, config):
        """
        Initialize self-play worker.

        Args:
            network: Neural network for MCTS
            config: AlphaZeroConfig instance
        """
        self.network = network
        self.config = config

        # Use batched MCTS for GPU efficiency (root parallelization)
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

        self.board_size = config.board_size

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

        # Determine number of random opening moves (for speedup)
        num_random_moves = self._get_random_moves_count(current_iteration)

        while move_count < max_moves:
            # Check if we're in random opening phase
            if move_count < num_random_moves:
                # Random opening move (no MCTS, no training data)
                legal_moves = get_legal_moves(board)
                if len(legal_moves) == 0:
                    break  # Board full (draw)
                action = np.random.choice(legal_moves)
            else:
                # Normal MCTS-guided move
                # Get canonical board (current player always sees themselves as 1)
                canonical_board = get_canonical_board(board, current_player)

                # Determine temperature (high early for exploration, low late for exploitation)
                if move_count < self.config.temperature_threshold:
                    temperature = 1.0
                else:
                    temperature = 0.1

                # Run MCTS with Dirichlet noise for exploration
                mcts_policy, _ = self.mcts.search(
                    canonical_board,
                    current_player=1,  # Always 1 in canonical form
                    add_noise=True,
                    temperature=temperature
                )

                # Store training example (before making move)
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
                # Calculate reward based on sequences (2→10, 3→100, 4→1000, 5→10000)
                shaped_reward = calculate_reward(
                    example['state'],
                    player=1,  # Already in canonical form (player is always 1)
                    board_size=self.board_size
                )
                # Normalize to [-1, 1] range: typical mid-game ~500, max ~10000
                # Using tanh for safe saturation at extreme values
                normalized_shaped = np.tanh(shaped_reward / 5000.0)

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

    def generate_games(self, num_games: int, current_iteration: int = 1):
        """
        Generate games sequentially (or in parallel with multiprocessing).

        Args:
            num_games: Number of games to generate
            current_iteration: Current training iteration (for adaptive random moves)

        Returns:
            List of all training examples from all games
        """
        all_examples = []

        # For now, generate sequentially
        # TODO: Add multiprocessing support for parallel generation
        worker = SelfPlayWorker(self.network, self.config)

        # Use tqdm if available, otherwise fall back to basic logging
        if TQDM_AVAILABLE:
            # Use tqdm progress bar (works great in notebooks and terminals)
            pbar = tqdm(range(num_games), desc="  Self-play", unit="game")
            for game_idx in pbar:
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
                    # Print every 10 games for notebook compatibility
                    if (game_idx + 1) % 10 == 0 or (game_idx + 1) == num_games:
                        print(f"  Generating game {game_idx + 1}/{num_games}...", flush=True)
                        sys.stdout.flush()  # Force immediate output
                else:
                    # Terminal: overwrite same line
                    print(f"  Generating game {game_idx + 1}/{num_games}...", end='\r')

                examples = worker.play_game(current_iteration)
                all_examples.extend(examples)

            if not in_notebook:
                print()  # Newline after \r progress

        print(f"  Generated {num_games} games ({len(all_examples)} examples)", flush=True)

        return all_examples
