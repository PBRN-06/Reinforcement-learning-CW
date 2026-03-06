"""
Self-Play for AlphaZero Training Data Generation

Generates games via self-play using MCTS for training the neural network.
"""

import numpy as np
from src.alphazero.mcts import MCTS
from src.alphazero.utils import (
    get_canonical_board, check_terminal, augment_example,
    get_legal_moves, apply_action
)

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
        self.mcts = MCTS(
            network,
            num_simulations=config.num_simulations,
            c_puct=config.c_puct,
            dirichlet_alpha=config.dirichlet_alpha,
            dirichlet_epsilon=config.dirichlet_epsilon
        )
        self.board_size = config.board_size

    def play_game(self):
        """
        Play one complete game and return training examples.

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
            if winner == 0:  # Draw
                example['outcome'] = 0.0
            elif winner == example['player']:  # Win
                example['outcome'] = 1.0
            else:  # Loss
                example['outcome'] = -1.0

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

    def generate_games(self, num_games: int):
        """
        Generate games sequentially (or in parallel with multiprocessing).

        Args:
            num_games: Number of games to generate

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
                examples = worker.play_game()
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

                examples = worker.play_game()
                all_examples.extend(examples)

            if not in_notebook:
                print()  # Newline after \r progress

        print(f"  Generated {num_games} games ({len(all_examples)} examples)", flush=True)

        return all_examples
