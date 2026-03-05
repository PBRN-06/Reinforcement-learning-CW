"""
AlphaZero Agent

Agent that uses trained neural network and MCTS for playing Gomoku.
Integrates with existing Agent interface.
"""

import torch
import numpy as np
from src.agent import Agent
from src.board import Board
from src.alphazero.network import AlphaZeroNetwork
from src.alphazero.mcts import MCTS
from src.alphazero.utils import get_canonical_board


class AlphaZeroAgent(Agent):
    """
    AlphaZero agent that integrates with existing Agent interface.

    Uses neural network + MCTS for move selection.
    """

    def __init__(self, checkpoint_path: str = None, num_simulations: int = 400,
                 temperature: float = 0.0, num_res_blocks: int = 6,
                 num_filters: int = 64, board_size: int = 9):
        """
        Initialize AlphaZero agent.

        Args:
            checkpoint_path: Path to trained model checkpoint (None for random weights)
            num_simulations: Number of MCTS simulations per move
            temperature: Temperature for action selection (0 = deterministic)
            num_res_blocks: Number of residual blocks in network
            num_filters: Number of filters in network
            board_size: Size of game board
        """
        super().__init__()

        self.board_size = board_size
        self.num_simulations = num_simulations
        self.temperature = temperature

        # Create network
        self.network = AlphaZeroNetwork(
            num_res_blocks=num_res_blocks,
            num_filters=num_filters,
            board_size=board_size
        )

        # Load checkpoint if provided
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)

        # Set to eval mode
        self.network.eval()

        # Create MCTS
        self.mcts = MCTS(
            self.network,
            num_simulations=num_simulations
        )

    def command(self, board: Board, reward) -> tuple[int, int]:
        """
        Implement Agent interface - select move using MCTS.

        Args:
            board: Current Board state
            reward: Reward value (not used by AlphaZero)

        Returns:
            (row, col) position to play
        """
        # Infer current player from board state
        player = self._get_current_player(board)

        # Get canonical board (current player as 1)
        canonical_board = get_canonical_board(board.base, player)

        # Run MCTS (no noise during play, use configured temperature)
        policy, _ = self.mcts.search(
            canonical_board,
            current_player=1,  # Always 1 in canonical form
            add_noise=False,
            temperature=self.temperature
        )

        # Select action
        if self.temperature == 0:
            # Deterministic: choose most probable action
            action = np.argmax(policy)
        else:
            # Stochastic: sample from policy
            action = np.random.choice(len(policy), p=policy)

        # Convert action to (row, col)
        row = action // self.board_size
        col = action % self.board_size

        return (row, col)

    def _get_current_player(self, board: Board) -> int:
        """
        Infer current player from board state.

        Args:
            board: Board instance

        Returns:
            Current player (1 or 2)
        """
        count_1 = np.sum(board.base == 1)
        count_2 = np.sum(board.base == 2)

        # If equal counts, player 1 goes next; otherwise player 2
        return 1 if count_1 == count_2 else 2

    def _load_checkpoint(self, path: str):
        """
        Load model from checkpoint.

        Args:
            path: Path to checkpoint file
        """
        print(f"Loading AlphaZero model from {path}")
        checkpoint = torch.load(path, map_location='cpu')

        # Load model weights
        if 'model_state_dict' in checkpoint:
            self.network.load_state_dict(checkpoint['model_state_dict'])
            print(f"  Loaded from iteration {checkpoint.get('iteration', 'unknown')}")
        else:
            # Direct state dict
            self.network.load_state_dict(checkpoint)
            print(f"  Loaded model weights")


class RandomNetworkAgent(Agent):
    """
    Agent using AlphaZero architecture but with random (untrained) weights.

    Useful for baseline comparisons.
    """

    def __init__(self, num_simulations: int = 400, temperature: float = 0.0,
                 num_res_blocks: int = 6, num_filters: int = 64, board_size: int = 9):
        """Initialize random network agent."""
        super().__init__()

        self.alphazero_agent = AlphaZeroAgent(
            checkpoint_path=None,  # Random weights
            num_simulations=num_simulations,
            temperature=temperature,
            num_res_blocks=num_res_blocks,
            num_filters=num_filters,
            board_size=board_size
        )

    def command(self, board: Board, reward) -> tuple[int, int]:
        """Select move using MCTS with random network."""
        return self.alphazero_agent.command(board, reward)
