"""
Utility functions for AlphaZero implementation.

Includes board encoding, legal move generation, and game state management.
"""

import numpy as np
import os
import glob
import re
from src.rewards import count_sequences


def encode_board_state(board_state: np.ndarray, board_size: int = 9) -> np.ndarray:
    """
    Encode board state as 3-channel tensor for neural network input.

    Args:
        board_state: (board_size, board_size) numpy array where
                     0=empty, 1=Player 1, 2=Player 2
        board_size: Size of the board (default 9)

    Returns:
        (3, board_size, board_size) numpy array with channels:
        - Channel 0: Player 1 pieces (binary)
        - Channel 1: Player 2 pieces (binary)
        - Channel 2: Empty squares (binary)
    """
    encoded = np.zeros((3, board_size, board_size), dtype=np.float32)
    encoded[0] = (board_state == 1).astype(np.float32)  # Player 1
    encoded[1] = (board_state == 2).astype(np.float32)  # Player 2
    encoded[2] = (board_state == 0).astype(np.float32)  # Empty
    return encoded


def get_legal_moves(board_state: np.ndarray) -> list:
    """
    Get list of legal move indices.

    Args:
        board_state: (board_size, board_size) numpy array

    Returns:
        List of action indices (0 to board_size²-1) for empty positions
    """
    board_size = board_state.shape[0]
    legal = []
    for i in range(board_size):
        for j in range(board_size):
            if board_state[i, j] == 0:
                legal.append(i * board_size + j)
    return legal


def action_to_coords(action: int, board_size: int = 9) -> tuple:
    """
    Convert action index to board coordinates.

    Args:
        action: Action index (0 to board_size²-1)
        board_size: Size of the board

    Returns:
        (row, col) tuple
    """
    return (action // board_size, action % board_size)


def coords_to_action(row: int, col: int, board_size: int = 9) -> int:
    """
    Convert board coordinates to action index.

    Args:
        row: Row index
        col: Column index
        board_size: Size of the board

    Returns:
        Action index
    """
    return row * board_size + col


def apply_action(board_state: np.ndarray, action: int, player: int) -> np.ndarray:
    """
    Apply action to board state (non-destructive).

    Args:
        board_state: (board_size, board_size) numpy array
        action: Action index (0 to board_size²-1)
        player: Current player (1 or 2)

    Returns:
        New board state with action applied
    """
    new_state = board_state.copy()
    row, col = action_to_coords(action, board_state.shape[0])
    new_state[row, col] = player
    return new_state


def get_canonical_board(board_state: np.ndarray, player: int) -> np.ndarray:
    """
    Convert board to canonical form where current player is always '1'.

    This ensures the neural network sees consistent perspective
    regardless of which player is moving.

    Args:
        board_state: (board_size, board_size) numpy array with 0=empty, 1=P1, 2=P2
        player: Current player (1 or 2)

    Returns:
        Canonical board state where current player is represented as 1
    """
    if player == 1:
        return board_state.copy()
    else:
        # Swap 1s and 2s
        canonical = np.zeros_like(board_state)
        canonical[board_state == 1] = 2
        canonical[board_state == 2] = 1
        return canonical


def check_terminal(board_state: np.ndarray, board_size: int = None) -> int | None:
    """
    Check if game is terminal and return winner.

    Args:
        board_state: (board_size, board_size) numpy array
        board_size: Size of the board

    Returns:
        0 (draw), 1 (player 1 wins), 2 (player 2 wins), or None (not terminal)
    """
    if board_size is None:
        board_size = board_state.shape[0]

    # Check Player 1
    seqs = count_sequences(board_state, 1, board_size)
    if seqs[5] >= 1:
        return 1

    # Check Player 2
    seqs = count_sequences(board_state, 2, board_size)
    if seqs[5] >= 1:
        return 2

    # Check draw (board full)
    if np.sum(board_state == 0) == 0:
        return 0

    return None  # Game continues


def get_terminal_value(winner: int, current_player: int) -> float:
    """
    Get the terminal value from current player's perspective.

    Args:
        winner: Winner of the game (0=draw, 1=player 1, 2=player 2)
        current_player: Current player (1 or 2)

    Returns:
        1.0 for win, -1.0 for loss, 0.0 for draw
    """
    if winner == 0:
        return 0.0  # Draw
    elif winner == current_player:
        return 1.0  # Win
    else:
        return -1.0  # Loss


def augment_example(state: np.ndarray, policy: np.ndarray, outcome: float) -> list:
    """
    Generate 8 symmetrically equivalent examples via rotations and flips.

    Args:
        state: (board_size, board_size) board state
        policy: (board_size²,) MCTS policy vector
        outcome: Game outcome (-1, 0, or 1)

    Returns:
        List of 8 (state, policy, outcome) tuples
    """
    board_size = state.shape[0]
    augmented = []

    # Reshape policy to 2D for transformations
    policy_2d = policy.reshape(board_size, board_size)

    for k in range(4):  # 4 rotations
        # Rotate state and policy
        rot_state = np.rot90(state, k)
        rot_policy = np.rot90(policy_2d, k)

        # Add rotated version
        augmented.append({
            'state': rot_state.copy(),
            'policy': rot_policy.flatten().copy(),
            'outcome': outcome
        })

        # Add flipped version
        flip_state = np.fliplr(rot_state)
        flip_policy = np.fliplr(rot_policy)

        augmented.append({
            'state': flip_state.copy(),
            'policy': flip_policy.flatten().copy(),
            'outcome': outcome
        })

    return augmented


def create_valid_move_mask(board_state: np.ndarray) -> np.ndarray:
    """
    Create a mask for valid moves (1 for legal, 0 for illegal).

    Args:
        board_state: (board_size, board_size) numpy array

    Returns:
        (board_size²,) binary mask
    """
    board_size = board_state.shape[0]
    mask = (board_state == 0).astype(np.float32).flatten()
    return mask


def find_latest_checkpoint(checkpoint_dir: str, current_iteration: int) -> str | None:
    """
    Find the most recent checkpoint before the current iteration.

    Args:
        checkpoint_dir: Directory containing checkpoints
        current_iteration: Current training iteration

    Returns:
        Path to latest checkpoint, or None if no checkpoints exist
    """
    if not os.path.exists(checkpoint_dir):
        return None

    # Find all checkpoint files
    checkpoint_pattern = os.path.join(checkpoint_dir, "checkpoint_*.pt")
    checkpoint_files = glob.glob(checkpoint_pattern)

    if not checkpoint_files:
        return None

    # Extract iteration numbers from filenames
    checkpoint_iterations = []
    for filepath in checkpoint_files:
        filename = os.path.basename(filepath)
        match = re.search(r'checkpoint_(\d+)\.pt', filename)
        if match:
            iteration = int(match.group(1))
            # Only consider checkpoints before current iteration
            if iteration < current_iteration:
                checkpoint_iterations.append((iteration, filepath))

    if not checkpoint_iterations:
        return None

    # Return the most recent checkpoint
    checkpoint_iterations.sort(key=lambda x: x[0], reverse=True)
    latest_path = checkpoint_iterations[0][1]

    return latest_path


def find_checkpoint_pool(checkpoint_dir: str, current_iteration: int,
                        pool_size_max: int = 15) -> list:
    """
    Find a pool of recent checkpoints before the current iteration.

    Returns the most recent checkpoints up to pool_size_max.
    No minimum requirement - returns whatever is available.

    Args:
        checkpoint_dir: Directory containing checkpoints
        current_iteration: Current training iteration
        pool_size_max: Maximum number of checkpoints to return

    Returns:
        List of checkpoint paths (most recent first), or empty list if no checkpoints
    """
    if not os.path.exists(checkpoint_dir):
        return []

    # Find all checkpoint files
    checkpoint_pattern = os.path.join(checkpoint_dir, "checkpoint_*.pt")
    checkpoint_files = glob.glob(checkpoint_pattern)

    if not checkpoint_files:
        return []

    # Extract iteration numbers from filenames
    checkpoint_iterations = []
    for filepath in checkpoint_files:
        filename = os.path.basename(filepath)
        match = re.search(r'checkpoint_(\d+)\.pt', filename)
        if match:
            iteration = int(match.group(1))
            # Only consider checkpoints before current iteration
            if iteration < current_iteration:
                checkpoint_iterations.append((iteration, filepath))

    if not checkpoint_iterations:
        return []

    # Sort by iteration (most recent first)
    checkpoint_iterations.sort(key=lambda x: x[0], reverse=True)

    # Return up to pool_size_max checkpoints
    pool_paths = [path for _, path in checkpoint_iterations[:pool_size_max]]

    return pool_paths
