"""9x9 Gomoku board: state, legal moves, win/draw check, 5-in-a-row.
Aligned with repo: 0=empty, 1=P1, 2=P2 (same as src.board.Board.base).
"""

import numpy as np

from gomoku_rl.config import BOARD_SIZE, WIN_LEN, STATE_CHANNELS


class GomokuBoard:
    """9x9 Gomoku board. Empty=0, Player 1=1, Player 2=2 (matches repo Board.base)."""

    def __init__(self):
        self.size = BOARD_SIZE
        self.win_len = WIN_LEN
        self.board = np.zeros((self.size, self.size), dtype=np.int8)
        self.current_player = 1  # 1 or 2
        self.turn_count = 0
        self.last_move: tuple[int, int] | None = None
        self._done = False
        self._winner: int | None = None  # 1, 2, or 0 for draw

    def reset(self) -> "GomokuBoard":
        """Reset board to initial state."""
        self.board.fill(0)
        self.current_player = 1
        self.turn_count = 0
        self.last_move = None
        self._done = False
        self._winner = None
        return self

    def to_state_tensor(self) -> np.ndarray:
        """Return (STATE_CHANNELS, size, size) for NN: [P1, P2, empty]."""
        p1 = (self.board == 1).astype(np.float32)
        p2 = (self.board == 2).astype(np.float32)
        empty = (self.board == 0).astype(np.float32)
        return np.stack([p1, p2, empty], axis=0)

    @staticmethod
    def state_from_repo_board(base: np.ndarray) -> np.ndarray:
        """Build (3,9,9) state from repo Board.base (0=empty, 1=P1, 2=P2)."""
        p1 = (base == 1).astype(np.float32)
        p2 = (base == 2).astype(np.float32)
        empty = (base == 0).astype(np.float32)
        return np.stack([p1, p2, empty], axis=0)

    def legal_actions(self) -> list[tuple[int, int]]:
        """List of (row, col) legal moves (empty cells)."""
        if self._done:
            return []
        return [
            (r, c)
            for r in range(self.size)
            for c in range(self.size)
            if self.board[r, c] == 0
        ]

    def legal_action_indices(self) -> np.ndarray:
        """Flat indices in [0, 80] that are legal."""
        legal = self.legal_actions()
        return np.array([r * self.size + c for r, c in legal], dtype=np.int64)

    def step(self, row: int, col: int) -> tuple[np.ndarray, float, bool, dict]:
        """
        Play (row, col) for current player.
        Returns: (next_state, reward, done, info).
        Reward is 0 here; use RewardWrapper for win/loss/pattern bonuses.
        """
        if self._done:
            raise RuntimeError("Game already finished")
        if self.board[row, col] != 0:
            raise ValueError(f"Cell ({row},{col}) is not empty")
        self.board[row, col] = self.current_player
        self.last_move = (row, col)
        self.turn_count += 1

        winner = self._check_winner(row, col)
        if winner is not None:
            self._done = True
            self._winner = winner
            reward = 0.0
            return self.to_state_tensor(), reward, True, {"winner": winner}

        if self.turn_count >= self.size * self.size:
            self._done = True
            self._winner = 0
            return self.to_state_tensor(), 0.0, True, {"winner": 0}

        self.current_player = 3 - self.current_player  # 1 -> 2, 2 -> 1
        return self.to_state_tensor(), 0.0, False, {}

    def _check_winner(self, row: int, col: int) -> int | None:
        """Return 1 or 2 if that player just won, else None. Check 4 directions."""
        p = int(self.board[row, col])
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            for sign in (-1, 1):
                r, c = row + sign * dr, col + sign * dc
                while 0 <= r < self.size and 0 <= c < self.size and self.board[r, c] == p:
                    count += 1
                    r += sign * dr
                    c += sign * dc
            if count >= self.win_len:
                return p
        return None

    @property
    def done(self) -> bool:
        return self._done

    @property
    def winner(self) -> int | None:
        return self._winner

    def copy(self) -> "GomokuBoard":
        """Deep copy of the board."""
        b = GomokuBoard()
        b.board = self.board.copy()
        b.current_player = self.current_player
        b.turn_count = self.turn_count
        b.last_move = self.last_move
        b._done = self._done
        b._winner = self._winner
        return b
