"""Reward wrapper: sparse win/loss/draw + pattern bonuses (3/4-in-a-row, block 4) with decay."""

import numpy as np

from gomoku_rl.config import (
    BOARD_SIZE,
    WIN_LEN,
    REWARD_WIN,
    REWARD_LOSS,
    REWARD_DRAW,
    BONUS_3_IN_ROW,
    BONUS_4_IN_ROW,
    BONUS_BLOCK_4,
    PATTERN_DECAY_START,
    PATTERN_DECAY_END,
)
from gomoku_rl.env.board import GomokuBoard


def _count_patterns(board: np.ndarray, player: int, row: int, col: int) -> tuple[int, int, int]:
    """
    Scan 4 directions from (row,col). Return (has_3, has_4, opponent_4_blocked).
    has_3: 1 if player has an open/half-open 3-in-a-row including this cell.
    has_4: 1 if player has an open 4-in-a-row.
    opponent_4_blocked: 1 if this move blocks an opponent 4-in-a-row.
    """
    size = board.shape[0]
    opponent = 3 - player  # 1 -> 2, 2 -> 1 (repo convention)
    has_3 = 0
    has_4 = 0
    opp_4_blocked = 0
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

    for dr, dc in directions:
        # Count consecutive player stones in + and - direction
        count_p = 1
        open_neg = True
        open_pos = True
        r, c = row - dr, col - dc
        if 0 <= r < size and 0 <= c < size:
            if board[r, c] == player:
                count_p += 1
                rr, cc = r - dr, c - dc
                while 0 <= rr < size and 0 <= cc < size and board[rr, cc] == player:
                    count_p += 1
                    rr -= dr
                    cc -= dc
            elif board[r, c] == opponent:
                open_neg = False
        r, c = row + dr, col + dc
        if 0 <= r < size and 0 <= c < size:
            if board[r, c] == player:
                count_p += 1
                rr, cc = r + dr, c + dc
                while 0 <= rr < size and 0 <= cc < size and board[rr, cc] == player:
                    count_p += 1
                    rr += dr
                    cc += dc
            elif board[r, c] == opponent:
                open_pos = False
        if count_p >= 4 and (open_neg or open_pos):
            has_4 = 1
        if count_p == 3 and (open_neg or open_pos):
            has_3 = 1

        # Opponent 4 blocked: in this direction opponent had 4 in a row with one open end, we just filled it
        count_o = 0
        r, c = row, col
        for _ in range(WIN_LEN):
            if 0 <= r < size and 0 <= c < size and board[r, c] == opponent:
                count_o += 1
            else:
                break
            r += dr
            c += dc
        r, c = row - dr, col - dc
        for _ in range(WIN_LEN):
            if 0 <= r < size and 0 <= c < size and board[r, c] == opponent:
                count_o += 1
            else:
                break
            r -= dr
            c -= dc
        if count_o >= 4:
            opp_4_blocked = 1

    return has_3, has_4, opp_4_blocked


def compute_pattern_bonus(
    board: np.ndarray,
    player: int,
    row: int,
    col: int,
    bonus_3: float = BONUS_3_IN_ROW,
    bonus_4: float = BONUS_4_IN_ROW,
    bonus_block: float = BONUS_BLOCK_4,
) -> float:
    """Return scalar bonus for the move (row, col) by player on given board."""
    has_3, has_4, opp_block = _count_patterns(board, player, row, col)
    return has_3 * bonus_3 + has_4 * bonus_4 + opp_block * bonus_block


class RewardWrapper:
    """
    Wraps board step: adds sparse win/loss/draw and optional pattern bonuses.
    Pattern bonus scale decays from PATTERN_DECAY_START to PATTERN_DECAY_END
    as training_step increases (set via set_decay_factor or decay_for_step).
    """

    def __init__(
        self,
        win_reward: float = REWARD_WIN,
        loss_reward: float = REWARD_LOSS,
        draw_reward: float = REWARD_DRAW,
        bonus_3: float = BONUS_3_IN_ROW,
        bonus_4: float = BONUS_4_IN_ROW,
        bonus_block: float = BONUS_BLOCK_4,
        decay_start: float = PATTERN_DECAY_START,
        decay_end: float = PATTERN_DECAY_END,
        decay_steps: int = 50_000,
    ):
        self.win_reward = win_reward
        self.loss_reward = loss_reward
        self.draw_reward = draw_reward
        self.bonus_3 = bonus_3
        self.bonus_4 = bonus_4
        self.bonus_block = bonus_block
        self.decay_start = decay_start
        self.decay_end = decay_end
        self.decay_steps = decay_steps
        self._training_step = 0
        self._decay_factor = decay_start

    def set_decay_factor(self, factor: float) -> None:
        """Set pattern bonus multiplier directly (e.g. 1.0 -> 0.2 over training)."""
        self._decay_factor = max(self.decay_end, min(self.decay_start, factor))

    def decay_for_step(self, step: int) -> None:
        """Update decay factor from current training step (linear decay)."""
        self._training_step = step
        t = min(1.0, step / max(1, self.decay_steps))
        self._decay_factor = self.decay_start + t * (self.decay_end - self.decay_start)

    def step(
        self,
        board: GomokuBoard,
        row: int,
        col: int,
    ) -> tuple[np.ndarray, float, bool, dict]:
        """
        Perform board step and add rewards: sparse outcome + decayed pattern bonus.
        """
        state, reward, done, info = board.step(row, col)
        player_who_moved = board.board[row, col]  # already set by board.step

        if done:
            winner = info.get("winner", 0)
            if winner == 0:
                reward = self.draw_reward
            else:
                reward = self.win_reward if winner == player_who_moved else self.loss_reward
        else:
            bonus = compute_pattern_bonus(
                board.board,
                player_who_moved,
                row,
                col,
                self.bonus_3,
                self.bonus_4,
                self.bonus_block,
            )
            reward = bonus * self._decay_factor

        return state, reward, done, info
