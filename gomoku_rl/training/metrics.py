"""Metrics: Win/Loss ratio, Turns per game (running window)."""

from collections import deque
import numpy as np


class MetricsLogger:
    """Running window of last METRICS_WINDOW games: wins, losses, turns."""

    def __init__(self, window: int = 100):
        self.window = window
        self.wins_p1: deque[int] = deque(maxlen=window)
        self.wins_p2: deque[int] = deque(maxlen=window)
        self.draws: deque[int] = deque(maxlen=window)
        self.turns_per_game: deque[int] = deque(maxlen=window)

    def log_game(self, winner: int, turns: int) -> None:
        """winner: 1 = P1 wins, 2 = P2 wins, 0 = draw (repo convention)."""
        if winner == 1:
            self.wins_p1.append(1)
            self.wins_p2.append(0)
            self.draws.append(0)
        elif winner == 2:
            self.wins_p1.append(0)
            self.wins_p2.append(1)
            self.draws.append(0)
        else:
            self.wins_p1.append(0)
            self.wins_p2.append(0)
            self.draws.append(1)
        self.turns_per_game.append(turns)

    def win_ratio_p1(self) -> float:
        """Win rate as player 1 in window."""
        if not self.wins_p1:
            return 0.0
        return sum(self.wins_p1) / len(self.wins_p1)

    def win_ratio_p2(self) -> float:
        """Win rate as player 2 in window."""
        if not self.wins_p2:
            return 0.0
        return sum(self.wins_p2) / len(self.wins_p2)

    def turns_mean_std(self) -> tuple[float, float]:
        """(mean, std) of turns per game in window."""
        if not self.turns_per_game:
            return 0.0, 0.0
        arr = np.array(self.turns_per_game, dtype=np.float32)
        return float(np.mean(arr)), float(np.std(arr))

    def summary(self) -> dict[str, float]:
        """Dict for logging/TensorBoard."""
        mean_t, std_t = self.turns_mean_std()
        return {
            "win_ratio_p1": self.win_ratio_p1(),
            "win_ratio_p2": self.win_ratio_p2(),
            "draw_ratio": sum(self.draws) / len(self.draws) if self.draws else 0.0,
            "turns_per_game_mean": mean_t,
            "turns_per_game_std": std_t,
            "games_in_window": len(self.turns_per_game),
        }
