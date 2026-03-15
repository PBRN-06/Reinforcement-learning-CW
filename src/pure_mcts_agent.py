"""
Pure Monte Carlo Tree Search Agent

Uses UCT (UCB1 applied to trees) with random rollouts — no neural network.
Stronger than random play, useful as a baseline opponent in evaluation.
"""

import math
import random
import numpy as np

from src.agent import Agent
from src.board import Board
from src.alphazero.utils import get_legal_moves, apply_action, check_terminal, action_to_coords


class MCTSNode:
    __slots__ = ['board_state', 'player', 'parent', 'action',
                 'children', 'visits', 'wins', 'untried_actions']

    def __init__(self, board_state: np.ndarray, player: int,
                 parent=None, action: int = None):
        self.board_state = board_state
        self.player = player          # player to move at this state
        self.parent = parent
        self.action = action          # action taken by parent's player to reach this node
        self.children = {}            # action -> MCTSNode
        self.visits = 0
        self.wins = 0.0               # wins from perspective of parent's player
        self.untried_actions = None   # lazy-initialised


def _check_win_at(board_state: np.ndarray, row: int, col: int,
                  player: int, board_size: int = 9) -> bool:
    """Fast win check: only examines lines through the last-played position."""
    directions = ((0, 1), (1, 0), (1, 1), (1, -1))
    for dr, dc in directions:
        count = 1
        r, c = row + dr, col + dc
        while 0 <= r < board_size and 0 <= c < board_size and board_state[r, c] == player:
            count += 1
            r += dr; c += dc
        r, c = row - dr, col - dc
        while 0 <= r < board_size and 0 <= c < board_size and board_state[r, c] == player:
            count += 1
            r -= dr; c -= dc
        if count >= 5:
            return True
    return False


class PureMCTSAgent(Agent):
    """
    Pure MCTS agent using UCT selection and random rollouts.

    Args:
        num_simulations: Number of MCTS simulations per move.
        c_puct: UCB exploration constant (sqrt(2) ≈ 1.414 is the classic value).
    """

    def __init__(self, num_simulations: int = 800, c_puct: float = 1.414):
        super().__init__()
        self.num_simulations = num_simulations
        self.c_puct = c_puct

    def command(self, board: Board, reward) -> tuple[int, int]:
        board_state = board.base.copy()
        player = self._get_current_player(board_state)
        action = self._search(board_state, player)
        return action_to_coords(action)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_current_player(self, board_state: np.ndarray) -> int:
        p1 = int(np.sum(board_state == 1))
        p2 = int(np.sum(board_state == 2))
        return 1 if p1 <= p2 else 2

    def _search(self, board_state: np.ndarray, player: int) -> int:
        root = MCTSNode(board_state, player)
        root.untried_actions = get_legal_moves(board_state)

        for _ in range(self.num_simulations):
            node = self._select(root)

            terminal = check_terminal(node.board_state)

            # Expand if node has been visited before and game isn't over
            if terminal is None and node.visits > 0:
                node = self._expand(node)
                terminal = check_terminal(node.board_state)

            # Rollout (or use known terminal result)
            if terminal is not None:
                winner = terminal
            else:
                winner = self._rollout(node.board_state, node.player)

            self._backpropagate(node, winner)

        # Robust child: pick the action with the most visits
        best_action = max(root.children, key=lambda a: root.children[a].visits)
        return best_action

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Traverse the tree using UCB until we reach a node with untried actions or a terminal."""
        while True:
            terminal = check_terminal(node.board_state)
            if terminal is not None:
                return node

            if node.untried_actions is None:
                node.untried_actions = get_legal_moves(node.board_state)

            if node.untried_actions:
                return node  # stop here; _expand will handle it

            # Fully expanded — descend via UCB
            node = self._ucb_select(node)

    def _ucb_select(self, node: MCTSNode) -> MCTSNode:
        log_n = math.log(node.visits)
        best_val = -float('inf')
        best_child = None
        for child in node.children.values():
            if child.visits == 0:
                return child  # always try unvisited children first
            ucb = child.wins / child.visits + self.c_puct * math.sqrt(log_n / child.visits)
            if ucb > best_val:
                best_val = ucb
                best_child = child
        return best_child

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Pick a random untried action and create a child node."""
        idx = random.randrange(len(node.untried_actions))
        action = node.untried_actions.pop(idx)
        next_state = apply_action(node.board_state, action, node.player)
        next_player = 3 - node.player
        child = MCTSNode(next_state, next_player, parent=node, action=action)
        child.untried_actions = get_legal_moves(next_state)
        node.children[action] = child
        return child

    def _rollout(self, board_state: np.ndarray, player: int) -> int:
        """Play random moves to a terminal state and return the winner."""
        state = board_state.copy()
        current_player = player
        board_size = state.shape[0]

        while True:
            legal = get_legal_moves(state)
            if not legal:
                return 0  # draw

            action = random.choice(legal)
            row, col = action // board_size, action % board_size
            state[row, col] = current_player

            if _check_win_at(state, row, col, current_player, board_size):
                return current_player

            # Check draw
            if not get_legal_moves(state):
                return 0

            current_player = 3 - current_player

    def _backpropagate(self, node: MCTSNode, winner: int) -> None:
        """Update visit counts and win values up the tree."""
        current = node
        while current is not None:
            current.visits += 1
            if current.parent is not None:
                mover = current.parent.player
                if winner == 0:
                    current.wins += 0.5
                elif winner == mover:
                    current.wins += 1.0
            current = current.parent
