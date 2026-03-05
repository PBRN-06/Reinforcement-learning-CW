"""Light Monte Carlo Tree Search using policy network for 3-5 move lookahead."""

import math
import numpy as np
import torch

from gomoku_rl.config import BOARD_SIZE, MCTS_SIMULATIONS, MCTS_C_PUCT, ACTION_SIZE
from gomoku_rl.models.policy_net import GomokuPolicyNet
from gomoku_rl.env.board import GomokuBoard


class MCTSNode:
    """Single node: board after move that led here, children, visit count, total value."""

    __slots__ = ("board", "parent", "action", "children", "N", "W", "P")

    def __init__(
        self,
        board: GomokuBoard,
        parent: "MCTSNode | None" = None,
        action: tuple[int, int] | None = None,
        prior: float = 0.0,
    ):
        self.board = board
        self.parent = parent
        self.action = action
        self.children: dict[tuple[int, int], MCTSNode] = {}
        self.N = 0
        self.W = 0.0
        self.P = prior

    def Q(self) -> float:
        return self.W / self.N if self.N > 0 else 0.0

    def U(self, c_puct: float) -> float:
        if self.parent is None:
            return 0.0
        return c_puct * math.sqrt(self.parent.N) / (1 + self.N)


class MCTS:
    """
    MCTS using policy_net for policy prior and value. Run 3-5 simulations per move.
    Output: (x, y) from most-visits child, or blend with raw policy.
    """

    def __init__(
        self,
        policy_net: GomokuPolicyNet,
        n_simulations: int = MCTS_SIMULATIONS,
        c_puct: float = MCTS_C_PUCT,
        device: torch.device | None = None,
    ):
        self.policy_net = policy_net
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.device = device or next(policy_net.parameters()).device
        self.policy_net.eval()

    def _get_prior_and_value(
        self,
        board: GomokuBoard,
    ) -> tuple[dict[tuple[int, int], float], float]:
        """Get policy prior over legal actions and state value from network."""
        legal = board.legal_actions()
        if not legal:
            return {}, 0.0
        state = board.to_state_tensor()
        indices = [r * BOARD_SIZE + c for r, c in legal]
        x = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        legal_mask = torch.zeros(1, ACTION_SIZE, dtype=torch.bool, device=self.device)
        legal_mask[0, indices] = True
        with torch.no_grad():
            action_probs, _, value = self.policy_net(x, legal_mask)
        action_probs = action_probs[0].cpu().numpy()
        prior = {a: float(action_probs[r * BOARD_SIZE + c]) for (r, c) in legal}
        v = float(value[0, 0].cpu().item())
        return prior, v

    def run(
        self,
        board: GomokuBoard,
    ) -> tuple[int, int]:
        """
        Run n_simulations and return (row, col) with highest visit count.
        """
        root = MCTSNode(board.copy())
        prior, value = self._get_prior_and_value(root.board)
        root.W = value
        root.N = 1
        legal = root.board.legal_actions()
        for (r, c) in legal:
            b_child = root.board.copy()
            b_child.step(r, c)
            root.children[(r, c)] = MCTSNode(
                b_child,
                parent=root,
                action=(r, c),
                prior=prior.get((r, c), 1.0 / max(len(legal), 1)),
            )

        for _ in range(self.n_simulations - 1):
            node = root
            # Select until we find an unvisited or leaf
            while node.children:
                best_action = None
                best_score = -float("inf")
                for (r, c), child in node.children.items():
                    u = self.c_puct * child.P * math.sqrt(node.N) / (1 + child.N)
                    score = child.Q() + u
                    if score > best_score:
                        best_score = score
                        best_action = (r, c)
                if best_action is None:
                    break
                node = node.children[best_action]
                if node.board.done:
                    break
            if node.board.done:
                value = 1.0 if node.board.winner == node.board.current_player else -1.0
                # Backup from node (current player lost after opponent moved)
                n = node
                while n is not None:
                    n.N += 1
                    n.W += value
                    n = n.parent
                    value = -value
                continue
            legal = node.board.legal_actions()
            if not legal:
                continue
            prior, value = self._get_prior_and_value(node.board)
            for (r, c) in legal:
                b_child = node.board.copy()
                b_child.step(r, c)
                node.children[(r, c)] = MCTSNode(
                    b_child,
                    parent=node,
                    action=(r, c),
                    prior=prior.get((r, c), 1.0 / max(len(legal), 1)),
                )
            # Backup
            n = node
            while n is not None:
                n.N += 1
                n.W += value
                n = n.parent
                value = -value

        if not root.children:
            return legal[0] if (legal := root.board.legal_actions()) else (0, 0)
        best_action = max(root.children.items(), key=lambda t: t[1].N)[0]
        return best_action

    def predict(
        self,
        board_state: np.ndarray | GomokuBoard,
        blend_policy: float = 0.7,
        device: torch.device | None = None,
    ) -> tuple[int, int]:
        """
        MCTS move blended with raw policy: blend_policy * MCTS + (1-blend_policy) * policy_net.
        board_state: (3,9,9) or GomokuBoard.
        """
        if isinstance(board_state, np.ndarray):
            board = GomokuBoard()
            # Reconstruct 0/1/2 from channels [P1, P2, empty]
            board.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
            board.board[board_state[0] > 0.5] = 1
            board.board[board_state[1] > 0.5] = 2
            board.turn_count = int((board.board != 0).sum())
            board.current_player = 1 if (board.turn_count % 2 == 0) else 2
        else:
            board = board_state
        mcts_move = self.run(board)
        if blend_policy >= 1.0:
            return mcts_move
        row_p, col_p = self.policy_net.predict(
            board.to_state_tensor(),
            legal_actions=board.legal_actions(),
            deterministic=True,
            device=device or self.device,
        )
        return mcts_move if np.random.rand() < blend_policy else (row_p, col_p)
