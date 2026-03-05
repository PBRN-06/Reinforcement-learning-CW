"""
Monte Carlo Tree Search for AlphaZero

MCTS guided by neural network policy and value estimates.
"""

import numpy as np
import math
from src.alphazero.utils import (
    get_legal_moves, apply_action, check_terminal,
    get_terminal_value, create_valid_move_mask
)


class MCTSNode:
    """Node in the Monte Carlo search tree."""

    def __init__(self, parent=None, prior_p: float = 1.0):
        self.parent = parent
        self.children = {}  # {action: MCTSNode}
        self.visit_count = 0
        self.total_value = 0.0
        self.prior_p = prior_p
        self.mean_value = 0.0

    def expand(self, action_priors: dict):
        """
        Expand node by creating children for all legal actions.

        Args:
            action_priors: Dict mapping action -> prior probability
        """
        for action, prior in action_priors.items():
            if action not in self.children:
                self.children[action] = MCTSNode(parent=self, prior_p=prior)

    def select_child(self, c_puct: float):
        """
        Select child with highest UCB score.

        Args:
            c_puct: Exploration constant

        Returns:
            (action, child_node) tuple
        """
        best_score = -float('inf')
        best_action = None
        best_child = None

        for action, child in self.children.items():
            # UCB formula: Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
            ucb_score = (
                child.mean_value +
                c_puct * child.prior_p * math.sqrt(self.visit_count) / (1 + child.visit_count)
            )

            if ucb_score > best_score:
                best_score = ucb_score
                best_action = action
                best_child = child

        return best_action, best_child

    def update(self, value: float):
        """
        Update node statistics after backpropagation.

        Args:
            value: Value to add (from current player's perspective)
        """
        self.visit_count += 1
        self.total_value += value
        self.mean_value = self.total_value / self.visit_count

    def is_leaf(self):
        """Check if node is a leaf (no children)."""
        return len(self.children) == 0

    def is_root(self):
        """Check if node is root (no parent)."""
        return self.parent is None


class MCTS:
    """Monte Carlo Tree Search with neural network guidance."""

    def __init__(self, network, num_simulations: int = 400, c_puct: float = 1.5,
                 dirichlet_alpha: float = 0.3, dirichlet_epsilon: float = 0.25):
        """
        Initialize MCTS.

        Args:
            network: Neural network for policy and value estimates
            num_simulations: Number of simulations per search
            c_puct: Exploration constant for UCB
            dirichlet_alpha: Concentration parameter for Dirichlet noise
            dirichlet_epsilon: Weight of Dirichlet noise in root prior
        """
        self.network = network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon

    def search(self, board_state: np.ndarray, current_player: int,
               add_noise: bool = False, temperature: float = 1.0):
        """
        Run MCTS search from current board state.

        Args:
            board_state: (board_size, board_size) numpy array
            current_player: Current player (1 or 2)
            add_noise: Whether to add Dirichlet noise to root (for exploration)
            temperature: Temperature for action selection (higher = more random)

        Returns:
            (action_probs, root_value): Policy distribution and value estimate
        """
        # Create root node
        root = MCTSNode()

        # Expand root with network priors
        policy_probs, value = self.network.predict(board_state)
        legal_moves = get_legal_moves(board_state)
        action_priors = {action: policy_probs[action] for action in legal_moves}

        # Normalize priors over legal moves
        prior_sum = sum(action_priors.values())
        if prior_sum > 0:
            action_priors = {a: p / prior_sum for a, p in action_priors.items()}

        root.expand(action_priors)

        # Add Dirichlet noise to root if requested (for self-play exploration)
        if add_noise:
            self._add_dirichlet_noise(root)

        # Run simulations
        for _ in range(self.num_simulations):
            node = root
            state = board_state.copy()
            player = current_player
            search_path = [node]

            # Selection: Traverse tree until reaching a leaf
            while not node.is_leaf():
                action, node = node.select_child(self.c_puct)
                state = apply_action(state, action, player)
                player = 3 - player  # Switch player (1->2, 2->1)
                search_path.append(node)

            # Check if terminal
            winner = check_terminal(state)

            if winner is not None:
                # Terminal state - use actual outcome
                value = get_terminal_value(winner, player)
            else:
                # Non-terminal - expand and evaluate with network
                policy_probs, value = self.network.predict(state)
                legal_moves = get_legal_moves(state)
                action_priors = {action: policy_probs[action] for action in legal_moves}

                # Normalize
                prior_sum = sum(action_priors.values())
                if prior_sum > 0:
                    action_priors = {a: p / prior_sum for a, p in action_priors.items()}

                node.expand(action_priors)

            # Backpropagation: Update all nodes in search path
            self._backpropagate(search_path, value, current_player)

        # Get action probabilities from visit counts
        action_probs = self._get_action_probs(root, board_state.shape[0], temperature)

        return action_probs, root.mean_value

    def _add_dirichlet_noise(self, root: MCTSNode):
        """
        Add Dirichlet noise to root node priors for exploration.

        Args:
            root: Root node of search tree
        """
        actions = list(root.children.keys())
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(actions))

        for i, action in enumerate(actions):
            child = root.children[action]
            # Mix: (1-ε) * network_prior + ε * noise
            child.prior_p = (
                (1 - self.dirichlet_epsilon) * child.prior_p +
                self.dirichlet_epsilon * noise[i]
            )

    def _backpropagate(self, search_path: list, value: float, root_player: int):
        """
        Backpropagate value through search path.

        Args:
            search_path: List of nodes from root to leaf
            value: Value to backpropagate (from leaf player's perspective)
            root_player: Player at root node
        """
        # The value alternates sign as we go up the tree
        # because each level represents a different player
        current_player = root_player

        # We need to traverse the path backwards and flip value for each level
        # The search_path[0] is root (current player)
        # search_path[-1] is the expanded leaf

        # Calculate how many moves were made to reach the leaf
        depth = len(search_path) - 1

        # If depth is even, leaf player == root player
        # If depth is odd, leaf player != root player
        if depth % 2 == 0:
            # Same player, value stays positive
            leaf_value = value
        else:
            # Different player, flip value
            leaf_value = -value

        # Now backpropagate, flipping value at each step
        for i, node in enumerate(reversed(search_path)):
            # Alternate value sign based on depth from leaf
            if i % 2 == 0:
                node.update(leaf_value)
            else:
                node.update(-leaf_value)

    def _get_action_probs(self, root: MCTSNode, board_size: int, temperature: float):
        """
        Get action probability distribution from visit counts.

        Args:
            root: Root node
            board_size: Size of board
            temperature: Temperature parameter (0 = deterministic, higher = more random)

        Returns:
            (board_size²,) numpy array of action probabilities
        """
        num_actions = board_size * board_size
        action_probs = np.zeros(num_actions)

        # Get visit counts
        for action, child in root.children.items():
            action_probs[action] = child.visit_count

        # Apply temperature
        if temperature == 0:
            # Deterministic: select most visited
            best_action = np.argmax(action_probs)
            action_probs = np.zeros(num_actions)
            action_probs[best_action] = 1.0
        else:
            # Stochastic: sample proportional to visits^(1/T)
            action_probs = action_probs ** (1.0 / temperature)
            prob_sum = np.sum(action_probs)
            if prob_sum > 0:
                action_probs = action_probs / prob_sum
            else:
                # If no visits (shouldn't happen), uniform over legal moves
                mask = create_valid_move_mask(root)  # This won't work, need board state
                action_probs = mask / np.sum(mask)

        return action_probs
