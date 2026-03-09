"""
Batched Monte Carlo Tree Search for AlphaZero

MCTS with batched neural network evaluations for GPU efficiency.
Collects multiple leaf nodes and evaluates them in a single batch.
"""

import numpy as np
import math
from src.alphazero.mcts import MCTSNode
from src.alphazero.utils import (
    get_legal_moves, apply_action, check_terminal,
    get_terminal_value, create_valid_move_mask
)


class BatchedMCTS:
    """
    Monte Carlo Tree Search with batched neural network evaluations.

    Evaluates multiple leaf nodes simultaneously for better GPU utilization.
    """

    def __init__(self, network, num_simulations: int = 400, batch_size: int = 16,
                 c_puct: float = 1.5, dirichlet_alpha: float = 0.3,
                 dirichlet_epsilon: float = 0.25):
        """
        Initialize Batched MCTS.

        Args:
            network: Neural network for policy and value estimates
            num_simulations: Number of simulations per search
            batch_size: Number of simulations to batch together
            c_puct: Exploration constant for UCB
            dirichlet_alpha: Concentration parameter for Dirichlet noise
            dirichlet_epsilon: Weight of Dirichlet noise in root prior
        """
        self.network = network
        self.num_simulations = num_simulations
        self.batch_size = batch_size
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon

    def search(self, board_state: np.ndarray, current_player: int,
               add_noise: bool = False, temperature: float = 1.0):
        """
        Run MCTS search with batched evaluations.

        Args:
            board_state: (board_size, board_size) numpy array
            current_player: Current player (1 or 2)
            add_noise: Whether to add Dirichlet noise to root
            temperature: Temperature for action selection

        Returns:
            (action_probs, root_value): Policy distribution and value estimate
        """
        # Create root node
        root = MCTSNode()

        # Expand root with network priors
        policy_probs, value = self.network.predict(board_state)
        legal_moves = get_legal_moves(board_state)
        action_priors = {action: policy_probs[action] for action in legal_moves}

        # Normalize priors
        prior_sum = sum(action_priors.values())
        if prior_sum > 0:
            action_priors = {a: p / prior_sum for a, p in action_priors.items()}

        root.expand(action_priors)

        # Add Dirichlet noise if requested
        if add_noise:
            self._add_dirichlet_noise(root)

        # Run simulations in batches
        num_batches = (self.num_simulations + self.batch_size - 1) // self.batch_size

        for batch_idx in range(num_batches):
            batch_size_actual = min(self.batch_size,
                                   self.num_simulations - batch_idx * self.batch_size)

            # Collect batch of leaf nodes
            leaf_data = []
            
            for _ in range(batch_size_actual):
                node = root
                state = board_state.copy()
                player = current_player
                search_path = [node]
                
                # Apply virtual loss to root immediately
                root.virtual_loss += 1
                root.visit_count += 1

                # Selection: Traverse to leaf
                while not node.is_leaf():
                    action, node = node.select_child(self.c_puct)
                    assert node is not None and action is not None
                    
                    # Apply virtual loss immediately so next simulation sees it
                    node.virtual_loss += 1
                    node.visit_count += 1
                    
                    state = apply_action(state, action, player)
                    player = 3 - player
                    search_path.append(node)

                # Check if terminal
                winner = check_terminal(state)

                leaf_data.append({
                    'search_path': search_path,
                    'state': state,
                    'player': player,
                    'node': node,
                    'winner': winner
                })

            # Separate terminal and non-terminal leaves
            terminal_leaves = [d for d in leaf_data if d['winner'] is not None]
            non_terminal_leaves = [d for d in leaf_data if d['winner'] is None]

            # Batch evaluate non-terminal leaves
            if non_terminal_leaves:
                states = [d['state'] for d in non_terminal_leaves]
                policies_batch, values_batch = self.network.predict_batch(states)

                # Expand and store values for non-terminal leaves
                for i, data in enumerate(non_terminal_leaves):
                    policy_probs = policies_batch[i]
                    value = values_batch[i]
                    
                    legal_moves = get_legal_moves(data['state'])
                    action_priors = {action: policy_probs[action] for action in legal_moves}
                    prior_sum = sum(action_priors.values())
                    if prior_sum > 0:
                        action_priors = {a: p / prior_sum for a, p in action_priors.items()}
                    
                    data['node'].expand(action_priors)
                    data['value'] = value  # already from data['player']'s perspective

            # For terminal leaves
            for data in terminal_leaves:
                data['value'] = get_terminal_value(data['winner'], data['player'])
                # also already from data['player']'s perspective

            # Backpropagate
            for data in leaf_data:
                self._backpropagate(data['search_path'], data['value'])

        # Get action probabilities from visit counts
        action_probs = self._get_action_probs(root, board_state.shape[0], temperature)

        return action_probs, root.mean_value

    def _add_dirichlet_noise(self, root: MCTSNode):
        """Add Dirichlet noise to root node priors."""
        actions = list(root.children.keys())
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(actions))

        for i, action in enumerate(actions):
            child = root.children[action]
            child.prior_p = (
                (1 - self.dirichlet_epsilon) * child.prior_p +
                self.dirichlet_epsilon * noise[i]
            )

    def _backpropagate(self, search_path: list, value: float):
        """
        Backpropagate value through search path.
        Value should be from the perspective of the player at the leaf node.
        """
        for node in reversed(search_path):
            node.visit_count -= 1        # remove temporary inflation
            node.virtual_loss -= 1
            node.update(value)
            value = -value  # parent has opposite perspective

    def _get_action_probs(self, root: MCTSNode, board_size: int, temperature: float):
        """Get action probability distribution from visit counts."""
        num_actions = board_size * board_size
        action_probs = np.zeros(num_actions)

        # Get visit counts
        for action, child in root.children.items():
            action_probs[action] = child.visit_count

        # Apply temperature
        if temperature == 0:
            # Deterministic
            best_action = np.argmax(action_probs)
            action_probs = np.zeros(num_actions)
            action_probs[best_action] = 1.0
        else:
            # Stochastic
            action_probs = action_probs ** (1.0 / temperature)
            prob_sum = np.sum(action_probs)
            if prob_sum > 0:
                action_probs = action_probs / prob_sum
            else:
                # Fallback: uniform over visited actions
                visited = np.array([child.visit_count for child in root.children.values()])
                if visited.sum() > 0:
                    for action, child in root.children.items():
                        action_probs[action] = 1.0 / len(root.children)

        return action_probs
