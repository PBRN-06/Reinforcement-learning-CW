"""Self-play: Model_v1 vs Model_v1 to generate training trajectories."""

import torch
import numpy as np
from gomoku_rl.env.board import GomokuBoard
from gomoku_rl.env.reward_wrapper import RewardWrapper
from gomoku_rl.models.policy_net import GomokuPolicyNet
from gomoku_rl.config import BOARD_SIZE, ACTION_SIZE


def run_self_play(
    policy_net: GomokuPolicyNet,
    reward_wrapper: RewardWrapper,
    n_games: int = 1,
    device: torch.device | None = None,
    deterministic: bool = False,
) -> tuple[list[dict], list[int], list[int]]:
    """
    Run n_games of self-play (same policy for both players).
    Returns: list of trajectory dicts (one per game), list of winners, list of turn counts.
    Each trajectory dict has: states, actions, rewards, dones, log_probs, values, legal_masks.
    """
    if device is None:
        device = next(policy_net.parameters()).device
    policy_net.eval()
    all_trajectories = []
    winners = []
    turn_counts = []

    for _ in range(n_games):
        board = GomokuBoard()
        board.reset()
        reward_wrapper.set_decay_factor(reward_wrapper._decay_factor)
        states = []
        actions = []
        rewards = []
        dones = []
        log_probs = []
        values = []
        legal_masks_list = []

        while not board.done:
            state = board.to_state_tensor()
            legal = board.legal_actions()
            if not legal:
                break
            legal_indices = np.array([r * BOARD_SIZE + c for r, c in legal], dtype=np.int64)
            x = torch.from_numpy(state).float().unsqueeze(0).to(device)
            legal_mask = torch.zeros(1, ACTION_SIZE, dtype=torch.bool, device=device)
            legal_mask[0, legal_indices] = True

            with torch.no_grad():
                action_idx, log_prob, value, _ = policy_net.get_action_and_log_prob(
                    x, legal_mask, deterministic=deterministic
                )
            action_idx = action_idx.item()
            log_prob = log_prob.item()
            value = value.item()
            row = action_idx // BOARD_SIZE
            col = action_idx % BOARD_SIZE

            states.append(state)
            actions.append(action_idx)
            log_probs.append(log_prob)
            values.append(value)
            legal_masks_list.append(legal_mask.cpu().numpy())

            state, reward, done, info = reward_wrapper.step(board, row, col)
            rewards.append(reward)
            dones.append(done)

        # Pad rewards/values for return computation: last step gets bootstrap value 0 if not done
        turn_counts.append(len(states))
        winners.append(board.winner if board.winner is not None else 0)

        traj = {
            "states": np.stack(states),
            "actions": np.array(actions, dtype=np.int64),
            "rewards": np.array(rewards, dtype=np.float32),
            "dones": np.array(dones, dtype=np.float32),
            "log_probs": np.array(log_probs, dtype=np.float32),
            "values": np.array(values, dtype=np.float32),
            "legal_masks": np.concatenate(legal_masks_list, axis=0),
        }
        all_trajectories.append(traj)

    return all_trajectories, winners, turn_counts
