"""Main training script: self-play -> PPO update -> metrics."""

import argparse
import random
import numpy as np
import torch

from gomoku_rl.config import (
    GAMES_PER_COLLECTION,
    BATCH_SIZE,
    LR,
    METRICS_WINDOW,
)
from gomoku_rl.env.board import GomokuBoard
from gomoku_rl.env.reward_wrapper import RewardWrapper
from gomoku_rl.models.policy_net import GomokuPolicyNet
from gomoku_rl.training.self_play import run_self_play
from gomoku_rl.training.train_step import train_step, compute_returns_and_advantages
from gomoku_rl.training.metrics import MetricsLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--games-per-iter", type=int, default=GAMES_PER_COLLECTION)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    policy_net = GomokuPolicyNet().to(device)
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=args.lr)
    reward_wrapper = RewardWrapper()
    metrics = MetricsLogger(window=METRICS_WINDOW)
    global_step = 0

    for iteration in range(args.iters):
        reward_wrapper.decay_for_step(global_step)
        trajectories, winners, turn_counts = run_self_play(
            policy_net,
            reward_wrapper,
            n_games=args.games_per_iter,
            device=device,
            deterministic=False,
        )
        for w, t in zip(winners, turn_counts):
            metrics.log_game(w, t)

        all_states = []
        all_actions = []
        all_old_log_probs = []
        all_returns = []
        all_advantages = []
        all_legal_masks = []
        for traj in trajectories:
            T = len(traj["rewards"])
            returns, advantages = compute_returns_and_advantages(
                traj["rewards"].tolist(),
                traj["values"].tolist(),
                traj["dones"].tolist(),
                last_value=0.0,
                last_done=True,
            )
            all_states.append(traj["states"])
            all_actions.append(traj["actions"])
            all_old_log_probs.append(traj["log_probs"])
            all_returns.append(returns)
            all_advantages.append(advantages)
            all_legal_masks.append(traj["legal_masks"])

        states = torch.from_numpy(np.concatenate(all_states, axis=0)).float()
        actions = torch.from_numpy(np.concatenate(all_actions, axis=0)).long()
        old_log_probs = torch.from_numpy(np.concatenate(all_old_log_probs, axis=0)).float()
        returns = torch.cat(all_returns, dim=0)
        advantages = torch.cat(all_advantages, dim=0)
        legal_masks = torch.from_numpy(np.concatenate(all_legal_masks, axis=0)).bool()

        n_samples = states.shape[0]
        perm = torch.randperm(n_samples, device=device)
        loss_info = None
        for start in range(0, n_samples, args.batch_size):
            end = min(start + args.batch_size, n_samples)
            idx = perm[start:end]
            loss_info = train_step(
                policy_net,
                optimizer,
                states[idx],
                actions[idx],
                old_log_probs[idx],
                returns[idx],
                advantages[idx],
                legal_masks[idx],
                device=device,
            )
        global_step += n_samples

        if (iteration + 1) % args.log_every == 0:
            summary = metrics.summary()
            loss_str = f"policy={loss_info['policy_loss']:.4f} value={loss_info['value_loss']:.4f} entropy={loss_info['entropy']:.4f}" if loss_info else ""
            print(
                f"Iter {iteration+1} | "
                f"WinP1={summary['win_ratio_p1']:.2f} WinP2={summary['win_ratio_p2']:.2f} "
                f"Turns={summary['turns_per_game_mean']:.1f}±{summary['turns_per_game_std']:.1f} | "
                f"{loss_str}"
            )
        if (iteration + 1) % args.save_every == 0:
            path = f"checkpoint_iter_{iteration+1}.pt"
            torch.save({"policy_net": policy_net.state_dict(), "optimizer": optimizer.state_dict(), "step": global_step}, path)
            print(f"Saved {path}")

    print("Training complete.")
    return policy_net


if __name__ == "__main__":
    main()
