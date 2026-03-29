#!/usr/bin/env python3
"""
Batch Checkpoint Evaluation Script

Evaluates all checkpoints in a directory against Random and Pure MCTS agents,
optionally computes Elo ratings between checkpoints via round-robin.

Usage:
    python scripts/evaluate_checkpoints.py --checkpoint-dir ./checkpoints
    python scripts/evaluate_checkpoints.py --checkpoint-dir ./checkpoints --elo
    python scripts/evaluate_checkpoints.py --checkpoint-dir ./checkpoints --skip-mcts --games 50
    python scripts/evaluate_checkpoints.py --checkpoint-dir ./checkpoints --elo --elo-checkpoints 10 50 100 200
"""

import argparse
import sys
import os
import json
import glob
import time
import math
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.board import Board
from src.alphazero.agent import AlphaZeroAgent
from src.agent import RL_Agent
from src.pure_mcts_agent import PureMCTSAgent
from src.alphazero.utils import check_terminal
from src.alphazero.config import AlphaZeroConfig


def find_checkpoints(checkpoint_dir: str) -> List[Tuple[int, str]]:
    """Find all checkpoint files and return (iteration, path) tuples sorted by iteration."""
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.pt"))
    checkpoints = []
    for path in checkpoint_files:
        basename = os.path.basename(path)
        try:
            iteration = int(basename.replace("checkpoint_", "").replace(".pt", ""))
            checkpoints.append((iteration, path))
        except ValueError:
            continue
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


def play_game(agent1, agent2, board_size: int = 9) -> int:
    """Play one game between two agents. Returns winner (1, 2, or 0 for draw)."""
    board = Board(board_size)
    agents = [agent1, agent2]
    current_player = 0
    move_count = 0

    while not board.check_win() and move_count < board_size ** 2:
        move = agents[current_player].command(board, 0)
        if board.check_valid(move):
            board.update(move, current_player + 1)
        else:
            return 2 if current_player == 0 else 1
        current_player = 1 - current_player
        move_count += 1

    winner = check_terminal(board.base)
    return winner if winner is not None else 0


def evaluate_vs_opponent(checkpoint_path: str, opponent, num_games: int,
                         num_simulations: int, batch_size: int,
                         board_size: int, config) -> Dict:
    """
    Evaluate a checkpoint against any opponent agent.
    Alternates starting player for fairness.
    """
    agent = AlphaZeroAgent(
        checkpoint_path=checkpoint_path,
        num_simulations=num_simulations,
        temperature=0.0,
        batch_size=batch_size,
        num_res_blocks=config.num_res_blocks,
        num_filters=config.num_filters,
        board_size=board_size
    )

    wins = 0
    losses = 0
    draws = 0

    for i in range(num_games):
        if i % 2 == 0:
            winner = play_game(agent, opponent, board_size=board_size)
            if winner == 1:
                wins += 1
            elif winner == 2:
                losses += 1
            else:
                draws += 1
        else:
            winner = play_game(opponent, agent, board_size=board_size)
            if winner == 2:
                wins += 1
            elif winner == 1:
                losses += 1
            else:
                draws += 1

    win_rate = (wins / num_games) * 100
    return {
        'wins': wins,
        'losses': losses,
        'draws': draws,
        'win_rate': round(win_rate, 1),
        'loss_rate': round((losses / num_games) * 100, 1),
        'draw_rate': round((draws / num_games) * 100, 1),
    }


def subsample_checkpoints(checkpoints: List[Tuple[int, str]], max_count: int) -> List[Tuple[int, str]]:
    """
    Subsample checkpoints evenly spaced by iteration number.
    Accounts for non-uniform checkpoint frequency (dense early, sparse late).
    """
    if len(checkpoints) <= max_count:
        return checkpoints

    iterations = [cp[0] for cp in checkpoints]
    min_iter, max_iter = iterations[0], iterations[-1]

    # Generate evenly spaced target iterations
    targets = np.linspace(min_iter, max_iter, max_count)

    selected = []
    used_indices = set()
    for target in targets:
        # Find closest checkpoint to this target
        best_idx = min(range(len(iterations)), key=lambda i: abs(iterations[i] - target))
        if best_idx not in used_indices:
            used_indices.add(best_idx)
            selected.append(checkpoints[best_idx])

    selected.sort(key=lambda x: x[0])
    return selected


def compute_elo_ratings(checkpoints: List[Tuple[int, str]], num_games: int,
                        num_simulations: int, batch_size: int,
                        board_size: int, config) -> Dict[int, float]:
    """
    Compute Elo ratings via round-robin between checkpoints.
    Returns dict mapping iteration -> Elo rating.
    """
    K = 32
    ratings = {iteration: 1000.0 for iteration, _ in checkpoints}
    total_matchups = len(checkpoints) * (len(checkpoints) - 1) // 2
    matchup_num = 0

    for i in range(len(checkpoints)):
        for j in range(i + 1, len(checkpoints)):
            iter_a, path_a = checkpoints[i]
            iter_b, path_b = checkpoints[j]
            matchup_num += 1

            print(f"  Elo matchup {matchup_num}/{total_matchups}: "
                  f"iter {iter_a} vs iter {iter_b}", end="")

            agent_a = AlphaZeroAgent(
                checkpoint_path=path_a,
                num_simulations=num_simulations,
                temperature=0.0,
                batch_size=batch_size,
                num_res_blocks=config.num_res_blocks,
                num_filters=config.num_filters,
                board_size=board_size
            )
            agent_b = AlphaZeroAgent(
                checkpoint_path=path_b,
                num_simulations=num_simulations,
                temperature=0.0,
                batch_size=batch_size,
                num_res_blocks=config.num_res_blocks,
                num_filters=config.num_filters,
                board_size=board_size
            )

            wins_a = 0
            wins_b = 0
            draws = 0

            for g in range(num_games):
                if g % 2 == 0:
                    winner = play_game(agent_a, agent_b, board_size=board_size)
                    if winner == 1:
                        wins_a += 1
                    elif winner == 2:
                        wins_b += 1
                    else:
                        draws += 1
                else:
                    winner = play_game(agent_b, agent_a, board_size=board_size)
                    if winner == 2:
                        wins_a += 1
                    elif winner == 1:
                        wins_b += 1
                    else:
                        draws += 1

            # Elo update
            score_a = (wins_a + 0.5 * draws) / num_games
            score_b = (wins_b + 0.5 * draws) / num_games
            expected_a = 1.0 / (1.0 + 10.0 ** ((ratings[iter_b] - ratings[iter_a]) / 400.0))
            expected_b = 1.0 - expected_a

            ratings[iter_a] += K * (score_a - expected_a)
            ratings[iter_b] += K * (score_b - expected_b)

            print(f" -> {wins_a}W/{wins_b}L/{draws}D "
                  f"(Elo: {ratings[iter_a]:.0f} / {ratings[iter_b]:.0f})")

    return {k: round(v, 1) for k, v in ratings.items()}


def main():
    parser = argparse.ArgumentParser(description='Batch checkpoint evaluation')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                        help='Checkpoint directory (default: ./checkpoints)')
    parser.add_argument('--checkpoints', type=int, nargs='+', default=None,
                        help='Specific checkpoint iterations to evaluate (default: all)')
    parser.add_argument('--games', type=int, default=20,
                        help='Number of games per evaluation (default: 20)')
    parser.add_argument('--simulations', type=int, default=100,
                        help='MCTS simulations for AlphaZero agents (default: 100)')
    parser.add_argument('--mcts-simulations', type=int, default=200,
                        help='Simulations for Pure MCTS opponent (default: 200)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='MCTS batch size (default: 16)')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Configuration file (default: config.yaml)')
    parser.add_argument('--output', type=str, default='./logs/checkpoint_evaluation.json',
                        help='Output JSON path (default: ./logs/checkpoint_evaluation.json)')
    parser.add_argument('--skip-random', action='store_true',
                        help='Skip evaluation against random agent')
    parser.add_argument('--skip-mcts', action='store_true',
                        help='Skip evaluation against Pure MCTS agent')
    parser.add_argument('--elo', action='store_true',
                        help='Compute Elo ratings between checkpoints via round-robin')
    parser.add_argument('--elo-checkpoints', type=int, nargs='+', default=None,
                        help='Specific iterations for Elo (default: auto-subsample up to 10)')
    parser.add_argument('--elo-games', type=int, default=20,
                        help='Games per Elo matchup (default: 20)')
    args = parser.parse_args()

    config = AlphaZeroConfig.from_yaml(args.config)

    print("\n" + "=" * 60)
    print("Batch Checkpoint Evaluation")
    print("=" * 60)

    # Discover checkpoints
    all_checkpoints = find_checkpoints(args.checkpoint_dir)
    if not all_checkpoints:
        print(f"\nNo checkpoints found in {args.checkpoint_dir}")
        return

    # Filter to requested checkpoints
    if args.checkpoints:
        checkpoint_map = {it: path for it, path in all_checkpoints}
        checkpoints_to_eval = []
        for it in args.checkpoints:
            if it in checkpoint_map:
                checkpoints_to_eval.append((it, checkpoint_map[it]))
            else:
                print(f"Warning: Checkpoint for iteration {it} not found")
        checkpoints_to_eval.sort(key=lambda x: x[0])
    else:
        checkpoints_to_eval = all_checkpoints

    if not checkpoints_to_eval:
        print("No valid checkpoints to evaluate")
        return

    print(f"\nFound {len(checkpoints_to_eval)} checkpoints to evaluate")
    print(f"Iterations: {[cp[0] for cp in checkpoints_to_eval]}")
    print(f"Board size: {config.board_size}")

    results = {
        'metadata': {
            'checkpoint_dir': args.checkpoint_dir,
            'num_games': args.games,
            'simulations': args.simulations,
            'mcts_simulations': args.mcts_simulations,
            'board_size': config.board_size,
            'timestamp': datetime.now().isoformat(),
        },
        'vs_random': [],
        'vs_mcts': [],
    }

    # Phase 1: vs Random
    if not args.skip_random:
        print(f"\n{'=' * 60}")
        print(f"Phase 1: Evaluating {len(checkpoints_to_eval)} checkpoints vs Random")
        print(f"{'=' * 60}")
        random_agent = RL_Agent()

        for iteration, checkpoint_path in checkpoints_to_eval:
            print(f"\n  Iteration {iteration}: ", end="", flush=True)
            start = time.time()
            result = evaluate_vs_opponent(
                checkpoint_path, random_agent, args.games,
                args.simulations, args.batch_size, config.board_size, config
            )
            elapsed = time.time() - start
            result['iteration'] = iteration
            results['vs_random'].append(result)
            print(f"{result['wins']}W/{result['losses']}L/{result['draws']}D "
                  f"({result['win_rate']}%) [{elapsed:.1f}s]")

    # Phase 2: vs Pure MCTS
    if not args.skip_mcts:
        print(f"\n{'=' * 60}")
        print(f"Phase 2: Evaluating {len(checkpoints_to_eval)} checkpoints vs Pure MCTS "
              f"(sims={args.mcts_simulations})")
        print(f"{'=' * 60}")
        mcts_agent = PureMCTSAgent(num_simulations=args.mcts_simulations)

        for iteration, checkpoint_path in checkpoints_to_eval:
            print(f"\n  Iteration {iteration}: ", end="", flush=True)
            start = time.time()
            result = evaluate_vs_opponent(
                checkpoint_path, mcts_agent, args.games,
                args.simulations, args.batch_size, config.board_size, config
            )
            elapsed = time.time() - start
            result['iteration'] = iteration
            results['vs_mcts'].append(result)
            print(f"{result['wins']}W/{result['losses']}L/{result['draws']}D "
                  f"({result['win_rate']}%) [{elapsed:.1f}s]")

    # Phase 3: Elo ratings
    if args.elo:
        print(f"\n{'=' * 60}")
        print("Phase 3: Elo Rating Computation")
        print(f"{'=' * 60}")

        if args.elo_checkpoints:
            checkpoint_map = {it: path for it, path in all_checkpoints}
            elo_checkpoints = []
            for it in args.elo_checkpoints:
                if it in checkpoint_map:
                    elo_checkpoints.append((it, checkpoint_map[it]))
                else:
                    print(f"Warning: Checkpoint for iteration {it} not found")
            elo_checkpoints.sort(key=lambda x: x[0])
        else:
            elo_checkpoints = subsample_checkpoints(checkpoints_to_eval, max_count=10)

        n = len(elo_checkpoints)
        total_matchups = n * (n - 1) // 2
        total_games = total_matchups * args.elo_games

        print(f"\n  Checkpoints: {[cp[0] for cp in elo_checkpoints]}")
        print(f"  Matchups: {total_matchups}, Games per matchup: {args.elo_games}, "
              f"Total games: {total_games}")

        if n < 2:
            print("  Need at least 2 checkpoints for Elo computation")
        else:
            start = time.time()
            elo_ratings = compute_elo_ratings(
                elo_checkpoints, args.elo_games,
                args.simulations, args.batch_size, config.board_size, config
            )
            elapsed = time.time() - start

            results['elo_ratings'] = {str(k): v for k, v in elo_ratings.items()}

            print(f"\n  Elo computation completed in {elapsed:.1f}s")
            print(f"\n  {'Iteration':<12} {'Elo Rating':<12}")
            print(f"  {'-' * 24}")
            for iteration in sorted(elo_ratings.keys()):
                print(f"  {iteration:<12} {elo_ratings[iteration]:<12.1f}")

    # Save results
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output}")

    # Print summary
    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")

    if results['vs_random']:
        print(f"\n  vs Random:")
        for r in results['vs_random']:
            print(f"    Iter {r['iteration']:<6} {r['win_rate']:>5.1f}% win rate")

    if results['vs_mcts']:
        print(f"\n  vs Pure MCTS (sims={args.mcts_simulations}):")
        for r in results['vs_mcts']:
            print(f"    Iter {r['iteration']:<6} {r['win_rate']:>5.1f}% win rate")

    print()


if __name__ == '__main__':
    main()
