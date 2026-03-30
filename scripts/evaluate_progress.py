#!/usr/bin/env python3
"""
AlphaZero Training Progress Evaluation Script

Evaluates training progress by:
1. Plotting loss curves from training logs
2. Evaluating checkpoints against random agent
3. Showing win rate progression over iterations
4. Generating a comprehensive report

Usage:
    python scripts/evaluate_progress.py
    python scripts/evaluate_progress.py --checkpoints 10 20 30 40 50
    python scripts/evaluate_progress.py --games 50 --simulations 200
    python scripts/evaluate_progress.py --plot-only  # Just show loss curves
"""

import argparse
import sys
import os
import json
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from typing import List, Dict

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.board import Board
from src.alphazero.agent import AlphaZeroAgent
from src.agent import RL_Agent
from src.alphazero.utils import check_terminal
from src.alphazero.config import AlphaZeroConfig


def load_training_metrics(log_path: str = "./logs/training_metrics.json") -> List[Dict]:
    """
    Load training metrics from JSON log file.

    Args:
        log_path: Path to training metrics JSON file

    Returns:
        List of metric dictionaries
    """
    if not os.path.exists(log_path):
        print(f"Warning: Training metrics not found at {log_path}")
        return []

    with open(log_path, 'r') as f:
        metrics = json.load(f)

    return metrics


def plot_loss_curves(metrics: List[Dict], output_path: str = "./logs/loss_curves.png"):
    """
    Plot training loss curves.

    Args:
        metrics: List of training metrics
        output_path: Path to save plot
    """
    if not metrics:
        print("No metrics to plot")
        return

    iterations = [m['iteration'] for m in metrics]
    policy_loss = [m['policy_loss'] for m in metrics]
    value_loss = [m['value_loss'] for m in metrics]
    total_loss = [m['total_loss'] for m in metrics]
    d_entropy = [m['target_entropy'] - m['network_entropy'] for m in metrics]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Policy loss
    axes[0][0].plot(iterations, policy_loss, linewidth=2, color='#2196F3')
    axes[0][0].set_xlabel('Iteration')
    axes[0][0].set_ylabel('Policy Loss')
    axes[0][0].set_title('Policy Loss Over Training')
    axes[0][0].grid(True, alpha=0.3)

    # Value loss
    axes[0][1].plot(iterations, value_loss, linewidth=2, color='#4CAF50')
    axes[0][1].set_xlabel('Iteration')
    axes[0][1].set_ylabel('Value Loss')
    axes[0][1].set_title('Value Loss Over Training')
    axes[0][1].grid(True, alpha=0.3)

    # Total loss
    axes[1][0].plot(iterations, total_loss, linewidth=2, color='#FF5722')
    axes[1][0].set_xlabel('Iteration')
    axes[1][0].set_ylabel('Total Loss')
    axes[1][0].set_title('Total Loss Over Training')
    axes[1][0].grid(True, alpha=0.3)

    # Delta entropy
    axes[1][1].plot(iterations, d_entropy, linewidth=2)
    axes[1][1].set_xlabel('Iteration')
    axes[1][1].set_ylabel('Delta Entropy')
    axes[1][1].set_title('Delta Entropy Over Training')
    axes[1][1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nLoss curves saved to: {output_path}")


def evaluate_checkpoint_vs_checkpoint(checkpoint1: str, checkpoint2: str,
                                       num_games: int = 20, num_simulations: int = 100,
                                       verbose: bool = True, batch_size: int = 16,
                                       board_size: int = 9) -> Dict:
    """
    Evaluate two checkpoints against each other.

    Args:
        checkpoint1: Path to first checkpoint (the one being tested)
        checkpoint2: Path to second checkpoint (baseline)
        num_games: Number of games to play
        num_simulations: MCTS simulations per move
        verbose: Show progress per game
        batch_size: MCTS batch size for GPU optimization

    Returns:
        Dict with win/loss/draw statistics (from checkpoint1's perspective)
    """
    # Load agents
    agent1 = AlphaZeroAgent(checkpoint1, num_simulations=num_simulations,
                           temperature=0.0, batch_size=batch_size)
    agent2 = AlphaZeroAgent(checkpoint2, num_simulations=num_simulations,
                           temperature=0.0, batch_size=batch_size)

    wins = 0
    losses = 0
    draws = 0

    for i in range(num_games):
        # Alternate starting player
        if i % 2 == 0:
            # Agent1 plays first
            winner = play_game(agent1, agent2, board_size=board_size)
            if winner == 1:
                wins += 1
            elif winner == 2:
                losses += 1
            else:
                draws += 1
        else:
            # Agent2 plays first, agent1 is player 2
            winner = play_game(agent2, agent1, board_size=board_size)
            if winner == 2:
                wins += 1
            elif winner == 1:
                losses += 1
            else:
                draws += 1

        # Show progress
        if verbose and (i + 1) % 5 == 0:
            print(f"    Progress: {i + 1}/{num_games} games, current: {wins}W/{losses}L/{draws}D", end='\r')

    if verbose:
        print()  # Newline after progress

    win_rate = (wins / num_games) * 100
    loss_rate = (losses / num_games) * 100
    draw_rate = (draws / num_games) * 100

    return {
        'wins': wins,
        'losses': losses,
        'draws': draws,
        'win_rate': win_rate,
        'loss_rate': loss_rate,
        'draw_rate': draw_rate
    }


def evaluate_checkpoint_vs_random(checkpoint_path: str, num_games: int = 20,
                                   num_simulations: int = 100, verbose: bool = True,
                                   batch_size: int = 16, board_size: int = 9) -> Dict:
    """
    Evaluate a checkpoint against random agent.

    Args:
        checkpoint_path: Path to checkpoint file
        num_games: Number of games to play
        num_simulations: MCTS simulations per move
        verbose: Show progress per game
        batch_size: MCTS batch size for GPU optimization

    Returns:
        Dict with win/loss/draw statistics
    """
    # Load AlphaZero agent with batched MCTS
    agent = AlphaZeroAgent(
        checkpoint_path=checkpoint_path,
        num_simulations=num_simulations,
        temperature=0.0,  # Deterministic play
        batch_size=batch_size  # Enable GPU optimization
    )

    # Random agent
    random_agent = RL_Agent()

    wins = 0
    losses = 0
    draws = 0

    for i in range(num_games):
        # Alternate who plays first
        if i % 2 == 0:
            # AlphaZero plays first (Player 1)
            winner = play_game(agent, random_agent, board_size=board_size)
            if winner == 1:
                wins += 1
            elif winner == 2:
                losses += 1
            else:
                draws += 1
        else:
            # Random plays first (Player 1), AlphaZero is Player 2
            winner = play_game(random_agent, agent, board_size=board_size)
            if winner == 2:
                wins += 1
            elif winner == 1:
                losses += 1
            else:
                draws += 1

        # Show progress
        if verbose and (i + 1) % 5 == 0:
            print(f"    Progress: {i + 1}/{num_games} games, current: {wins}W/{losses}L/{draws}D", end='\r')

    if verbose:
        print()  # Newline after progress

    win_rate = (wins / num_games) * 100
    loss_rate = (losses / num_games) * 100
    draw_rate = (draws / num_games) * 100

    return {
        'wins': wins,
        'losses': losses,
        'draws': draws,
        'win_rate': win_rate,
        'loss_rate': loss_rate,
        'draw_rate': draw_rate
    }


def play_game(agent1, agent2, board_size: int = 9) -> int:
    """
    Play one game between two agents.

    Args:
        agent1: First agent (Player 1)
        agent2: Second agent (Player 2)
        board_size: Size of the board

    Returns:
        Winner (1, 2, or 0 for draw)
    """
    board = Board(board_size)
    agents = [agent1, agent2]
    current_player = 0
    move_count = 0

    while not board.check_win() and move_count < board_size ** 2:
        # Get move from current agent
        move = agents[current_player].command(board, 0)

        # Apply move
        if board.check_valid(move):
            board.update(move, current_player + 1)
        else:
            # Invalid move = instant loss
            return 2 if current_player == 0 else 1

        # Switch player
        current_player = 1 - current_player
        move_count += 1

    # Determine winner
    winner = check_terminal(board.base)
    return winner if winner is not None else 0


def find_checkpoints(checkpoint_dir: str = "./checkpoints") -> List[tuple]:
    """
    Find all checkpoint files and extract iteration numbers.

    Args:
        checkpoint_dir: Directory containing checkpoints

    Returns:
        List of (iteration, path) tuples, sorted by iteration
    """
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.pt"))

    checkpoints = []
    for path in checkpoint_files:
        basename = os.path.basename(path)
        # Extract iteration number from filename like "checkpoint_10.pt"
        try:
            iteration = int(basename.replace("checkpoint_", "").replace(".pt", ""))
            checkpoints.append((iteration, path))
        except ValueError:
            continue

    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


def plot_win_rate_progression(evaluation_results: List[Dict],
                               output_path: str = "./logs/win_rate_progression.png",
                               vs_random: bool = False):
    """
    Plot win rate progression over iterations.

    Args:
        evaluation_results: List of dicts with iteration and win_rate
        output_path: Path to save plot
        vs_random: Whether this is vs-random or vs-checkpoint
    """
    if not evaluation_results:
        print("No evaluation results to plot")
        return

    iterations = [r['iteration'] for r in evaluation_results]
    win_rates = [r['win_rate'] for r in evaluation_results]
    loss_rates = [r['loss_rate'] for r in evaluation_results]
    draw_rates = [r['draw_rate'] for r in evaluation_results]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Stacked area plot
    ax.fill_between(iterations, 0, win_rates, label='Wins', alpha=0.7, color='#4CAF50')
    ax.fill_between(iterations, win_rates, np.array(win_rates) + np.array(draw_rates),
                     label='Draws', alpha=0.7, color='#FFC107')
    ax.fill_between(iterations, np.array(win_rates) + np.array(draw_rates), 100,
                     label='Losses', alpha=0.7, color='#F44336')

    # Add win rate line
    ax.plot(iterations, win_rates, 'o-', color='#2E7D32', linewidth=2,
            markersize=8, label='Win Rate')

    # Add 50% baseline
    ax.axhline(y=50, color='black', linestyle='--', linewidth=1, alpha=0.5,
               label='50% Baseline')

    ax.set_xlabel('Training Iteration', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)

    if vs_random:
        baseline_iter = evaluation_results[0].get('baseline_iter', 'N/A')
        ax.set_title('AlphaZero vs Random Agent - Performance Progression',
                     fontsize=14, fontweight='bold')
    else:
        baseline_iter = evaluation_results[0].get('baseline_iter', 'N/A')
        ax.set_title(f'Checkpoint Performance vs Baseline (Iteration {baseline_iter})',
                     fontsize=14, fontweight='bold')

    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Win rate progression saved to: {output_path}")


def generate_report(metrics: List[Dict], evaluation_results: List[Dict],
                    output_path: str = "./logs/evaluation_report.txt",
                    vs_random: bool = False):
    """
    Generate a text report of training progress.

    Args:
        metrics: Training metrics
        evaluation_results: Evaluation results
        output_path: Path to save report
        vs_random: Whether evaluating vs random or vs checkpoint
    """
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("AlphaZero Training Progress Report\n")
        f.write("=" * 80 + "\n\n")

        # Training summary
        if metrics:
            latest = metrics[-1]
            f.write("Training Summary:\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Total iterations:    {latest['iteration']}\n")
            f.write(f"  Total games played:  {latest['total_games']}\n")
            f.write(f"  Latest policy loss:  {latest['policy_loss']:.4f}\n")
            f.write(f"  Latest value loss:   {latest['value_loss']:.4f}\n")
            f.write(f"  Latest total loss:   {latest['total_loss']:.4f}\n")
            f.write("\n")

        # Evaluation summary
        if evaluation_results:
            if vs_random:
                f.write("Evaluation vs Random Agent:\n")
                f.write("-" * 80 + "\n")
                for result in evaluation_results:
                    f.write(f"  Iteration {result['iteration']:3d}: "
                            f"Win Rate = {result['win_rate']:5.1f}% "
                            f"({result['wins']:2d}W / {result['losses']:2d}L / {result['draws']:2d}D)\n")
                f.write("\n")

                # Learning milestones
                f.write("Learning Milestones:\n")
                f.write("-" * 80 + "\n")

                for result in evaluation_results:
                    wr = result['win_rate']
                    it = result['iteration']

                    if wr >= 95:
                        f.write(f"  Iteration {it:3d}: STRONG - Dominates random play (>95% win rate)\n")
                        break
                    elif wr >= 85:
                        f.write(f"  Iteration {it:3d}: COMPETENT - Strong tactical play (>85% win rate)\n")
                        break
                    elif wr >= 70:
                        f.write(f"  Iteration {it:3d}: TACTICAL - Recognizes basic threats (>70% win rate)\n")
                        break
                    elif wr >= 60:
                        f.write(f"  Iteration {it:3d}: EMERGING - Showing signs of learning (>60% win rate)\n")
                        break

                f.write("\n")
            else:
                baseline = evaluation_results[0].get('baseline_iter', 'N/A')
                f.write(f"Checkpoint-vs-Checkpoint Evaluation (vs Baseline Iteration {baseline}):\n")
                f.write("-" * 80 + "\n")
                for result in evaluation_results:
                    f.write(f"  Iteration {result['iteration']:3d}: "
                            f"Win Rate = {result['win_rate']:5.1f}% "
                            f"({result['wins']:2d}W / {result['losses']:2d}L / {result['draws']:2d}D)\n")
                f.write("\n")

                # Strength progression
                f.write("Strength Assessment:\n")
                f.write("-" * 80 + "\n")

                improving = 0
                similar = 0
                regressing = 0

                for result in evaluation_results:
                    wr = result['win_rate']
                    if wr > 55:
                        improving += 1
                    elif wr < 45:
                        regressing += 1
                    else:
                        similar += 1

                if regressing > 0:
                    f.write(f"  ⚠ WARNING: {regressing} checkpoint(s) are WEAKER than baseline\n")
                    f.write(f"     Training may be unstable or overfitting\n\n")

                if improving == len(evaluation_results):
                    f.write(f"  ✓ All {improving} checkpoint(s) show improvement over baseline\n")
                    f.write(f"     Training is progressing well!\n\n")
                elif improving > 0:
                    f.write(f"  ≈ Mixed results: {improving} stronger, {similar} similar, {regressing} weaker\n\n")

                f.write("\n")

        f.write("=" * 80 + "\n")

    print(f"Evaluation report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate AlphaZero training progress')
    parser.add_argument('--checkpoints', type=int, nargs='+', default=None,
                        help='Specific checkpoint iterations to evaluate (e.g., 10 20 30)')
    parser.add_argument('--games', type=int, default=20,
                        help='Number of games per evaluation (default: 20)')
    parser.add_argument('--simulations', type=int, default=100,
                        help='MCTS simulations per move (default: 100 for speed, use 400 for accuracy)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='MCTS batch size for GPU optimization (default: 16, use 1 to disable)')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                        help='Checkpoint directory (default: ./checkpoints)')
    parser.add_argument('--log-dir', type=str, default='./logs',
                        help='Log directory (default: ./logs)')
    parser.add_argument('--plot-only', action='store_true',
                        help='Only plot existing loss curves, skip evaluation')
    parser.add_argument('--vs-random', action='store_true',
                        help='Evaluate against random agent instead of checkpoint-vs-checkpoint (less meaningful)')
    parser.add_argument('--baseline', type=int, default=None,
                        help='Baseline checkpoint for comparison (default: first checkpoint)')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file (default: config.yaml)')
    args = parser.parse_args()

    config = AlphaZeroConfig.from_yaml(args.config)

    print("\n" + "=" * 80)
    print("AlphaZero Training Progress Evaluation")
    print("=" * 80 + "\n")

    # Load training metrics
    metrics_path = os.path.join(args.log_dir, "training_metrics.json")
    metrics = load_training_metrics(metrics_path)

    if metrics:
        print(f"Loaded {len(metrics)} training metric entries")
        print(f"Latest iteration: {metrics[-1]['iteration']}")
        print(f"Latest losses - Policy: {metrics[-1]['policy_loss']:.4f}, "
              f"Value: {metrics[-1]['value_loss']:.4f}, "
              f"Total: {metrics[-1]['total_loss']:.4f}")
    else:
        print("No training metrics found. Have you started training yet?")

    # Plot loss curves
    if metrics:
        loss_curve_path = os.path.join(args.log_dir, "loss_curves.png")
        plot_loss_curves(metrics, loss_curve_path)

    # If plot-only mode, exit here
    if args.plot_only:
        print("\n" + "=" * 80)
        return

    # Find checkpoints to evaluate
    if args.checkpoints:
        # Evaluate specific iterations
        all_checkpoints = find_checkpoints(args.checkpoint_dir)
        checkpoints_to_eval = []
        for iteration in args.checkpoints:
            # Find checkpoint with matching iteration
            matching = [cp for cp in all_checkpoints if cp[0] == iteration]
            if matching:
                checkpoints_to_eval.append(matching[0])
            else:
                print(f"Warning: Checkpoint for iteration {iteration} not found")
    else:
        # Evaluate all available checkpoints
        checkpoints_to_eval = find_checkpoints(args.checkpoint_dir)

    if not checkpoints_to_eval:
        print(f"\nNo checkpoints found in {args.checkpoint_dir}")
        print("Train your model first or specify --checkpoint-dir")
        return

    print(f"\nFound {len(checkpoints_to_eval)} checkpoints to evaluate")

    # Evaluate each checkpoint
    evaluation_results = []

    # Determine evaluation mode
    if args.vs_random:
        # Legacy mode: vs random agent
        print(f"\nEvaluation mode: vs Random Agent")
        print("Note: This metric saturates quickly (~95% by iteration 20-30)")
        print("      Use checkpoint-vs-checkpoint for meaningful comparison\n")

        for iteration, checkpoint_path in checkpoints_to_eval:
            print(f"\n{'-' * 80}")
            print(f"Evaluating iteration {iteration}: {os.path.basename(checkpoint_path)}")
            print(f"Playing {args.games} games vs random agent...")

            try:
                results = evaluate_checkpoint_vs_random(
                    checkpoint_path,
                    num_games=args.games,
                    num_simulations=args.simulations,
                    verbose=True,
                    batch_size=args.batch_size,
                    board_size=config.board_size
                )

                results['iteration'] = iteration
                evaluation_results.append(results)

                print(f"Results: {results['wins']}W / {results['losses']}L / {results['draws']}D "
                      f"(Win Rate: {results['win_rate']:.1f}%)")

            except Exception as e:
                print(f"Error evaluating checkpoint: {e}")
                continue

    else:
        # Default mode: checkpoint vs checkpoint (more meaningful)
        print(f"\nEvaluation mode: Checkpoint-vs-Checkpoint")

        if args.baseline:
            baseline_path = os.path.join(args.checkpoint_dir, f"checkpoint_{args.baseline}.pt")
            if not os.path.exists(baseline_path):
                print(f"Error: Baseline checkpoint not found: {baseline_path}")
                return
            baseline_iter = args.baseline
            print(f"Comparing all checkpoints against baseline: Iteration {baseline_iter}\n")
        else:
            # Use first checkpoint as baseline
            if len(checkpoints_to_eval) < 2:
                print("Error: Need at least 2 checkpoints for comparison")
                print("      Use --vs-random to evaluate single checkpoints against random")
                return
            baseline_iter, baseline_path = checkpoints_to_eval[0]
            print(f"Using first checkpoint as baseline: Iteration {baseline_iter}\n")

        for iteration, checkpoint_path in checkpoints_to_eval:
            if iteration == baseline_iter:
                # Skip baseline vs itself
                continue

            print(f"\n{'-' * 80}")
            print(f"Evaluating iteration {iteration} vs iteration {baseline_iter}")
            print(f"  Later: {os.path.basename(checkpoint_path)}")
            print(f"  Earlier: {os.path.basename(baseline_path)}")
            print(f"  Playing {args.games} games...")

            try:
                results = evaluate_checkpoint_vs_checkpoint(
                    checkpoint_path,
                    baseline_path,
                    num_games=args.games,
                    num_simulations=args.simulations,
                    verbose=True,
                    batch_size=args.batch_size,
                    board_size=config.board_size
                )

                results['iteration'] = iteration
                results['baseline_iter'] = baseline_iter
                evaluation_results.append(results)

                print(f"Results: {results['wins']}W / {results['losses']}L / {results['draws']}D "
                      f"(Win Rate: {results['win_rate']:.1f}%)")

                # Interpretation
                if results['win_rate'] > 65:
                    print(f"  ⬆⬆ Iteration {iteration} is MUCH STRONGER than {baseline_iter}")
                elif results['win_rate'] > 55:
                    print(f"  ⬆ Iteration {iteration} is STRONGER than {baseline_iter}")
                elif results['win_rate'] < 35:
                    print(f"  ⬇⬇ Iteration {iteration} is MUCH WEAKER than {baseline_iter} (training issue!)")
                elif results['win_rate'] < 45:
                    print(f"  ⬇ Iteration {iteration} is WEAKER than {baseline_iter} (training issue!)")
                else:
                    print(f"  ≈ Iteration {iteration} is SIMILAR to {baseline_iter}")

            except Exception as e:
                print(f"Error evaluating checkpoint: {e}")
                continue

    # Plot win rate progression
    if evaluation_results:
        win_rate_path = os.path.join(args.log_dir, "win_rate_progression.png")
        plot_win_rate_progression(evaluation_results, win_rate_path, vs_random=args.vs_random)

        # Generate report
        report_path = os.path.join(args.log_dir, "evaluation_report.txt")
        generate_report(metrics, evaluation_results, report_path, vs_random=args.vs_random)

        print("\n" + "=" * 80)
        print("Evaluation Summary:")
        print("=" * 80)

        if args.vs_random:
            for result in evaluation_results:
                print(f"  Iteration {result['iteration']:3d}: Win Rate vs Random = {result['win_rate']:5.1f}%")
        else:
            baseline = evaluation_results[0]['baseline_iter'] if evaluation_results else None
            print(f"\nAll vs Baseline (Iteration {baseline}):")
            print(f"{'Iteration':<12} {'Win Rate':<12} {'Assessment':<20}")
            print("-" * 80)
            for result in evaluation_results:
                if result['win_rate'] > 65:
                    assessment = "⬆⬆ Much Stronger"
                elif result['win_rate'] > 55:
                    assessment = "⬆ Stronger"
                elif result['win_rate'] < 35:
                    assessment = "⬇⬇ Much Weaker"
                elif result['win_rate'] < 45:
                    assessment = "⬇ Weaker"
                else:
                    assessment = "≈ Similar"
                print(f"Iter {result['iteration']:<7} {result['win_rate']:>5.1f}%      {assessment}")

        print("\n" + "=" * 80)
        print("All results saved to:", args.log_dir)
        print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
