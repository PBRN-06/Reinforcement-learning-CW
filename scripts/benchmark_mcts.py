#!/usr/bin/env python3
"""
MCTS Performance Benchmark

Compares sequential MCTS vs batched MCTS performance.
Measures speedup from GPU-optimized batch evaluation.

Usage:
    python scripts/benchmark_mcts.py
    python scripts/benchmark_mcts.py --simulations 800 --batch-sizes 8 16 32
    python scripts/benchmark_mcts.py --games 20
"""

import argparse
import sys
import os
import time
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.alphazero.network import AlphaZeroNetwork
from src.alphazero.mcts import MCTS
from src.alphazero.mcts_batched import BatchedMCTS
from src.alphazero.config import AlphaZeroConfig


def benchmark_mcts_mode(network, mcts_class, config_kwargs, num_games: int = 10, board_size: int = 9):
    """
    Benchmark a specific MCTS configuration.

    Args:
        network: Neural network
        mcts_class: MCTS or BatchedMCTS class
        config_kwargs: Kwargs for MCTS initialization
        num_games: Number of games to simulate

    Returns:
        Average time per game
    """
    # Create MCTS instance
    mcts = mcts_class(network, **config_kwargs)

    # Run games
    total_time = 0.0

    for _ in range(num_games):
        # Random starting position (simulate mid-game)
        board = np.zeros((board_size, board_size), dtype=np.int8)

        # Add a few random moves to make it more realistic
        num_starting_moves = np.random.randint(5, 15)
        for move_idx in range(num_starting_moves):
            empty_cells = np.argwhere(board == 0)
            if len(empty_cells) == 0:
                break
            pos = empty_cells[np.random.randint(len(empty_cells))]
            board[pos[0], pos[1]] = (move_idx % 2) + 1

        # Time MCTS search
        start_time = time.time()
        _ = mcts.search(board, current_player=1, add_noise=False, temperature=1.0)
        elapsed = time.time() - start_time

        total_time += elapsed

    avg_time = total_time / num_games
    return avg_time


def main():
    parser = argparse.ArgumentParser(description='Benchmark MCTS performance')
    parser.add_argument('--simulations', type=int, default=400,
                        help='Number of MCTS simulations (default: 400)')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[8, 16, 32],
                        help='Batch sizes to test (default: 8 16 32)')
    parser.add_argument('--games', type=int, default=10,
                        help='Number of games to average (default: 10)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Load network from checkpoint (optional)')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file (default: config.yaml)')
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("MCTS Performance Benchmark")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  MCTS simulations: {args.simulations}")
    print(f"  Games to average: {args.games}")
    print(f"  Batch sizes: {args.batch_sizes}")
    print("=" * 80 + "\n")

    # Create network
    config = AlphaZeroConfig.from_yaml(args.config)
    network = AlphaZeroNetwork(
        num_res_blocks=config.num_res_blocks,
        num_filters=config.num_filters,
        board_size=config.board_size
    )

    # Move to GPU if available
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network.to(device)
    network.eval()

    print(f"Device: {device}")
    print(f"Network: {config.num_res_blocks} ResBlocks, {config.num_filters} filters\n")

    # Load checkpoint if provided
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        network.load_state_dict(checkpoint['model_state_dict'])
        print("Checkpoint loaded\n")

    print("=" * 80)
    print("Running benchmarks...")
    print("=" * 80 + "\n")

    results = []

    # Benchmark sequential MCTS (baseline)
    print("[1/{}] Testing Sequential MCTS (baseline)...".format(len(args.batch_sizes) + 1))
    sequential_time = benchmark_mcts_mode(
        network,
        MCTS,
        {
            'num_simulations': args.simulations,
            'c_puct': 1.5,
            'dirichlet_alpha': 0.3,
            'dirichlet_epsilon': 0.25
        },
        num_games=args.games,
        board_size=config.board_size
    )

    results.append({
        'name': 'Sequential MCTS',
        'time': sequential_time,
        'speedup': 1.0
    })

    print(f"  Average time per game: {sequential_time:.3f}s\n")

    # Benchmark batched MCTS with different batch sizes
    for idx, batch_size in enumerate(args.batch_sizes, start=2):
        print(f"[{idx}/{len(args.batch_sizes) + 1}] Testing Batched MCTS (batch_size={batch_size})...")

        batched_time = benchmark_mcts_mode(
            network,
            BatchedMCTS,
            {
                'num_simulations': args.simulations,
                'batch_size': batch_size,
                'c_puct': 1.5,
                'dirichlet_alpha': 0.3,
                'dirichlet_epsilon': 0.25
            },
            num_games=args.games,
            board_size=config.board_size
        )

        speedup = sequential_time / batched_time

        results.append({
            'name': f'Batched MCTS (batch={batch_size})',
            'time': batched_time,
            'speedup': speedup
        })

        print(f"  Average time per game: {batched_time:.3f}s")
        print(f"  Speedup: {speedup:.2f}x\n")

    # Print summary
    print("=" * 80)
    print("Benchmark Summary")
    print("=" * 80)

    print(f"\n{'Mode':<30} {'Time/Game':<15} {'Speedup':<10}")
    print("-" * 80)

    for result in results:
        print(f"{result['name']:<30} {result['time']:.3f}s{'':<10} {result['speedup']:.2f}x")

    print("\n" + "=" * 80)

    # Find best configuration
    best = max(results[1:], key=lambda x: x['speedup'])  # Exclude sequential
    print(f"Best Configuration: {best['name']}")
    print(f"  Speedup: {best['speedup']:.2f}x faster than sequential")
    print(f"  Time saved per game: {sequential_time - best['time']:.3f}s")

    # Extrapolate to training
    games_per_iteration = 100
    iterations = 100
    total_games = games_per_iteration * iterations

    time_saved_total = (sequential_time - best['time']) * total_games
    hours_saved = time_saved_total / 3600

    print(f"\nProjected time saved over {iterations} iterations ({total_games} games):")
    print(f"  {hours_saved:.1f} hours")

    print("=" * 80 + "\n")

    # Recommendation
    print("Recommendation:")
    if device.type == 'cuda':
        print(f"  ✓ GPU detected - use batched MCTS with batch_size={best['name'].split('=')[1].rstrip(')')}")
        print(f"  ✓ Add to config: mcts_batch_size = {best['name'].split('=')[1].rstrip(')')}")
    else:
        print("  ⚠ CPU detected - batching may not provide significant speedup")
        print("  ⚠ Consider training on a GPU for best performance")

    print("\n")


if __name__ == '__main__':
    main()
