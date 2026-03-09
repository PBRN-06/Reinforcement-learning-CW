#!/usr/bin/env python3
"""
AlphaZero Training Script

Main entry point for training the AlphaZero agent.

Usage:
    python scripts/train.py --config config.yaml
    python scripts/train.py --config config.yaml --resume checkpoints/checkpoint_100.pt
    python scripts/train.py --config config.yaml --device cpu
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.alphazero.config import AlphaZeroConfig
from src.alphazero.trainer import AlphaZeroTrainer
import torch


def main():
    parser = argparse.ArgumentParser(description='Train AlphaZero for Gomoku')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file (default: config.yaml)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu), overrides config')
    parser.add_argument('--iterations', type=int, default=None,
                        help='Number of training iterations, overrides config')
    args = parser.parse_args()

    # Load configuration
    print(f"\nLoading configuration from {args.config}...")
    config = AlphaZeroConfig.from_yaml(args.config)

    # Override device if specified
    if args.device:
        config.device = args.device

    # Override iterations if specified
    if args.iterations:
        config.num_iterations = args.iterations

    print(f"\nTraining Configuration:")
    print(f"  Device: {config.device}")
    print(f"  Network: {config.num_res_blocks} ResBlocks, {config.num_filters} filters")
    print(f"  MCTS: {config.num_simulations} simulations per move")
    print(f"  Self-play: {config.games_per_iteration} games per iteration")
    print(f"  Training: {config.epochs_per_iteration} epochs, batch size {config.batch_size}, recency fraction {config.recent_priority}")
    print(f"  Total iterations: {config.num_iterations}")
    print(f"  Checkpoint frequency: every {config.checkpoint_freq} iterations")

    # Create trainer
    trainer = AlphaZeroTrainer(config)

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Start training
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user. Saving checkpoint...")
        trainer.save_checkpoint()
        print("Checkpoint saved. Exiting.")
        sys.exit(0)

    # Save final checkpoint
    print("\nSaving final checkpoint...")
    trainer.save_checkpoint()
    print("Training complete!")


if __name__ == '__main__':
    main()
