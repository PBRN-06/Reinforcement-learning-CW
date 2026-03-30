#!/usr/bin/env python3
"""
Play Gomoku vs an AI agent with GUI

Interactive GUI for playing against a trained AlphaZero agent or pure MCTS.

Usage:
    python scripts/play_gui.py --checkpoint checkpoints/checkpoint_1000.pt
    python scripts/play_gui.py --checkpoint checkpoints_5x5/checkpoint_100.pt
    python scripts/play_gui.py --checkpoint checkpoints/checkpoint_1000.pt --simulations 800
    python scripts/play_gui.py --checkpoint checkpoints/checkpoint_1000.pt --human-first
    python scripts/play_gui.py --mcts --config config_5x5.yaml
    python scripts/play_gui.py --mcts --board-size 5
    python scripts/play_gui.py --mcts --simulations 800
    python scripts/play_gui.py --mcts --human-first
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agent import Player
from src.board import Board
from src.alphazero.agent import AlphaZeroAgent
from src.alphazero.config import AlphaZeroConfig
from src.pure_mcts_agent import PureMCTSAgent
from PySide6.QtWidgets import QApplication
from main import GameWindow


def load_config_from_checkpoint(checkpoint_path: str):
    """Try to extract AlphaZeroConfig from a checkpoint file."""
    import torch
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = checkpoint.get('config')
    if config is not None and isinstance(config, AlphaZeroConfig):
        return config
    return None


def main():
    parser = argparse.ArgumentParser(description='Play Gomoku vs an AI agent')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--checkpoint', type=str,
                       help='Path to AlphaZero checkpoint')
    group.add_argument('--mcts', action='store_true',
                       help='Play against pure MCTS (no neural network)')
    parser.add_argument('--simulations', type=int, default=400,
                        help='MCTS simulations per move (default: 400)')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='Temperature for AlphaZero moves (default: 0.0 = deterministic)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='MCTS batch size for GPU optimization (default: 16, use 1 to disable)')
    parser.add_argument('--human-first', action='store_true',
                        help='Human plays first (default: AI plays first)')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to configuration file (overrides checkpoint config)')
    parser.add_argument('--board-size', type=int, default=None,
                        help='Board size (default: from checkpoint or config)')
    args = parser.parse_args()

    # Resolve config: checkpoint > --config flag > default config.yaml
    config = None
    if args.checkpoint:
        config = load_config_from_checkpoint(args.checkpoint)
        if config is not None:
            print(f"Loaded config from checkpoint (board_size={config.board_size})")

    if args.config:
        config = AlphaZeroConfig.from_yaml(args.config)
        print(f"Using config file: {args.config} (board_size={config.board_size})")
    elif config is None:
        config = AlphaZeroConfig.from_yaml('config.yaml')
        print(f"Using default config.yaml (board_size={config.board_size})")

    # CLI --board-size overrides everything
    if args.board_size is not None:
        config.board_size = args.board_size
        print(f"Board size overridden to {args.board_size}")

    print("\n" + "=" * 60)

    if args.mcts:
        print(f"Gomoku {config.board_size}x{config.board_size} vs Pure MCTS")
        print("=" * 60)
        print(f"\nUsing Pure MCTS agent")
        print(f"  Simulations: {args.simulations}")
        ai = PureMCTSAgent(num_simulations=args.simulations)
        ai_name = f"Pure MCTS (sims={args.simulations})"
    else:
        print(f"Gomoku {config.board_size}x{config.board_size} vs AlphaZero")
        print("=" * 60)
        print(f"\nLoading AlphaZero agent...")
        print(f"  Checkpoint: {args.checkpoint}")
        print(f"  Board size: {config.board_size}x{config.board_size}")
        print(f"  Network: {config.num_res_blocks} res blocks, {config.num_filters} filters")
        print(f"  MCTS simulations: {args.simulations}")
        print(f"  Temperature: {args.temperature}")
        ai = AlphaZeroAgent(
            checkpoint_path=args.checkpoint,
            num_simulations=args.simulations,
            temperature=args.temperature,
            num_res_blocks=config.num_res_blocks,
            num_filters=config.num_filters,
            board_size=config.board_size,
            batch_size=args.batch_size
        )
        ai_name = f"AlphaZero ({os.path.basename(args.checkpoint)})"

    # Setup players
    human = Player()
    if args.human_first:
        players = (human, ai)
        player_names = ("Human", ai_name)
        print(f"\nYou are playing as Player 1 (Red)")
        print(f"{ai_name} is Player 2 (Blue)")
    else:
        players = (ai, human)
        player_names = (ai_name, "Human")
        print(f"\n{ai_name} is Player 1 (Red)")
        print(f"You are playing as Player 2 (Blue)")

    print("\nStarting game...")
    print("=" * 60 + "\n")

    # Start game
    board = Board(config.board_size)
    app = QApplication(sys.argv)
    window = GameWindow(board, config.board_size, players, player_names)
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
