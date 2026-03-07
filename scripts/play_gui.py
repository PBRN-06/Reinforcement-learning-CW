#!/usr/bin/env python3
"""
Play Gomoku vs AlphaZero Agent with GUI

Interactive GUI for playing against a trained AlphaZero agent.

Usage:
    python scripts/play_gui.py --checkpoint checkpoints/checkpoint_1000.pt
    python scripts/play_gui.py --checkpoint checkpoints/checkpoint_1000.pt --simulations 800
    python scripts/play_gui.py --checkpoint checkpoints/checkpoint_1000.pt --human-first
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agent import Player
from src.board import Board
from src.alphazero.agent import AlphaZeroAgent
from PySide6.QtWidgets import QApplication
from main import GameWindow


def main():
    parser = argparse.ArgumentParser(description='Play Gomoku vs AlphaZero')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to AlphaZero checkpoint')
    parser.add_argument('--simulations', type=int, default=400,
                        help='MCTS simulations per move (default: 400)')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='Temperature for AI moves (default: 0.0 = deterministic)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='MCTS batch size for GPU optimization (default: 16, use 1 to disable)')
    parser.add_argument('--human-first', action='store_true',
                        help='Human plays first (default: AI plays first)')
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("Gomoku vs AlphaZero")
    print("=" * 60)

    # Load AlphaZero agent
    print(f"\nLoading AlphaZero agent...")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  MCTS simulations: {args.simulations}")
    print(f"  Temperature: {args.temperature}")

    alphazero = AlphaZeroAgent(
        checkpoint_path=args.checkpoint,
        num_simulations=args.simulations,
        temperature=args.temperature,
        batch_size=args.batch_size
    )

    # Setup players
    human = Player()
    if args.human_first:
        players = (human, alphazero)
        print(f"\nYou are playing as Player 1 (Red)")
        print(f"AlphaZero is Player 2 (Blue)")
    else:
        players = (alphazero, human)
        print(f"\nAlphaZero is Player 1 (Red)")
        print(f"You are playing as Player 2 (Blue)")

    print("\nStarting game...")
    print("=" * 60 + "\n")

    # Start game
    board = Board()
    app = QApplication(sys.argv)
    window = GameWindow(board, 9, players)
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
