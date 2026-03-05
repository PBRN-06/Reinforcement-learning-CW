#!/usr/bin/env python3
"""
AlphaZero Evaluation Script

Evaluate AlphaZero agents against each other or against baselines.

Usage:
    python scripts/evaluate.py --agent1 checkpoints/checkpoint_1000.pt --agent2 random
    python scripts/evaluate.py --agent1 checkpoints/checkpoint_1000.pt --agent2 checkpoints/checkpoint_500.pt --games 50
    python scripts/evaluate.py --agent1 checkpoints/checkpoint_100.pt --agent2 random --simulations 200
"""

import argparse
import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.board import Board
from src.alphazero.agent import AlphaZeroAgent
from src.agent import RL_Agent
from src.alphazero.utils import check_terminal


def play_game(agent1, agent2, verbose: bool = False):
    """
    Play one game between two agents.

    Args:
        agent1: First agent (plays as Player 1)
        agent2: Second agent (plays as Player 2)
        verbose: Whether to print game progress

    Returns:
        Winner (1, 2, or 0 for draw)
    """
    board = Board()
    agents = [agent1, agent2]
    current_player = 0
    move_count = 0

    if verbose:
        print("\n" + "=" * 40)
        print("Starting new game")
        print("=" * 40)

    while not board.check_win() and move_count < 81:
        # Get move from current agent
        move = agents[current_player].command(board, 0)

        if verbose:
            print(f"Player {current_player + 1} plays {move}")

        # Apply move
        if board.check_valid(move):
            board.update(move, current_player + 1)
        else:
            if verbose:
                print(f"Invalid move by Player {current_player + 1}: {move}")
            # Invalid move = instant loss
            return 2 if current_player == 0 else 1

        # Switch player
        current_player = 1 - current_player
        move_count += 1

    # Determine winner
    winner = check_terminal(board.base)

    if verbose:
        if winner == 0:
            print("Game ended in a draw")
        else:
            print(f"Player {winner} wins!")

    return winner if winner is not None else 0


def main():
    parser = argparse.ArgumentParser(description='Evaluate AlphaZero agents')
    parser.add_argument('--agent1', type=str, required=True,
                        help='Path to agent1 checkpoint or "random"')
    parser.add_argument('--agent2', type=str, required=True,
                        help='Path to agent2 checkpoint or "random"')
    parser.add_argument('--games', type=int, default=20,
                        help='Number of games to play (default: 20)')
    parser.add_argument('--simulations', type=int, default=400,
                        help='MCTS simulations per move (default: 400)')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='Temperature for action selection (default: 0.0)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed game information')
    args = parser.parse_args()

    # Load agents
    print("\n" + "=" * 60)
    print("AlphaZero Evaluation")
    print("=" * 60)

    if args.agent1 == 'random':
        agent1 = RL_Agent()
        agent1_name = "Random"
    else:
        agent1 = AlphaZeroAgent(
            checkpoint_path=args.agent1,
            num_simulations=args.simulations,
            temperature=args.temperature
        )
        agent1_name = f"AlphaZero({os.path.basename(args.agent1)})"

    if args.agent2 == 'random':
        agent2 = RL_Agent()
        agent2_name = "Random"
    else:
        agent2 = AlphaZeroAgent(
            checkpoint_path=args.agent2,
            num_simulations=args.simulations,
            temperature=args.temperature
        )
        agent2_name = f"AlphaZero({os.path.basename(args.agent2)})"

    print(f"\nAgent 1 (Player 1 & 2 alternating): {agent1_name}")
    print(f"Agent 2 (Player 2 & 1 alternating): {agent2_name}")
    print(f"MCTS simulations: {args.simulations}")
    print(f"Temperature: {args.temperature}")
    print(f"Number of games: {args.games}")
    print("\n" + "=" * 60)

    # Play games
    wins = {1: 0, 2: 0, 0: 0}

    for i in range(args.games):
        # Alternate starting player
        if i % 2 == 0:
            # agent1 plays first
            winner = play_game(agent1, agent2, verbose=args.verbose)
        else:
            # agent2 plays first
            winner = play_game(agent2, agent1, verbose=args.verbose)
            # Swap winner perspective
            if winner == 1:
                winner = 2
            elif winner == 2:
                winner = 1

        wins[winner] += 1

        result_str = (
            f"Agent1" if winner == 1 else
            f"Agent2" if winner == 2 else
            "Draw"
        )
        print(f"Game {i + 1}/{args.games}: {result_str} wins")

    # Print final results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Agent1 wins: {wins[1]:3d} ({wins[1] / args.games * 100:5.1f}%)")
    print(f"Agent2 wins: {wins[2]:3d} ({wins[2] / args.games * 100:5.1f}%)")
    print(f"Draws:       {wins[0]:3d} ({wins[0] / args.games * 100:5.1f}%)")
    print("=" * 60)

    # Determine stronger agent
    if wins[1] > wins[2]:
        print(f"\n{agent1_name} is stronger!")
    elif wins[2] > wins[1]:
        print(f"\n{agent2_name} is stronger!")
    else:
        print(f"\nAgents are evenly matched!")


if __name__ == '__main__':
    main()
