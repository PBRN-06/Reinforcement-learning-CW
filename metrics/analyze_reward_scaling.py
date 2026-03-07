"""
Analyze reward scaling with new weights to determine optimal normalization factor.
"""
import numpy as np
from src.rewards import calculate_reward

print("=" * 70)
print("REWARD WEIGHT ANALYSIS")
print("=" * 70)

# Current weights from rewards.py
print("\nCurrent reward configuration:")
print("  Player sequences:  2→1, 3→5, 4→50")
print("  Opponent penalties: 2→-3, 3→-10, 4→-80")
print("  (Asymmetric: blocking opponent is weighted higher)\n")

# Test various board scenarios
scenarios = []

# Scenario 1: Early game - single 2-in-a-row
board = np.zeros((9, 9), dtype=np.int8)
board[4, 4] = 1
board[4, 5] = 1
reward = calculate_reward(board, player=1, board_size=9)
scenarios.append(("Early game: 1× 2-in-a-row", reward))

# Scenario 2: Multiple 2s
board = np.zeros((9, 9), dtype=np.int8)
board[4, 4] = 1
board[4, 5] = 1
board[5, 4] = 1
board[6, 4] = 1
reward = calculate_reward(board, player=1, board_size=9)
scenarios.append(("Early game: 2× 2-in-a-row", reward))

# Scenario 3: One 3-in-a-row
board = np.zeros((9, 9), dtype=np.int8)
board[4, 4] = 1
board[4, 5] = 1
board[4, 6] = 1
reward = calculate_reward(board, player=1, board_size=9)
scenarios.append(("Mid game: 1× 3-in-a-row", reward))

# Scenario 4: Threatening 4-in-a-row
board = np.zeros((9, 9), dtype=np.int8)
board[4, 4] = 1
board[4, 5] = 1
board[4, 6] = 1
board[4, 7] = 1
reward = calculate_reward(board, player=1, board_size=9)
scenarios.append(("Late game: 1× 4-in-a-row", reward))

# Scenario 5: Mixed position (realistic mid-game)
board = np.zeros((9, 9), dtype=np.int8)
# Player sequences
board[4, 4] = 1
board[4, 5] = 1
board[4, 6] = 1  # 3-in-a-row
board[2, 2] = 1
board[3, 3] = 1  # 2-in-a-row diagonal
# Opponent sequences
board[6, 6] = 2
board[6, 7] = 2  # opponent 2-in-a-row
reward = calculate_reward(board, player=1, board_size=9)
scenarios.append(("Realistic mid-game (mixed)", reward))

# Scenario 6: Opponent threatening
board = np.zeros((9, 9), dtype=np.int8)
board[4, 4] = 2
board[4, 5] = 2
board[4, 6] = 2
board[4, 7] = 2  # Opponent 4-in-a-row (dangerous!)
board[2, 2] = 1
board[2, 3] = 1  # Our weak 2-in-a-row
reward = calculate_reward(board, player=1, board_size=9)
scenarios.append(("Opponent 4-in-a-row threat", reward))

# Scenario 7: Strong offensive position
board = np.zeros((9, 9), dtype=np.int8)
board[4, 4] = 1
board[4, 5] = 1
board[4, 6] = 1  # 3-in-a-row
board[2, 2] = 1
board[3, 3] = 1
board[4, 4] = 1  # Another 3 (diagonal overlaps)
board[5, 5] = 1
board[6, 6] = 1  # 2-in-a-row
reward = calculate_reward(board, player=1, board_size=9)
scenarios.append(("Strong offensive position", reward))

print("SCENARIO ANALYSIS:")
print("-" * 70)
for desc, r in scenarios:
    print(f"{desc:40s} → reward = {r:6.0f}")

# Calculate statistics
rewards = [r for _, r in scenarios]
print("\nReward range:")
print(f"  Min: {min(rewards):.0f}")
print(f"  Max: {max(rewards):.0f}")
print(f"  Mean: {np.mean(rewards):.1f}")
print(f"  Std: {np.std(rewards):.1f}")

# Test different normalization factors
print("\n" + "=" * 70)
print("NORMALIZATION FACTOR ANALYSIS")
print("=" * 70)

norm_factors = [50, 100, 150, 200]
print("\nTesting different normalization factors with tanh(reward / factor):\n")

for factor in norm_factors:
    print(f"\n--- Normalization factor: {factor} ---")
    print(f"{'Scenario':<40s} {'Raw':>6s} {'Norm':>7s} {'×0.1':>7s}")
    print("-" * 70)

    for desc, r in scenarios:
        normalized = np.tanh(r / factor)
        weighted = 0.1 * normalized
        print(f"{desc:40s} {r:6.0f} {normalized:7.3f} {weighted:7.4f}")

    # Show saturation points
    print(f"\nSaturation analysis (factor={factor}):")
    for sat_reward in [factor, 2*factor, 3*factor]:
        sat_norm = np.tanh(sat_reward / factor)
        print(f"  reward={sat_reward:4.0f} → normalized={sat_norm:.3f}")

# Recommendation
print("\n" + "=" * 70)
print("RECOMMENDATION")
print("=" * 70)

recommended_factor = 100

print(f"\nRecommended normalization factor: {recommended_factor}")
print("\nRationale:")
print("  • Typical mid-game positions (reward ~5-20) normalize to 0.05-0.20")
print("  • With weight=0.1, contribution is 0.005-0.020 (meaningful but not dominant)")
print("  • Threatening 4-in-a-row (reward ~50) → normalized ~0.46 → contribution ~0.046")
print("  • Opponent threats (reward ~-80) → normalized ~-0.66 → contribution ~-0.066")
print("  • Allows shaped rewards to guide without overwhelming win/loss signal")
print("  • Good balance between early-game guidance and late-game precision")

print("\n" + "=" * 70)
print("COMBINED OUTCOME EXAMPLES (with factor=100, weight=0.1)")
print("=" * 70)

print("\n  Win + strong position (reward=50):")
print(f"    1.0 + 0.1×tanh(50/100) = 1.0 + 0.1×0.462 = 1.046")

print("\n  Win + weak position (reward=5):")
print(f"    1.0 + 0.1×tanh(5/100) = 1.0 + 0.1×0.050 = 1.005")

print("\n  Loss but decent position (reward=10):")
print(f"    -1.0 + 0.1×tanh(10/100) = -1.0 + 0.1×0.100 = -0.990")

print("\n  Loss with opponent dominating (reward=-80):")
print(f"    -1.0 + 0.1×tanh(-80/100) = -1.0 + 0.1×(-0.664) = -1.066")

print("\n  Draw with balanced position (reward=0):")
print(f"    0.0 + 0.1×tanh(0/100) = 0.0 + 0.1×0.0 = 0.000")

print("\n✅ Analysis complete!")
