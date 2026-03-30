import json
import argparse
import os
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--log-dir", type=str, default="logs_5x5", help="Log directory containing checkpoint_evaluation.json")
args = parser.parse_args()

eval_path = os.path.join(args.log_dir, "checkpoint_evaluation.json")
with open(eval_path) as f:
    data = json.load(f)

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

for ax, (key, title) in zip(axes, [("vs_random", "vs Random Agent"), ("vs_mcts", "vs Pure MCTS (1600 sims)")]):
    entries = data[key]
    iters = [e["iteration"] for e in entries]
    wins = [e["win_rate"] for e in entries]
    losses = [e["loss_rate"] for e in entries]
    draws = [e["draw_rate"] for e in entries]

    ax.stackplot(iters, wins, draws, losses,
                 labels=["Win %", "Draw %", "Loss %"],
                 colors=["#4CAF50", "#FFC107", "#F44336"], alpha=0.85)
    ax.set_title(f"AlphaZero {title}", fontsize=13)
    ax.set_xlabel("Training Iteration")
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)

board_size = data["metadata"].get("board_size", "?")
fig.suptitle(f"{board_size}x{board_size} — AlphaZero Checkpoint Evaluation", fontsize=14, y=1.02)
plt.tight_layout()
output_path = os.path.join(args.log_dir, "evaluation_plot.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved to {output_path}")
