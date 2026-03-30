# Reinforcement-learning-CW

AlphaZero implementation for Gomoku (5-in-a-row) on a configurable board size. Trains a neural network combined with Monte Carlo Tree Search through self-play reinforcement learning.

## Setup

Requires Python 3.10+ and PyTorch with CUDA support.

```bash
pip install torch numpy matplotlib pyyaml
```

Additionally requires pyside6 for `scripts/play_gui.py`

```bash
pip install pyside6
```

## Configuration

Training is controlled via YAML config files:

| Config | Description |
|--------|-------------|
| `config.yaml` | Default 9x9 board, tuned for 3050 Ti |
| `config_4090.yaml` | 9x9 board, tuned for RTX 4090 |
| `config_5x5.yaml` | 5x5 board variant |

Key parameters in the config:
- **Network**: `num_res_blocks`, `num_filters` — ResNet architecture size
- **MCTS**: `num_simulations`, `c_puct`, `mcts_batch_size` — search strength and GPU batching
- **Self-play**: `games_per_iteration`, `temperature_threshold` — data generation
- **Training**: `learning_rate`, `epochs_per_iteration`, `batch_size` — optimization
- **Board**: `board_size`, `win_length` — game rules

## Scripts

### Training

```bash
# Start training
python scripts/train.py --config config.yaml

# Resume from checkpoint
python scripts/train.py --config config.yaml --resume checkpoints/checkpoint_100.pt

# Override device or iteration count
python scripts/train.py --config config.yaml --device cpu --iterations 200
```

Checkpoints are saved to `checkpoint_dir` at decreasing frequency (every iteration early on, every 10th iteration past iteration 100). Training can be interrupted with Ctrl+C and will save a checkpoint before exiting.

### Playing

```bash
# Play against a trained AlphaZero agent
python scripts/play_gui.py --checkpoint checkpoints/checkpoint_200.pt

# Play against pure MCTS
python scripts/play_gui.py --mcts --simulations 800

# Human plays first (black)
python scripts/play_gui.py --checkpoint checkpoints/checkpoint_200.pt --human-first

# 5x5 board
python scripts/play_gui.py --mcts --config config_5x5.yaml
```

### Evaluation

#### Single matchup (`evaluate.py`)

Play games between any two agents (checkpoint, `random`, or `mcts`):

```bash
# Checkpoint vs random
python scripts/evaluate.py --agent1 checkpoints/checkpoint_200.pt --agent2 random

# Checkpoint vs checkpoint
python scripts/evaluate.py --agent1 checkpoints/checkpoint_200.pt --agent2 checkpoints/checkpoint_50.pt --games 50

# Checkpoint vs pure MCTS
python scripts/evaluate.py --agent1 checkpoints/checkpoint_200.pt --agent2 mcts --simulations 400
```

| Flag | Default | Description |
|------|---------|-------------|
| `--agent1` | required | Checkpoint path, `random`, or `mcts` |
| `--agent2` | required | Checkpoint path, `random`, or `mcts` |
| `--games` | 20 | Number of games (alternates starting player) |
| `--simulations` | 400 | MCTS simulations per move |
| `--temperature` | 0.0 | Action selection temperature (0 = deterministic) |
| `--batch-size` | 16 | MCTS batch size for GPU |
| `--verbose` | off | Print per-move details |

#### Training progress (`evaluate_progress.py`)

Evaluates training progression across checkpoints with loss curve plotting:

```bash
# Full evaluation: loss curves + checkpoint-vs-checkpoint
python scripts/evaluate_progress.py

# Only loss curves (no games played)
python scripts/evaluate_progress.py --plot-only

# Evaluate against random agent
python scripts/evaluate_progress.py --vs-random

# Compare all checkpoints against a specific baseline
python scripts/evaluate_progress.py --baseline 50 --games 30

# Evaluate specific checkpoints only
python scripts/evaluate_progress.py --checkpoints 10 50 100 200
```

Generates loss curve plots (`logs/loss_curves.png`), win rate progression charts (`logs/win_rate_progression.png`), and a text report (`logs/evaluation_report.txt`).

#### Batch checkpoint evaluation (`evaluate_checkpoints.py`)

Evaluates all checkpoints against Random and Pure MCTS, saves results to JSON. Optionally computes Elo ratings between checkpoints:

```bash
# Evaluate all checkpoints vs Random and Pure MCTS
python scripts/evaluate_checkpoints.py --checkpoint-dir ./checkpoints

# Skip MCTS evaluation (faster)
python scripts/evaluate_checkpoints.py --skip-mcts --games 50

# Only specific checkpoints
python scripts/evaluate_checkpoints.py --checkpoints 10 50 100 200

# With Elo ratings (auto-selects ~10 evenly-spaced checkpoints)
python scripts/evaluate_checkpoints.py --elo

# Elo with specific checkpoints
python scripts/evaluate_checkpoints.py --elo --elo-checkpoints 10 50 100 200

# Custom output
python scripts/evaluate_checkpoints.py --output ./results/eval.json
```

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint-dir` | `./checkpoints` | Directory containing checkpoints |
| `--checkpoints` | all | Specific iterations to evaluate |
| `--games` | 20 | Games per evaluation |
| `--simulations` | 100 | MCTS sims for AlphaZero agents |
| `--mcts-simulations` | 200 | Sims for Pure MCTS opponent (slower, so separate setting) |
| `--batch-size` | 16 | MCTS batch size |
| `--output` | `./logs/checkpoint_evaluation.json` | Output JSON path |
| `--skip-random` | off | Skip vs-random evaluation |
| `--skip-mcts` | off | Skip vs-MCTS evaluation |
| `--elo` | off | Compute Elo ratings via round-robin |
| `--elo-checkpoints` | auto (10) | Specific iterations for Elo |
| `--elo-games` | 20 | Games per Elo matchup |

Output JSON format:
```json
{
  "metadata": { "checkpoint_dir": "...", "num_games": 20, "board_size": 9, "timestamp": "..." },
  "vs_random": [{ "iteration": 10, "wins": 18, "losses": 1, "draws": 1, "win_rate": 90.0 }],
  "vs_mcts": [{ "iteration": 10, "wins": 12, "losses": 5, "draws": 3, "win_rate": 60.0 }],
  "elo_ratings": { "10": 980, "50": 1015, "100": 1105 }
}
```

### Other scripts

- **`scripts/benchmark_mcts.py`** — Benchmarks sequential vs batched MCTS performance to measure GPU speedup and assess the correct size for a specific setup.
- **`scripts/view_buffer.py`** — Visualises training positions and policy targets from the replay buffer to check training data at runtime.

## Project Structure

```
├── config.yaml              # Training configuration
├── scripts/
│   ├── train.py             # Training entry point
│   ├── play_gui.py          # Interactive GUI to play against agents
│   ├── evaluate.py          # Single matchup evaluation
│   ├── evaluate_progress.py # Training progress analysis with plots
│   ├── evaluate_checkpoints.py  # Batch evaluation with JSON output and Elo
│   ├── benchmark_mcts.py    # MCTS performance benchmarking
│   └── view_buffer.py       # Replay buffer visualisation
├── src/
│   ├── board.py             # Game board logic
│   ├── agent.py             # Random agent
│   ├── pure_mcts_agent.py   # Pure MCTS agent (no neural network)
│   └── alphazero/
│       ├── agent.py         # AlphaZero agent (neural MCTS)
│       ├── config.py        # Configuration dataclass
│       ├── trainer.py       # Self-play training loop
│       ├── network.py       # ResNet policy-value network
│       ├── mcts.py          # MCTS implementation (sequential + batched)
│       └── utils.py         # Game utilities
├── checkpoints/             # Saved model checkpoints
└── logs/                    # Training metrics and evaluation outputs
```
