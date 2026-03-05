# AlphaZero for Gomoku

Complete AlphaZero reinforcement learning implementation for training an AI to play Gomoku (9x9, 5-in-a-row).

## Features

- **ResNet Policy-Value Network**: 4-6 residual blocks with 64 filters
- **Monte Carlo Tree Search (MCTS)**: 200-400 simulations per move (configurable)
- **Self-Play Training**: Generates games for continuous improvement
- **Iteration-Based Checkpoints**: Saves models every N games
- **Full Integration**: Play against trained agent via GUI

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

**Requirements:**
- Python 3.8+
- PyTorch 2.0+
- NumPy
- PySide6
- PyYAML

## Quick Start

### 1. Training

Train a new AlphaZero agent from scratch:

```bash
python scripts/train.py --config config.yaml
```

**Training Options:**
```bash
# Resume from checkpoint
python scripts/train.py --config config.yaml --resume checkpoints/checkpoint_100.pt

# Use CPU instead of GPU
python scripts/train.py --config config.yaml --device cpu

# Override number of iterations
python scripts/train.py --config config.yaml --iterations 100
```

**What happens during training:**
1. **Self-Play**: Agent plays against itself using MCTS
2. **Data Collection**: Stores (state, policy, outcome) tuples
3. **Network Training**: Trains on collected data for 10 epochs
4. **Checkpointing**: Saves model every 10 iterations

### 2. Evaluation

Pit two models (or a model vs random) against each other:

```bash
# Trained model vs random baseline
python scripts/evaluate.py --agent1 checkpoints/checkpoint_100.pt --agent2 random --games 20

# Compare two checkpoints
python scripts/evaluate.py --agent1 checkpoints/checkpoint_1000.pt --agent2 checkpoints/checkpoint_500.pt --games 50

# Use fewer simulations for faster evaluation
python scripts/evaluate.py --agent1 checkpoints/checkpoint_100.pt --agent2 random --simulations 200
```

### 3. Play vs AI (GUI)

Play against a trained agent using the graphical interface:

```bash
# AI plays first
python scripts/play_gui.py --checkpoint checkpoints/checkpoint_1000.pt

# You play first
python scripts/play_gui.py --checkpoint checkpoints/checkpoint_1000.pt --human-first

# Faster play (fewer simulations)
python scripts/play_gui.py --checkpoint checkpoints/checkpoint_1000.pt --simulations 200
```

## Configuration

Edit `config.yaml` to customize training parameters:

```yaml
# Network Architecture
num_res_blocks: 6        # More blocks = stronger but slower
num_filters: 64          # More filters = more capacity

# MCTS Parameters
num_simulations: 200     # 200-800 (strength vs speed tradeoff)
c_puct: 1.5              # Exploration constant

# Training
games_per_iteration: 100 # Games to generate per iteration
batch_size: 256          # Training batch size
learning_rate: 0.001     # Adam learning rate
epochs_per_iteration: 10 # Training epochs per iteration

# Checkpointing
checkpoint_freq: 10      # Save every N iterations
```

## Project Structure

```
.
├── config.yaml                 # Training configuration
├── requirements.txt            # Python dependencies
│
├── src/
│   ├── alphazero/             # AlphaZero implementation
│   │   ├── config.py          # Configuration dataclass
│   │   ├── network.py         # ResNet policy-value network
│   │   ├── mcts.py            # Monte Carlo Tree Search
│   │   ├── self_play.py       # Self-play game generation
│   │   ├── trainer.py         # Training loop
│   │   ├── agent.py           # AlphaZeroAgent (game integration)
│   │   └── utils.py           # Helper functions
│   │
│   ├── data/                  # Data management
│   │   └── replay_buffer.py   # Training data storage
│   │
│   ├── agent.py               # Base agent classes
│   ├── board.py               # Game board logic
│   └── rewards.py             # Reward calculation
│
├── scripts/
│   ├── train.py               # Main training script
│   ├── evaluate.py            # Model evaluation
│   └── play_gui.py            # Play vs AI (GUI)
│
├── checkpoints/               # Saved model checkpoints
├── training_data/             # Self-play game data
└── logs/                      # Training logs
```

## Training Timeline & Performance

**Expected Training Time** (on modern GPU):
- 1 iteration: ~15-30 minutes
- 100 iterations: ~1-2 days
- 1000 iterations: ~2-3 weeks

**Strength Progression:**
- **Iteration 50**: Beats random agent >90% of the time
- **Iteration 200**: Competent intermediate play
- **Iteration 500**: Strong amateur level
- **Iteration 1000+**: Expert level (near-optimal play)

**Hardware Requirements:**
- **GPU**: GTX 1060 or better (8GB+ VRAM recommended)
- **RAM**: 16GB minimum
- **Disk**: 10GB for checkpoints and training data

## How It Works

### AlphaZero Algorithm

1. **Neural Network**
   - **Input**: 3-channel board representation (Player 1, Player 2, Empty)
   - **Output**: Policy (81 action probabilities) + Value (win probability)

2. **Monte Carlo Tree Search (MCTS)**
   - Uses neural network for move evaluation
   - UCB formula balances exploration/exploitation
   - Dirichlet noise encourages exploration during self-play

3. **Self-Play**
   - Agent plays against itself using MCTS
   - Stores (board state, MCTS policy, game outcome) for training
   - 8-fold data augmentation via rotations/flips

4. **Training**
   - **Policy Loss**: Cross-entropy between MCTS policy and network policy
   - **Value Loss**: MSE between predicted value and actual outcome
   - Trains on replay buffer of recent games

5. **Iteration**
   - Self-play → Add to buffer → Train network → Checkpoint
   - New checkpoint becomes opponent for next iteration

### Key Design Decisions

- **Canonical Representation**: Current player always sees themselves as '1'
- **Temperature Annealing**: High temperature (1.0) for first 15 moves (exploration), then low (0.1) for exploitation
- **No Rollouts**: Uses neural network value directly (AlphaZero approach, not AlphaGo)
- **Replay Buffer**: 500k examples with sliding window

## Customization

### Adjust Network Size

For faster training (weaker play):
```yaml
num_res_blocks: 4
num_filters: 32
num_simulations: 200
```

For stronger play (slower training):
```yaml
num_res_blocks: 8
num_filters: 128
num_simulations: 800
```

### Adjust Training Speed

For faster iterations (less data):
```yaml
games_per_iteration: 50
epochs_per_iteration: 5
```

For better performance (more data):
```yaml
games_per_iteration: 200
epochs_per_iteration: 20
```

## Troubleshooting

### Out of Memory Errors

Reduce batch size or network size:
```yaml
batch_size: 128
num_filters: 32
```

### Training Too Slow

- Reduce `num_simulations` (e.g., 200)
- Reduce `games_per_iteration` (e.g., 50)
- Use fewer `num_res_blocks` (e.g., 4)

### Network Not Learning

- Check loss curves (should decrease)
- Increase `replay_buffer_size`
- Reduce `learning_rate`
- Ensure using GPU (`device: "cuda"`)

## Advanced Usage

### Resume Training

```bash
python scripts/train.py --resume checkpoints/checkpoint_100.pt
```

### Custom Configuration

Create a new config file and use it:
```bash
python scripts/train.py --config my_config.yaml
```

### Batch Evaluation

Evaluate all checkpoints against random:
```bash
for ckpt in checkpoints/*.pt; do
    python scripts/evaluate.py --agent1 $ckpt --agent2 random --games 20
done
```

## Next Steps

1. **Start Training**: `python scripts/train.py --config config.yaml`
2. **Monitor Progress**: Check `checkpoints/` directory for saved models
3. **Evaluate**: Test against random baseline every 50-100 iterations
4. **Play**: Use the GUI to play against your trained agent
5. **Iterate**: Adjust hyperparameters based on results

## Technical Details

- **UCB Formula**: `Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))`
- **Policy Loss**: `-Σ π_mcts(a) * log(π_network(a))`
- **Value Loss**: `(z - v)²`
- **Data Augmentation**: 8x via rotations and flips
- **Optimizer**: Adam with weight decay (L2 regularization)
- **Gradient Clipping**: Norm clipping at 5.0

## References

- [AlphaGo Zero Paper](https://www.nature.com/articles/nature24270)
- [AlphaZero Paper](https://arxiv.org/abs/1712.01815)

---

**Happy Training!** 🎮🤖
