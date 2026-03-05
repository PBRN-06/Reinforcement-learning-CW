# AlphaZero Quick Start Guide

## Installation

1. **Install dependencies** (this may take a few minutes):
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify installation**:
   ```bash
   python -c "import torch; print(f'PyTorch {torch.__version__} installed successfully')"
   ```

## Usage

### Option 1: Start Training Immediately

Train from scratch with default settings:
```bash
python scripts/train.py --config config.yaml
```

This will:
- Generate 100 self-play games per iteration
- Run 400 MCTS simulations per move
- Train for 10 epochs on collected data
- Save checkpoints every 10 iterations to `checkpoints/`
- Continue for 1000 iterations (or until you stop it)

**Tip**: Training takes time! Start with fewer iterations to test:
```bash
python scripts/train.py --config config.yaml --iterations 10
```

### Option 2: Quick Test (CPU-only)

If you don't have a GPU, test on CPU (slower but works):
```bash
python scripts/train.py --config config.yaml --device cpu --iterations 5
```

### Option 3: Evaluate Random vs Random (Sanity Check)

Test that everything works:
```bash
python scripts/evaluate.py --agent1 random --agent2 random --games 5
```

This should show roughly 50/50 wins between two random agents.

## After Training

### 1. Evaluate Your Model

After training for some iterations, evaluate against random:
```bash
python scripts/evaluate.py \
    --agent1 checkpoints/checkpoint_10.pt \
    --agent2 random \
    --games 20
```

You should see your trained agent winning more than 50% after even just 10 iterations!

### 2. Play Against Your Agent

```bash
python scripts/play_gui.py --checkpoint checkpoints/checkpoint_10.pt
```

Click on the board to make your moves. The AI will automatically respond.

### 3. Watch It Get Better

As you train for more iterations, the agent gets stronger:
```bash
# Compare iteration 10 vs iteration 100
python scripts/evaluate.py \
    --agent1 checkpoints/checkpoint_100.pt \
    --agent2 checkpoints/checkpoint_10.pt \
    --games 20
```

## Expected Results

- **After 10 iterations**: Beats random ~60-70% of the time
- **After 50 iterations**: Beats random >90% of the time
- **After 100 iterations**: Strong intermediate player
- **After 500+ iterations**: Expert level

## Troubleshooting

### "CUDA out of memory"
Reduce batch size in `config.yaml`:
```yaml
batch_size: 128  # Instead of 256
```

### Training is too slow
Reduce simulations or games per iteration:
```yaml
num_simulations: 200  # Instead of 400
games_per_iteration: 50  # Instead of 100
```

### Want faster testing
Use fewer simulations during play:
```bash
python scripts/play_gui.py --checkpoint checkpoints/checkpoint_10.pt --simulations 100
```

## Next Steps

1. ✅ Install dependencies: `pip install -r requirements.txt`
2. ✅ Start training: `python scripts/train.py --config config.yaml --iterations 10`
3. ✅ Evaluate: `python scripts/evaluate.py --agent1 checkpoints/checkpoint_10.pt --agent2 random --games 20`
4. ✅ Play: `python scripts/play_gui.py --checkpoint checkpoints/checkpoint_10.pt`
5. ✅ Train more: Resume with `python scripts/train.py --resume checkpoints/checkpoint_10.pt`

For more details, see `ALPHAZERO_README.md`.

---

**Pro Tip**: Leave training running overnight to get a strong agent! 🚀
