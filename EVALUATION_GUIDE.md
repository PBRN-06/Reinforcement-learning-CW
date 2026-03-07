# AlphaZero Evaluation Guide

This guide explains how to monitor and evaluate your AlphaZero training progress.

## Overview

The training system now automatically logs metrics, and you can evaluate progress at any time using the evaluation scripts.

---

## Automatic Metrics Logging

During training, the system automatically logs:
- **Policy Loss**: How well the network predicts MCTS policy
- **Value Loss**: How accurately the network predicts game outcomes
- **Total Loss**: Combined loss (policy + value)

These are saved to: `logs/training_metrics.json`

---

## Evaluation Scripts

### 1. **Quick Loss Curve Visualization**

View training loss curves without running evaluations:

```bash
python scripts/evaluate_progress.py --plot-only
```

**Output**: `logs/loss_curves.png` with 3 plots (policy, value, total loss)

---

### 2. **Full Progress Evaluation**

Evaluate ALL checkpoints against random agent:

```bash
# Fast mode (recommended) - 100 simulations with GPU batching
python scripts/evaluate_progress.py

# Accurate mode - 400 simulations
python scripts/evaluate_progress.py --simulations 400
```

This will:
- Load all checkpoints in `./checkpoints/`
- Play 20 games vs random agent for each checkpoint
- Generate loss curves, win rate progression, and a text report
- **Uses batched MCTS by default** for 2-4x GPU speedup!

**Note**: Default is 100 simulations with batch_size=16 for speed. To disable batching (slower): `--batch-size 1`

**Output**:
- `logs/loss_curves.png` - Training loss over time
- `logs/win_rate_progression.png` - Win rate vs random over iterations
- `logs/evaluation_report.txt` - Detailed text summary

---

### 3. **Evaluate Specific Checkpoints**

Evaluate only certain iterations (faster):

```bash
# Evaluate iterations 10, 20, 30, 40, 50
python scripts/evaluate_progress.py --checkpoints 10 20 30 40 50
```

---

### 4. **Custom Evaluation Settings**

Adjust number of games and MCTS simulations:

```bash
# More games = more accurate win rate estimate (but slower)
python scripts/evaluate_progress.py --games 50

# Fewer simulations = faster evaluation (but weaker play)
python scripts/evaluate_progress.py --simulations 200

# Combine options
python scripts/evaluate_progress.py --checkpoints 20 30 40 --games 50 --simulations 800
```

---

### 5. **Head-to-Head Checkpoint Comparison**

Compare two specific checkpoints against each other:

```bash
# Compare iteration 50 vs iteration 30
python scripts/evaluate.py \
    --agent1 checkpoints/checkpoint_50.pt \
    --agent2 checkpoints/checkpoint_30.pt \
    --games 20

# Compare against random baseline
python scripts/evaluate.py \
    --agent1 checkpoints/checkpoint_50.pt \
    --agent2 random \
    --games 50
```

---

## Expected Learning Timeline

Based on your current configuration (9x9 board, 400 MCTS sims, 10 random opening moves):

| Iteration | Expected Behavior | Win Rate vs Random |
|-----------|-------------------|--------------------|
| **1-20**  | Random chaos, no visible strategy | ~50-60% |
| **20-50** | **First patterns emerge**, blocks obvious threats | ~70-85% |
| **50-100** | Basic tactics, creates simple forks | ~85-95% |
| **100-300** | Competent play, multi-move planning | ~95-99% |
| **300+** | Strong play with opening theory | ~99%+ |

**You should see "some kind of gameplay" around iteration 30-50.**

---

## Monitoring During Training

### Option 1: Check Loss Curves (Fast)

While training is running, in another terminal:

```bash
# Just visualize current loss curves
python scripts/evaluate_progress.py --plot-only
```

Then open: `logs/loss_curves.png`

**What to look for:**
- ✅ Losses steadily decreasing = learning is working
- ❌ Flat or increasing losses = potential issue

---

### Option 2: Quick Evaluation (Medium)

Evaluate latest checkpoint only:

```bash
# Find latest iteration number first
ls checkpoints/

# Evaluate it (e.g., iteration 30)
python scripts/evaluate_progress.py --checkpoints 30 --games 20
```

---

### Option 3: Full Progress Report (Slow)

Full evaluation of all checkpoints (useful every 50-100 iterations):

```bash
python scripts/evaluate_progress.py --games 50
```

This gives you the complete picture but takes longer.

---

## Interpreting Results

### Loss Curves (`logs/loss_curves.png`)

**Policy Loss**:
- Should decrease from ~4-6 to ~2-3 over first 100 iterations
- Measures how well network predicts MCTS search results

**Value Loss**:
- Should decrease from ~0.5-1.0 to ~0.2-0.4
- Measures how well network predicts game outcomes

**Total Loss**:
- Combined metric, should steadily decrease
- Plateaus are normal after ~300 iterations

---

### Win Rate Progression (`logs/win_rate_progression.png`)

**Milestones**:
- **60% win rate**: Network showing signs of learning
- **70% win rate**: Basic pattern recognition (threat detection)
- **85% win rate**: Competent tactical play
- **95% win rate**: Strong play, rarely loses to random

**Stacked areas** show win/draw/loss distribution over time.

---

### Evaluation Report (`logs/evaluation_report.txt`)

Text summary with:
- Training statistics (iterations, games, latest losses)
- Win rates for each evaluated checkpoint
- Learning milestones achieved

---

## Troubleshooting

### "No training metrics found"
- You haven't started training yet
- Or `logs/training_metrics.json` was deleted
- Solution: Start training with `python scripts/train.py`

### "No checkpoints found"
- No checkpoint files in `./checkpoints/`
- Solution: Train for at least 10 iterations (checkpoints saved every 10)

### Win rate not improving after 50 iterations
Possible causes:
1. Network too small (increase `num_res_blocks` from 6 to 10)
2. Not enough MCTS simulations (increase from 400 to 800)
3. Too many random opening moves (reduce from 10 to 5)
4. Learning rate too high/low (try 5e-4 or 2e-3 instead of 1e-3)

### Out of memory during evaluation
- Reduce `--simulations` (e.g., 200 instead of 400)
- Reduce `--games` (e.g., 10 instead of 20)

---

## Quick Reference

```bash
# During training - check loss curves
python scripts/evaluate_progress.py --plot-only

# After training - full evaluation
python scripts/evaluate_progress.py

# Evaluate specific milestones
python scripts/evaluate_progress.py --checkpoints 20 30 40 50

# Quick check of latest checkpoint
python scripts/evaluate_progress.py --checkpoints 30 --games 10

# Deep evaluation with more games
python scripts/evaluate_progress.py --games 100 --simulations 800

# Compare two checkpoints head-to-head
python scripts/evaluate.py \
    --agent1 checkpoints/checkpoint_50.pt \
    --agent2 checkpoints/checkpoint_30.pt \
    --games 50
```

---

## Advanced: Programmatic Access

You can also load metrics in your own scripts:

```python
import json

# Load training metrics
with open('logs/training_metrics.json', 'r') as f:
    metrics = json.load(f)

# Get latest iteration info
latest = metrics[-1]
print(f"Iteration: {latest['iteration']}")
print(f"Policy Loss: {latest['policy_loss']:.4f}")
print(f"Value Loss: {latest['value_loss']:.4f}")

# Plot custom metrics
import matplotlib.pyplot as plt
iterations = [m['iteration'] for m in metrics]
policy_loss = [m['policy_loss'] for m in metrics]
plt.plot(iterations, policy_loss)
plt.show()
```

---

## Summary

**Recommended workflow:**

1. **Start training**: `python scripts/train.py`

2. **Monitor every 10 iterations**: `python scripts/evaluate_progress.py --plot-only`

3. **First evaluation at iteration 30**: `python scripts/evaluate_progress.py --checkpoints 30`

4. **Full evaluation at iteration 50**: `python scripts/evaluate_progress.py`

5. **Check for gameplay signs**: If win rate > 70% at iteration 50, you're on track!

6. **Continue monitoring every 50 iterations** as training progresses

Good luck with your training! 🚀
