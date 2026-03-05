# Google Colab Training Guide

Get 5-10x speedup by training on free Google Colab GPUs!

## Quick Start (3 steps)

### 1. Prepare Your Code

Create a zip file of your project:

```bash
cd /home/hsken/dev
zip -r alphazero-gomoku.zip Reinforcement-learning-CW/ \
    -x "*.git*" "*__pycache__*" "*.pyc" "*/checkpoints/*" "*/logs/*" "*/training_data/*"
```

This creates `alphazero-gomoku.zip` (~100 KB without checkpoints).

### 2. Upload to Google Colab

**Method A: Google Drive (Recommended)**

1. Upload `alphazero-gomoku.zip` to your Google Drive (drag & drop into drive.google.com)
2. Open the Colab notebook: Upload `AlphaZero_Gomoku_Colab.ipynb` to Colab
   - Or create new notebook at https://colab.research.google.com and copy/paste cells
3. **Enable GPU**: Runtime → Change runtime type → Hardware accelerator → **GPU (T4)**
4. Run all cells!

**Method B: GitHub**

1. Push your code to GitHub:
   ```bash
   git push origin alpha_zero
   ```
2. In Colab, use the GitHub clone cell instead of Drive mount

### 3. Start Training

The notebook will:
- Mount Google Drive
- Extract your code
- Install dependencies
- Verify GPU is enabled
- Start training!

Look for: `Device: cuda` ✓

## Expected Performance

| Hardware | Time per Iteration (100 games) | Cost |
|----------|-------------------------------|------|
| **Local CPU** | 1-3 hours | Free |
| **Colab Free GPU (T4)** | 5-15 minutes | Free* |
| **Colab Pro GPU (V100/A100)** | 2-8 minutes | $10/month |

*Free tier has usage limits (~12 hour sessions, daily quotas)

## Colab Notebook Workflow

### Training
```python
# Fresh training
!python scripts/train.py --config config.yaml --iterations 50
```

### Evaluation
```python
# Check progress
!python scripts/evaluate.py --agent1 checkpoints/checkpoint_10.pt --agent2 random --games 20
```

### Save Results
```python
# Backup to Google Drive (IMPORTANT!)
!cp -r checkpoints/* /content/drive/MyDrive/alphazero-checkpoints/
```

## Important Tips

### 1. Save Your Work!
Colab sessions are **temporary**. Always backup checkpoints to Google Drive:
```python
!cp -r checkpoints/* /content/drive/MyDrive/alphazero-checkpoints/
```

### 2. Resume Training After Disconnect
```python
# Restore checkpoints from Drive
!cp /content/drive/MyDrive/alphazero-checkpoints/*.pt checkpoints/

# Resume training
!python scripts/train.py --resume checkpoints/checkpoint_50.pt --iterations 100
```

### 3. Monitor GPU Usage
```python
import torch
print(f"GPU Memory: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
```

### 4. Adjust for Faster Testing
```python
# Edit config for quicker iterations
!sed -i 's/num_simulations: 400/num_simulations: 200/' config.yaml
!sed -i 's/games_per_iteration: 100/games_per_iteration: 50/' config.yaml
```

## Troubleshooting

### "Runtime disconnected"
- Colab free tier has session limits
- Solution: Save checkpoints frequently, resume from last checkpoint
- Consider Colab Pro for longer sessions

### "CUDA out of memory"
```python
# Reduce batch size
!sed -i 's/batch_size: 256/batch_size: 128/' config.yaml
```

### "No GPU detected"
1. Runtime → Change runtime type
2. Hardware accelerator → **GPU**
3. Click Save
4. Restart runtime

### Slow training even with GPU
- Verify: `!nvidia-smi` shows GPU usage
- Check: Training logs show `Device: cuda`
- If using T4: 5-15 min/iteration is normal

## Training Strategy for Colab

**Quick Test (1-2 hours):**
```python
!python scripts/train.py --config config.yaml --iterations 10
```

**Half Day Session (6-8 hours):**
```python
!python scripts/train.py --config config.yaml --iterations 50
```

**Full Session (12 hours max):**
```python
!python scripts/train.py --config config.yaml --iterations 100
```

Then resume the next day!

## Download Trained Models

### From Google Drive
After saving to Drive, download from drive.google.com

### Direct Download in Colab
```python
from google.colab import files
files.download('checkpoints/checkpoint_50.pt')
```

## Using Trained Models Locally

After downloading checkpoints from Colab:

```bash
# Copy checkpoint to local machine
cp ~/Downloads/checkpoint_50.pt /home/hsken/dev/Reinforcement-learning-CW/checkpoints/

# Play against it
python scripts/play_gui.py --checkpoint checkpoints/checkpoint_50.pt

# Evaluate it
python scripts/evaluate.py --agent1 checkpoints/checkpoint_50.pt --agent2 random --games 20
```

## Colab Pro Benefits

**Free vs Pro:**

| Feature | Free | Pro ($10/mo) |
|---------|------|-------------|
| Session length | ~12 hours | ~24 hours |
| GPU types | T4 | T4, V100, A100 |
| Usage limits | Daily quotas | No limits |
| Background execution | ❌ | ✓ |

**Worth it if:**
- Training more than 100 iterations
- Want faster GPUs (V100/A100)
- Need longer sessions

## Next Steps

1. ✅ Create zip: `zip -r alphazero-gomoku.zip Reinforcement-learning-CW/`
2. ✅ Upload to Google Drive
3. ✅ Open `AlphaZero_Gomoku_Colab.ipynb` in Colab
4. ✅ Enable GPU runtime
5. ✅ Run cells and start training!
6. ✅ Backup checkpoints to Drive regularly
7. ✅ Download and play against your trained agent!

Happy training! 🚀
