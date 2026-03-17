import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import torch

from src.data.replay_buffer import ReplayBuffer

def visualise_training_positions(replay_buffer, num_samples=9):
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    
    indices = np.random.choice(len(replay_buffer.buffer), num_samples)
    
    for idx, ax in zip(indices, axes.flat):
        example = replay_buffer.buffer[idx]
        state = example['state']
        policy = example['policy'].reshape(9, 9)
        
        # Draw board
        ax.imshow(np.zeros((9, 9)), cmap='YlOrBr', vmin=0, vmax=2)
        
        # Draw stones
        for r in range(9):
            for c in range(9):
                if state[r, c] == 1:
                    circle = Circle((c, r), 0.4, color='black')
                    ax.add_patch(circle)
                elif state[r, c] == 2:
                    circle = Circle((c, r), 0.4, color='white', 
                                      ec='black', linewidth=2)
                    ax.add_patch(circle)
        
        # Overlay policy as heatmap
        policy_overlay = ax.imshow(policy, cmap='hot', alpha=0.5, 
                                   vmin=0, vmax=policy.max())
        
        p1 = int((state == 1).sum())
        p2 = int((state == 2).sum())
        outcome = example['outcome']
        ax.set_title(f"P1:{p1} P2:{p2} outcome:{outcome:.1f}\n"
                    f"max_policy:{policy.max():.3f}")
        ax.set_xlim(-0.5, 8.5)
        ax.set_ylim(-0.5, 8.5)
    
    plt.tight_layout()
    plt.savefig('training_positions.png', dpi=100)
    plt.show()
    print("Saved to training_positions.png")

replay_buffer = ReplayBuffer()
buffer_path = os.path.join("checkpoints", "replay_buffer.pt")
if os.path.exists(buffer_path):
    replay_buffer.buffer = torch.load(buffer_path, weights_only=False)
    print(f"  Restored replay buffer: {len(replay_buffer)} examples")
else:
    print(f"  No replay buffer found at {buffer_path} - will regenerate")
visualise_training_positions(replay_buffer)