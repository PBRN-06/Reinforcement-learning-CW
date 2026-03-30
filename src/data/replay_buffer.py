"""
Replay Buffer for AlphaZero Training

Stores and samples training examples from self-play games.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class ReplayBuffer:
    """
    Stores training examples with a sliding window.

    Keeps the most recent examples up to a maximum size.
    """

    def __init__(self, max_size: int = 500000):
        """
        Initialize replay buffer.

        Args:
            max_size: Maximum number of examples to store
        """
        self.max_size = max_size
        self.buffer = []

    def add_examples(self, examples: list):
        """
        Add new examples to buffer.

        Args:
            examples: List of dicts with 'state', 'policy', 'outcome'
        """
        self.buffer.extend(examples)

        # Keep only most recent examples
        if len(self.buffer) > self.max_size:
            self.buffer = self.buffer[-self.max_size:]

    def sample(self, batch_size: int):
        """
        Sample a random batch of examples.

        Args:
            batch_size: Number of examples to sample

        Returns:
            Dict with 'states', 'policies', 'outcomes' as numpy arrays
        """
        if len(self.buffer) == 0:
            return None

        # Sample random indices
        indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), replace=False)

        # Gather batch
        states = np.array([self.buffer[i]['state'] for i in indices])
        policies = np.array([self.buffer[i]['policy'] for i in indices])
        outcomes = np.array([self.buffer[i]['outcome'] for i in indices])

        return {
            'states': states,
            'policies': policies,
            'outcomes': outcomes
        }

    def get_dataset(self, recent_priority: float = 0.5):
        buffer_size = len(self.buffer)
        
        if buffer_size < 1000:
            return AlphaZeroDataset(self.buffer)
        
        recent_cutoff = int(buffer_size * 0.8)
        recent_examples = self.buffer[recent_cutoff:]
        old_examples = self.buffer[:recent_cutoff]
        
        # recent_priority controls what fraction of the dataset is recent examples
        # e.g. 0.5 means 50% recent, 50% old regardless of their actual sizes
        target_size = buffer_size
        num_recent = int(target_size * recent_priority)
        num_old = target_size - num_recent
        
        # Oversample recent, undersample old to hit target ratio
        recent_sampled = [recent_examples[i % len(recent_examples)] 
                        for i in range(num_recent)]
        old_sampled = [old_examples[i % len(old_examples)] 
                    for i in range(num_old)]
        
        combined = recent_sampled + old_sampled
        return AlphaZeroDataset(combined)

    def get_data_loader(self, batch_size: int, shuffle: bool = True, recent_priority: float = 0.5):
        dataset = self.get_dataset(recent_priority)  # pass through
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0
        )


    def clear(self):
        """Clear all examples from buffer."""
        self.buffer = []

    def __len__(self):
        """Get number of examples in buffer."""
        return len(self.buffer)

    def is_empty(self):
        """Check if buffer is empty."""
        return len(self.buffer) == 0


class AlphaZeroDataset(Dataset):
    """PyTorch Dataset wrapper for replay buffer."""

    def __init__(self, examples: list):
        """
        Initialize dataset.

        Args:
            examples: List of example dicts
        """
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        """
        Get single example.

        Returns:
            (state, policy, outcome) tuple
        """
        example = self.examples[idx]
        state = example['state'].astype(np.float32)
        policy = example['policy'].astype(np.float32)
        outcome = np.array([example['outcome']], dtype=np.float32)

        return state, policy, outcome
