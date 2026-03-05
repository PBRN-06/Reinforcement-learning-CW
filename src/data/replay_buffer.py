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

    def get_dataset(self):
        """
        Get PyTorch Dataset for DataLoader.

        Returns:
            AlphaZeroDataset instance
        """
        return AlphaZeroDataset(self.buffer)

    def get_data_loader(self, batch_size: int, shuffle: bool = True):
        """
        Create PyTorch DataLoader.

        Args:
            batch_size: Batch size for training
            shuffle: Whether to shuffle data

        Returns:
            DataLoader instance
        """
        dataset = self.get_dataset()
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

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
