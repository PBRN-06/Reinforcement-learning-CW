"""
AlphaZero Configuration Module

Defines hyperparameters and settings for the AlphaZero training pipeline.
"""

from dataclasses import dataclass, asdict
import yaml
import torch


@dataclass
class AlphaZeroConfig:
    """Configuration for AlphaZero training pipeline."""

    # Network architecture
    num_res_blocks: int = 6
    num_filters: int = 64

    # MCTS parameters
    num_simulations: int = 400
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature_threshold: int = 15  # Move count to switch temperature

    # Self-play
    games_per_iteration: int = 100
    num_workers: int = 4

    # Training
    num_iterations: int = 1000
    epochs_per_iteration: int = 10
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 5.0

    # Replay buffer
    replay_buffer_size: int = 500000

    # Checkpointing
    checkpoint_freq: int = 10
    checkpoint_dir: str = "./checkpoints"

    # Evaluation
    eval_freq: int = 50
    eval_games: int = 20

    # Logging
    log_dir: str = "./logs"

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Board settings
    board_size: int = 9
    win_length: int = 5

    @classmethod
    def from_yaml(cls, path: str):
        """Load config from YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def to_yaml(self, path: str):
        """Save config to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    def __post_init__(self):
        """Validate configuration after initialization."""
        assert self.num_res_blocks >= 1, "Need at least 1 residual block"
        assert self.num_filters >= 16, "Need at least 16 filters"
        assert self.num_simulations >= 1, "Need at least 1 MCTS simulation"
        assert self.batch_size >= 1, "Batch size must be positive"
        assert self.learning_rate > 0, "Learning rate must be positive"
        assert 0 < self.dirichlet_epsilon < 1, "Dirichlet epsilon must be in (0, 1)"
