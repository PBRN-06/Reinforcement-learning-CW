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
    mcts_batch_size: int = 16  # Batch size for parallel MCTS evaluation (GPU optimization)
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature_threshold: int = 15  # Move count to switch from high to low temperature
    temperature_high: float = 1.5  # High temperature for exploration (adaptive decay)
    temperature_low: float = 0.1  # Low temperature for exploitation

    # Self-play
    games_per_iteration: int = 100
    num_workers: int = 4
    generation_frequency: int = 1  # Generate new games every N iterations (1=every iteration, 5=every 5th)
    random_opening_moves: int = 0  # kept for backward compatability

    # Opponent selection strategy with adaptive rolloff
    # Ratios adapt based on checkpoint pool size:
    # - No checkpoints: 70% self-play, 30% random
    # - Small pool (<5): 40% self-play, 30% pool, 30% random
    # - Full pool (≥5): 20% self-play, 60% pool, 20% random
    pool_size_max: int = 15  # Maximum number of checkpoints in pool

    # Deprecated (kept for backward compatibility)
    past_opponent_ratio: float = 0.0  # Legacy parameter

    # Training
    num_iterations: int = 1000
    epochs_per_iteration: int = 10
    batch_size: int = 256
    recent_priority: float = 0.5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    reward_shaping_weight: float = 0.0  # kept for backward compatibility

    # Replay buffer
    replay_buffer_size: int = 500000

    # Checkpointing
    checkpoint_freq: int = 10 # Kept for backward compat.
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
        assert self.mcts_batch_size >= 1, "MCTS batch size must be positive"
        assert self.mcts_batch_size <= self.num_simulations, "MCTS batch size cannot exceed num_simulations"
        assert self.batch_size >= 1, "Batch size must be positive"
        assert self.learning_rate > 0, "Learning rate must be positive"
        assert 0 < self.dirichlet_epsilon < 1, "Dirichlet epsilon must be in (0, 1)"
