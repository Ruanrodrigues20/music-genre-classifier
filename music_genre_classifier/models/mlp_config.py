from dataclasses import dataclass
from typing import Tuple


@dataclass
class MLPConfig:
    hidden_layer_sizes: Tuple[int, ...] = (128, 64)
    activation: str = "relu"
    solver: str = "adam"
    max_iter: int = 800
    alpha: float = 0.02
    learning_rate_init: float = 0.001
    early_stopping: bool = True
    validation_fraction: float = 0.1
    n_iter_no_change: int = 20
    verbose: bool = True
    random_state: int = 42
    batch_size: int = 32
