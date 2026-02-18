from dataclasses import dataclass
from typing import Tuple


@dataclass
class MLPConfig:
    hidden_layer_sizes: Tuple[int, ...] = (128, 64, 32)
    activation: str = "relu"
    solver: str = "adam"
    max_iter: int = 500
    alpha: float = 0.001
    early_stopping: bool = True
    validation_fraction: float = 0.1
    n_iter_no_change: int = 15
    verbose: bool = True
    random_state: int = 42
