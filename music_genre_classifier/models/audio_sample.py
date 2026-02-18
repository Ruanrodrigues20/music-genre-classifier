from dataclasses import dataclass
import numpy as np


@dataclass
class AudioSample:
    features: np.ndarray
    label: int
