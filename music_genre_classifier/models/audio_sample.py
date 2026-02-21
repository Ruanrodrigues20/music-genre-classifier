import numpy as np
from dataclasses import dataclass
from music_genre_classifier.models.genre_type import GenreType


@dataclass
class AudioSample:
    filename: str
    features: np.ndarray
    label: GenreType
