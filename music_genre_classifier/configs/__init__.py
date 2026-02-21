from music_genre_classifier.configs.logger import get_logger
from music_genre_classifier.configs.path import (
    DATASET_DIR,
    DATASET_CSV,
    MODEL_PATH,
    DATA_DIR,
    RESULTS_DIR,
)
from music_genre_classifier.configs.musics import (
    SAMPLE_RATE,
    NUM_SEGMENTS,
    SEGMENT_DURATION,
)

__all__ = [
    "get_logger",
    "DATASET_DIR",
    "SAMPLE_RATE",
    "DATASET_CSV",
    "MODEL_PATH",
    "DATA_DIR",
    "RESULTS_DIR",
    "NUM_SEGMENTS",
    "SEGMENT_DURATION",
]
