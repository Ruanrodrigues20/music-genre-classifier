from music_genre_classifier.configs.logger import get_logger
from music_genre_classifier.configs.path import (
    DATASET_DIR,
    TEST_CSV,
    TRAIN_CSV,
    MODEL_PATH,
    DATA_DIR,
    RESULTS_DIR,
)
from music_genre_classifier.configs.musics import SR

__all__ = [
    "get_logger",
    "DATASET_DIR",
    "SR",
    "TEST_CSV",
    "TRAIN_CSV",
    "MODEL_PATH",
    "DATA_DIR",
    "RESULTS_DIR",
]
