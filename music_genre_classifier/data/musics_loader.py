from typing import List

import numpy as np

from music_genre_classifier.models import AudioSample, GenreType
from music_genre_classifier.configs import get_logger, DATASET_DIR
from music_genre_classifier.data.audio_loader import AudioLoader
from music_genre_classifier.data.feature_extractor import FeatureExtractor

logger = get_logger(__name__)


class MusicsLoader:
    @staticmethod
    def load_dataset_train() -> List[AudioSample]:
        logger.info(DATASET_DIR)
        return MusicsLoader.__load_dataset("train")

    @staticmethod
    def load_dataset_test() -> List[AudioSample]:
        return MusicsLoader.__load_dataset("test")

    @staticmethod
    def __load_dataset(split: str) -> List[AudioSample]:
        if not split:
            raise ValueError("Split cannot be empty")

        samples: List[AudioSample] = []
        logger.info(f"📂 Loading dataset [{split}]")

        for genre in GenreType:
            genre_dir = DATASET_DIR / genre.value / split

            if not genre_dir.exists():
                logger.warning(f"⚠️ Directory not found: {genre_dir}")
                continue

            for audio_file in genre_dir.glob("*.wav"):
                logger.debug(f"🎵 Processing: {audio_file.name}")

                y, sr = AudioLoader.from_path(audio_file)
                features = FeatureExtractor.extract(y, sr)

                if features is None:
                    logger.error(f"Feature is None {audio_file}")

                label = GenreType.to_label(genre)
                samples.append(AudioSample(features, label))

        logger.info(f"Total [{split}]: {len(samples)} samples")
        return samples

    def load_audio(audio: bytes) -> np.ndarray | None:
        y, sr = AudioLoader.from_bytes(audio)
        return FeatureExtractor.extract(y, sr)
