import io
from pathlib import Path
from typing import List

import librosa
import numpy as np

from music_genre_classifier.models import AudioSample, GenreType
from music_genre_classifier.configs import (
    get_logger,
    DATASET_DIR,
    SAMPLE_RATE,
    NUM_SEGMENTS,
    SEGMENT_DURATION,
)
from music_genre_classifier.data.feature_extractor import FeatureExtractor
from music_genre_classifier.utils import extract_number

logger = get_logger(__name__)


class AudioLoader:
    @staticmethod
    def load_dataset() -> List[AudioSample]:

        logger.info("📂 Loading dataset")

        samples: List[AudioSample] = []

        for genre in GenreType:
            genre_name = GenreType.get_name(genre)
            genre_dir = DATASET_DIR / genre_name

            logger.info("📂 Loading [%s]", genre_name)

            if not genre_dir.exists():
                logger.warning("⚠️ Directory not found: %s", genre_dir)
                continue

            samples.extend(AudioLoader._load_genre_dir(genre_dir, genre))

        logger.info("Total [%s]: samples", len(samples))
        return samples

    @staticmethod
    def _load_genre_dir(genre_dir: Path, genre: GenreType) -> List[AudioSample]:
        samples: List[AudioSample] = []

        for audio_file in sorted(genre_dir.glob("*.wav"), key=extract_number):
            logger.info("🎵 Extracting features | genre=%s | file=%s", GenreType.get_name(genre), audio_file.name)

            try:
                y, sr = librosa.load(str(audio_file), sr=SAMPLE_RATE, mono=True)
                features = FeatureExtractor.extract(y, sr)

                if features is None:
                    logger.warning("⚠️ Feature is None: %s", audio_file)
                    continue

                label = genre
                samples.append(
                    AudioSample(
                        filename=audio_file.name, features=features, label=label
                    )
                )

            except Exception as e:
                logger.error("❌ Failed processing %s: %s", audio_file, e)

        return samples

    @staticmethod
    def load_audio(audio_bytes: bytes) -> np.ndarray | None:
        try:
            y, sr = librosa.load(io.BytesIO(audio_bytes), sr=SAMPLE_RATE, mono=True)

            min_duration = SEGMENT_DURATION * NUM_SEGMENTS
            duration = librosa.get_duration(y=y, sr=sr)

            if duration < min_duration:
                logger.warning("⚠️ Audio too short: %.2fs", duration)
                return None

            segment_samples = int(SEGMENT_DURATION * sr)

            starts_sec = [
                0,
                duration * 0.25,
                duration * 0.50,
                duration * 0.75,
                duration - SEGMENT_DURATION,
            ]

            segments = []
            for start_sec in starts_sec:
                start_sample = int(start_sec * sr)
                end_sample = start_sample + segment_samples
                segments.append(y[start_sample:end_sample])

            full_audio = np.concatenate(segments)

            return full_audio

        except Exception as e:
            logger.error("❌ Failed loading audio from bytes: %s", e)
            return None
