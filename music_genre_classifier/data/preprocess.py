from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from music_genre_classifier.configs import (
    DATASET_DIR,
    NUM_SEGMENTS,
    SAMPLE_RATE,
    SEGMENT_DURATION,
    get_logger,
)

logger = get_logger(__name__)


class Preprocess:
    def __init__(self, genre: str):
        self.genre = genre
        self.music_dir = Path(DATASET_DIR) / self.genre

        logger.info("Preprocess initialized for genre: %s", self.genre)
        logger.info("Looking for dataset in: %s", self.music_dir)

    def extract_30s_audio(self, audio_path: Path):
        logger.info("Processing: %s", audio_path)

        y, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)

        min_duration = SEGMENT_DURATION * NUM_SEGMENTS
        if duration < min_duration:
            logger.warning("Skipping %s (too short)", audio_path)
            return

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

        out_file = self.music_dir / f"{audio_path.stem}_30s.wav"
        sf.write(str(out_file), full_audio, sr)
        logger.info("Saved: %s", out_file)

    def process_directory(self):
        for f in self.music_dir.iterdir():
            if f.suffix == ".mp3":
                self.extract_30s_audio(f)

    def rename(self):
        for i, f in enumerate(sorted(self.music_dir.glob("*_30s.wav"))):
            new_name = f"{self.genre}{i}.wav"
            f.rename(self.music_dir / new_name)

    def run(self):
        logger.info("🚀 Starting preprocessing pipeline")

        mp3_files = list(self.music_dir.glob("*.mp3"))
        if not mp3_files:
            return

        logger.info("Processing music files")
        self.process_directory()

        logger.info("Rename")
        self.rename()

        logger.info("🎉 Preprocessing completed for genre: %s", self.genre)
