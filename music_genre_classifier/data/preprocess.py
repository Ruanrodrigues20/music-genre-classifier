from pathlib import Path
import random
import shutil

import librosa
import numpy as np
import soundfile as sf

from music_genre_classifier.models import GenreType
from music_genre_classifier.configs import get_logger, DATASET_DIR

logger = get_logger(__name__)


class Preprocess:
    SEGMENT_DURATION = 6
    NUM_SEGMENTS = 5
    SAMPLE_RATE = 22050

    def __init__(self, genre: str):
        self.genre = genre
        self.music_dir = Path(DATASET_DIR) / self.genre
        self.test_dir = self.music_dir / "test"
        self.train_dir = self.music_dir / "train"

        logger.info("Preprocess initialized for genre: %s", self.genre)
        logger.info("Looking for dataset in: %s", self.music_dir)

    def create_directories(self):
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.train_dir.mkdir(parents=True, exist_ok=True)

    def has_wav_files(self, dir_path: Path):
        return any(f.suffix == ".wav" for f in dir_path.iterdir())

    def split_dataset(self):
        if self.has_wav_files(self.test_dir) or self.has_wav_files(self.train_dir):
            logger.info("WAV files already exist, skipping split.")
            return

        logger.info("Shuffling and splitting dataset...")
        files = [f for f in self.music_dir.iterdir() if f.suffix == ".mp3"]
        random.shuffle(files)

        if len(files) < 200:
            logger.error("Need at least 200 MP3 files, found %d", len(files))
            raise SystemExit(1)

        for f in files[:50]:
            shutil.move(str(f), str(self.test_dir / f.name))

        for f in files[50:200]:
            shutil.move(str(f), str(self.train_dir / f.name))

        logger.info("Dataset split completed")

    def extract_30s_audio(self, audio_path: Path, out_dir: Path):
        if audio_path.suffix == ".wav":
            logger.info("WAV file detected, reusing: %s", audio_path)
        else:
            logger.info("Converting MP3 to WAV and extracting 30s: %s", audio_path)

        y, sr = librosa.load(str(audio_path), sr=self.SAMPLE_RATE, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)

        min_duration = self.SEGMENT_DURATION * self.NUM_SEGMENTS
        if duration < min_duration:
            logger.warning("Skipping %s (too short)", audio_path)
            return

        segment_samples = int(self.SEGMENT_DURATION * sr)
        starts_sec = [
            0,
            duration * 0.25,
            duration * 0.50,
            duration * 0.75,
            duration - self.SEGMENT_DURATION,
        ]

        segments = []
        for start_sec in starts_sec:
            start_sample = int(start_sec * sr)
            end_sample = start_sample + segment_samples
            segments.append(y[start_sample:end_sample])

        full_audio = np.concatenate(segments)

        out_file = out_dir / (audio_path.stem + ".wav")
        sf.write(str(out_file), full_audio, sr)
        logger.info("Saved 30s audio: %s", out_file)

        if audio_path.suffix == ".mp3":
            audio_path.unlink()

    def process_directory(self, dir_path: Path):
        for f in dir_path.iterdir():
            if f.suffix in [".mp3", ".wav"]:
                self.extract_30s_audio(f, dir_path)

    def rename(self):
        def rename_path(path: Path):
            for i, f in enumerate(path.iterdir()):
                if f.suffix == ".wav":
                    new_name = f"{self.genre}{i}.wav"
                    f.rename(path / new_name)

        rename_path(self.test_dir)
        rename_path(self.train_dir)

    def run(self):
        logger.info("🚀 Starting preprocessing pipeline")

        mp3_files = list(self.train_dir.glob("*.mp3")) + list(
            self.music_dir.glob("*.mp3")
        )
        if not mp3_files:
            return

        self.create_directories()
        self.split_dataset()

        logger.info("Processing test files")
        self.process_directory(self.test_dir)

        logger.info("Processing train files")
        self.process_directory(self.train_dir)

        logger.info("Rename")
        self.rename()

        logger.info("🎉 Preprocessing completed for genre: %s", self.genre)
