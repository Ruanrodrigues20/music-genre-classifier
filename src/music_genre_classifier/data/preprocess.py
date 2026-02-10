import os
import random
import shutil

import librosa
import numpy as np
import soundfile as sf

from music_genre_classifier.model import GenreType
from music_genre_classifier.config.logger import get_logger

logger = get_logger(__name__)


class Preprocess:
    SEGMENT_DURATION = 6  # seconds
    NUM_SEGMENTS = 5
    SAMPLE_RATE = 22050

    def __init__(self, genre: GenreType):
        self.genre = genre.value

        src_dir = os.path.dirname(
            os.path.abspath(__file__)
        )  # src/music_genre_classifier/data
        project_root = os.path.abspath(os.path.join(src_dir, "..", "..", ".."))
        self.music_dir = os.path.join(project_root, "dataset", self.genre)
        self.test_dir = os.path.join(self.music_dir, "teste")
        self.train_dir = os.path.join(self.music_dir, "treinamento")

        logger.info("Preprocess initialized for genre: %s", self.genre)
        logger.info("Looking for dataset in: %s", self.music_dir)

    def create_directories(self):
        os.makedirs(self.test_dir, exist_ok=True)
        os.makedirs(self.train_dir, exist_ok=True)

    def has_wav_files(self, dir_path):
        return any(f.endswith(".wav") for f in os.listdir(dir_path))

    def split_dataset(self):
        if self.has_wav_files(self.test_dir) or self.has_wav_files(self.train_dir):
            logger.info("WAV files already exist, skipping split.")
            return

        logger.info("Shuffling and splitting dataset...")
        files = [f for f in os.listdir(self.music_dir) if f.endswith(".mp3")]
        random.shuffle(files)

        if len(files) < 200:
            logger.error("Need at least 200 MP3 files, found %d", len(files))
            raise SystemExit(1)

        for f in files[:50]:
            shutil.move(os.path.join(self.music_dir, f), os.path.join(self.test_dir, f))

        for f in files[50:200]:
            shutil.move(
                os.path.join(self.music_dir, f), os.path.join(self.train_dir, f)
            )

        logger.info("Dataset split completed")

    def extract_30s_audio(self, audio_path, out_dir):
        if audio_path.endswith(".wav"):
            logger.info("WAV file detected, reusing: %s", audio_path)
        else:
            logger.info("Converting MP3 to WAV and extracting 30s: %s", audio_path)

        y, sr = librosa.load(audio_path, sr=self.SAMPLE_RATE, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)

        if duration < self.SEGMENT_DURATION * self.NUM_SEGMENTS:
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

        out_file = os.path.join(
            out_dir, os.path.splitext(os.path.basename(audio_path))[0] + ".wav"
        )
        sf.write(out_file, full_audio, sr)
        logger.info("Saved 30s audio: %s", out_file)

        if audio_path.endswith(".mp3"):
            os.remove(audio_path)

    def process_directory(self, dir_path):
        for f in os.listdir(dir_path):
            if f.endswith(".mp3") or f.endswith(".wav"):
                file_path = os.path.join(dir_path, f)
                self.extract_30s_audio(file_path, dir_path)

    def run(self):
        logger.info("🚀 Starting preprocessing pipeline")

        self.create_directories()
        self.split_dataset()

        logger.info("Processing test files")
        self.process_directory(self.test_dir)

        logger.info("Processing train files")
        self.process_directory(self.train_dir)

        logger.info("🎉 Preprocessing completed for genre: %s", self.genre)


if __name__ == "__main__":
    preprocess = Preprocess(GenreType.POP)
    preprocess.run()

    preprocess = Preprocess(GenreType.FORRO)
    preprocess.run()

    preprocess = Preprocess(GenreType.ELETRONICA)
    preprocess.run()

    preprocess = Preprocess(GenreType.CLASSICA)
    preprocess.run()

    preprocess = Preprocess(GenreType.ROCK)
    preprocess.run()
