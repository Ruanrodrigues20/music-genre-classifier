import csv
import pandas as pd

from pathlib import Path
from typing import List

from music_genre_classifier.models import AudioSample
from music_genre_classifier.configs import get_logger

logger = get_logger(__name__)


class DatasetBuilder:
    @staticmethod
    def save_to_csv(samples: List[AudioSample], file_path: Path):
        if not samples:
            logger.error("No samples to save")

        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, mode="w", newline="") as f:
            writer = csv.writer(f)

            feature_size = len(samples[0].features)
            header = [f"f{i}" for i in range(feature_size)] + ["label"]
            writer.writerow(header)

            for sample in samples:
                row = list(sample.features) + [sample.label]
                writer.writerow(row)

        print(f"✅ CSV write: {file_path}")

    @staticmethod
    def load_from_csv(file_path: Path):

        df = pd.read_csv(file_path)

        X = df.drop("label", axis=1).values
        y = df["label"].values

        return X, y
