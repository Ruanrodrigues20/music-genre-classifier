import csv
import pandas as pd

from pathlib import Path

from music_genre_classifier.models import AudioSample, GenreType
from music_genre_classifier.configs import get_logger, DATASET_CSV
from music_genre_classifier.data.feature_extractor import FeatureExtractor

logger = get_logger(__name__)


class DataSetLoader:
    @staticmethod
    def save_to_csv(samples: list[AudioSample], file_path: Path = DATASET_CSV):
        if not samples:
            logger.error("No samples to save")
            return

        file_path.parent.mkdir(parents=True, exist_ok=True)

        feature_names = FeatureExtractor.get_feature_names()
        header = ["filename"] + feature_names + ["label"]

        with open(file_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

            for sample in samples:
                row = (
                    [sample.filename]
                    + list(sample.features)
                    + [sample.label.name.lower()]
                )
                writer.writerow(row)

        logger.info(f"✅ CSV write: {file_path}")

    @staticmethod
    def load_from_csv(file_path: Path = DATASET_CSV):
        df = pd.read_csv(file_path)

        # remove colunas que não são features
        X = df.drop(["label", "filename"], axis=1).values

        # string → Enum → int
        y = df["label"].apply(lambda v: GenreType[v.upper()].value).values

        return X, y
