from pathlib import Path

from music_genre_classifier.data import MusicsLoader, DatasetBuilder, FeatureExtractor
from music_genre_classifier.utils import Preprocess
from music_genre_classifier.mlp import (
    MLPClassifier,
    ClassificationMetrics,
    ClassificationVisualizer,
)
from music_genre_classifier.configs import TRAIN_CSV, TEST_CSV, MODEL_PATH
from music_genre_classifier.models import GenreType, MLPConfig


class MlpService:
    def __init__(self):
        self.config = self.__get_config()
        self.model = MLPClassifier.load(MODEL_PATH) if MODEL_PATH.exists() else None
        self.genre_list = [genre.name for genre in GenreType]

    def preprocess(self) -> None:
        for genre in self.genre_list:
            Preprocess(genre).run()

    def train(self) -> None:
        self.__extract_and_save_datasets()
        X_train, y_train = self.__load_csv(TRAIN_CSV)
        X_test, y_test = self.__load_csv(TEST_CSV)

        self.model = MLPClassifier(config=self.config)
        self.model.train(X_train, y_train)

        self.__evaluate_and_save(X_test, y_test)
        self.model.save(MODEL_PATH)

    def predict_genre(self, audio_bytes: bytes) -> str:
        full_audio = MusicsLoader.load_audio(audio_bytes)
        features = FeatureExtractor.extract(full_audio)
        label = self.model.predict(features)
        return GenreType.get_name(label)

    def __get_config(self) -> MLPConfig:
        return MLPConfig()

    def __load_csv(self, csv_path: Path):
        return DatasetBuilder.load_from_csv(csv_path)

    def __extract_and_save_datasets(self) -> None:
        if not TRAIN_CSV.exists() or not TEST_CSV.exists():
            train_samples = MusicsLoader.load_dataset("train")
            test_samples = MusicsLoader.load_dataset("test")
            print(test_samples)
            DatasetBuilder.save_to_csv(train_samples, TRAIN_CSV)
            DatasetBuilder.save_to_csv(test_samples, TEST_CSV)

    def __evaluate_and_save(self, X_test, y_test):
        y_pred = self.model.predict_batch(X_test)
        metrics = ClassificationMetrics(
            y_true=y_test, y_pred=y_pred, num_classes=len(GenreType)
        )
        accuracy = metrics.accuracy() * 100

        class_names = [genre.name for genre in GenreType]

        precision = metrics.precision() * 100
        recall = metrics.recall() * 100
        f1 = metrics.f1_score() * 100

        print("\n" + "=" * 50)
        print("📊 Evaluation Metrics")
        print("=" * 50)
        print(f"Accuracy       : {accuracy:.2f} %")
        print(f"Macro F1       : {metrics.macro_f1() * 100:.2f} %\n")

        print("Class-wise Metrics:")
        print("-" * 50)
        print(f"{'Class':<15}{'Precision':<12}{'Recall':<12}{'F1 Score':<12}")
        print("-" * 50)
        for i, name in enumerate(class_names):
            print(f"{name:<15}{precision[i]:<12.2f}{recall[i]:<12.2f}{f1[i]:<12.2f}")

        print("\nConfusion Matrix:")
        print(metrics.conf_matrix)

        ClassificationVisualizer.plot_all_metrics(metrics, class_names, self.model)
        print("=" * 50 + "\n")
