from pathlib import Path

from sklearn.model_selection import train_test_split

from music_genre_classifier.data import MusicsLoader, DatasetBuilder, FeatureExtractor
from music_genre_classifier.utils import Preprocess
from music_genre_classifier.mlp import (
    MLPClassifier,
    ClassificationMetrics,
    ClassificationVisualizer,
)
from music_genre_classifier.configs import DATASET_CSV, MODEL_PATH
from music_genre_classifier.models import GenreType, MLPConfig


class MlpService:
    def __init__(self):
        self.config = self.__get_config()
        self.model = MLPClassifier.load(MODEL_PATH) if MODEL_PATH.exists() else None
        self.genre_list = [GenreType.get_name(g) for g in GenreType]

    def preprocess(self) -> None:
        for genre in self.genre_list:
            Preprocess(genre).run()

    def train(self) -> None:
        self.__extract_and_save_datasets()

        X, y = self.__load_csv(DATASET_CSV)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.3,
            random_state=42,
            shuffle=True,
            stratify=y,
        )

        self.model = MLPClassifier(config=self.config)
        self.model.train(X_train, y_train)

        self.__evaluate_and_save(X_test, y_test)
        self.model.save(MODEL_PATH)

    def predict_genre(self, audio_bytes: bytes) -> str:
        full_audio = MusicsLoader.load_audio(audio_bytes)
        features = FeatureExtractor.extract(full_audio)
        label = self.model.predict(features)
        return GenreType(label).name.lower()

    def __get_config(self) -> MLPConfig:
        return MLPConfig()

    def __load_csv(self, csv_path: Path):
        return DatasetBuilder.load_from_csv(csv_path)

    def __extract_and_save_datasets(self) -> None:
        if not DATASET_CSV.exists():
            data_samples = MusicsLoader.load_dataset()
            DatasetBuilder.save_to_csv(data_samples, DATASET_CSV)

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
