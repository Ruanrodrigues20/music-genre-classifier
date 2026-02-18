import joblib

from sklearn.neural_network import MLPClassifier as SklearnMLP
from sklearn.preprocessing import StandardScaler
from music_genre_classifier.models import MLPConfig
from dataclasses import asdict


class MLPClassifier:
    def __init__(
        self,
        config: MLPConfig | None = None,
        model: SklearnMLP | None = None,
        scaler: StandardScaler | None = None,
    ):
        if model is None and config is None:
            raise ValueError(
                "MLPClassifier requires either 'model' (for loading) "
                "or 'config' (for creation)."
            )

        if model is not None and config is not None:
            raise ValueError("Pass only one of 'model' or 'config', not both.")

        if model is not None:
            self.model = model
        else:
            self.model = SklearnMLP(**asdict(config))

        self.scaler = scaler if scaler is not None else StandardScaler()

    def train(self, X_train, y_train):
        X_train = self.scaler.fit_transform(X_train)
        self.model.fit(X_train, y_train)
        print("LOSS", self.model.loss_curve_)
        print("VALIDATION SCORE", self.model.validation_scores_)

    def evaluate(self, X_test, y_test):
        X_test = self.scaler.transform(X_test)
        return self.model.score(X_test, y_test)

    def predict(self, features):
        features = self.scaler.transform([features])
        return self.model.predict(features)[0]

    def save(self, file_path):
        joblib.dump(
            {
                "model": self.model,
                "scaler": self.scaler,
            },
            file_path,
        )

    def predict_batch(self, X):
        X = self.scaler.transform(X)
        return self.model.predict(X)

    @staticmethod
    def load(file_path):
        data = joblib.load(file_path)
        classifier = MLPClassifier(model=data["model"], scaler=data["scaler"])
        return classifier
