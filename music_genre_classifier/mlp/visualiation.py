import os

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from music_genre_classifier.configs import RESULTS_DIR
from music_genre_classifier.mlp import MLPClassifier
from music_genre_classifier.mlp.mlp_classifier import SklearnMLP


class ClassificationVisualizer:
    @staticmethod
    def plot_confusion_matrix(conf_matrix, class_names):
        plt.figure(figsize=(8, 6))

        im = plt.imshow(conf_matrix, interpolation="nearest", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.colorbar(im)

        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                plt.text(
                    j,
                    i,
                    format(conf_matrix[i, j], "d"),
                    horizontalalignment="center",
                    verticalalignment="center",
                    color="white"
                    if conf_matrix[i, j] > conf_matrix.max() / 2
                    else "black",
                    fontsize=10,
                )

        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()

        file_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
        plt.savefig(file_path, dpi=150)
        plt.close()

    @staticmethod
    def plot_training(model: SklearnMLP):

        if not hasattr(model, "loss_curve_"):
            print("O modelo ainda não foi treinado.")
            return

        loss = model.loss_curve_
        val_score = model.validation_scores_
        plt.figure(figsize=(10, 5))
        plt.plot(loss, label="Training Loss", color="blue")
        if val_score is not None:
            plt.plot(val_score, label="Validation Score", color="orange")
        plt.xlabel("Iteration")
        plt.ylabel("Value")
        plt.title("MLP Training Loss & Validation Score")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plt.savefig(RESULTS_DIR / "training_plot")

    @staticmethod
    def plot_class_metrics(values, class_names, title, ylabel, file_path):
        file_path = os.path.join(RESULTS_DIR, file_path)
        plt.figure(figsize=(8, 5))

        # Usa mapa de cores azul
        cmap = cm.get_cmap("Blues")  # Gradiente de azul
        colors = cmap(np.linspace(0.4, 0.8, len(values)))  # tom médio para claro

        bars = plt.bar(class_names, values, color=colors)
        plt.title(title)
        plt.ylabel(ylabel)
        plt.ylim(0, 1.05)
        plt.xticks(rotation=45)

        # Adiciona os valores acima das barras
        for bar, value in zip(bars, values):
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.01,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
                color="black",
            )

        plt.tight_layout()
        plt.savefig(file_path, dpi=150)
        plt.close()

    @staticmethod
    def plot_all_metrics(metrics, class_names, model: MLPClassifier):

        ClassificationVisualizer.plot_confusion_matrix(
            metrics.conf_matrix,
            class_names,
        )

        ClassificationVisualizer.plot_class_metrics(
            metrics.precision(),
            class_names,
            "Precision per Class",
            "Precision",
            "precision.png",
        )

        ClassificationVisualizer.plot_class_metrics(
            metrics.recall(),
            class_names,
            "Recall per Class",
            "Recall",
            "recall.png",
        )

        ClassificationVisualizer.plot_class_metrics(
            metrics.f1_score(),
            class_names,
            "F1 Score per Class",
            "F1 Score",
            "f1_score.png",
        )
        ClassificationVisualizer.plot_training(model=model.model)
