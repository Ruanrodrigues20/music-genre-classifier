import numpy as np
from unittest.mock import patch, MagicMock

from music_genre_classifier.mlp.visualiation import ClassificationVisualizer


@patch("music_genre_classifier.mlp.visualiation.plt")
@patch("music_genre_classifier.mlp.visualiation.os.path.join")
def test_plot_confusion_matrix_saves_file(mock_join, mock_plt):
    mock_join.return_value = "fake_path/confusion_matrix.png"

    conf_matrix = np.array([[5, 1], [2, 7]])
    class_names = ["rock", "jazz"]

    ClassificationVisualizer.plot_confusion_matrix(conf_matrix, class_names)

    mock_plt.figure.assert_called_once()
    mock_plt.imshow.assert_called_once()
    mock_plt.savefig.assert_called_once_with(
        "fake_path/confusion_matrix.png", dpi=150
    )
    mock_plt.close.assert_called_once()

@patch("music_genre_classifier.mlp.visualiation.plt")
def test_plot_training_with_data(mock_plt):
    mock_model = MagicMock()
    mock_model.loss_curve_ = [0.9, 0.7, 0.5]
    mock_model.validation_scores_ = [0.6, 0.75, 0.85]

    ClassificationVisualizer.plot_training(mock_model)

    mock_plt.figure.assert_called_once()
    mock_plt.plot.assert_any_call([0.9, 0.7, 0.5], label="Training Loss", color="blue")
    mock_plt.plot.assert_any_call(
        [0.6, 0.75, 0.85], label="Validation Score", color="orange"
    )
    mock_plt.savefig.assert_called_once()

@patch("music_genre_classifier.mlp.visualiation.plt")
@patch("music_genre_classifier.mlp.visualiation.os.path.join")
def test_plot_class_metrics(mock_join, mock_plt):
    mock_join.return_value = "fake_path/precision.png"

    values = [0.8, 0.6]
    class_names = ["rock", "jazz"]

    ClassificationVisualizer.plot_class_metrics(
        values,
        class_names,
        "Precision per Class",
        "Precision",
        "precision.png",
    )

    mock_plt.figure.assert_called_once()
    mock_plt.bar.assert_called_once()
    mock_plt.savefig.assert_called_once_with(
        "fake_path/precision.png", dpi=150
    )
    mock_plt.close.assert_called_once()


@patch.object(ClassificationVisualizer, "plot_training")
@patch.object(ClassificationVisualizer, "plot_class_metrics")
@patch.object(ClassificationVisualizer, "plot_confusion_matrix")
def test_plot_all_metrics_calls_all(
    mock_conf_matrix,
    mock_class_metrics,
    mock_training,
):
    mock_metrics = MagicMock()
    mock_metrics.conf_matrix = [[1, 0], [0, 1]]
    mock_metrics.precision.return_value = [0.9, 0.8]
    mock_metrics.recall.return_value = [0.85, 0.75]
    mock_metrics.f1_score.return_value = [0.87, 0.77]

    mock_model = MagicMock()
    mock_model.model = MagicMock()

    class_names = ["rock", "jazz"]

    ClassificationVisualizer.plot_all_metrics(
        mock_metrics,
        class_names,
        mock_model,
    )

    mock_conf_matrix.assert_called_once()
    assert mock_class_metrics.call_count == 3
    mock_training.assert_called_once()