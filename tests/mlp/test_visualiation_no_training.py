from unittest.mock import patch
from music_genre_classifier.mlp.visualiation import ClassificationVisualizer


class DummyModel:
    pass


@patch("music_genre_classifier.mlp.visualiation.plt")
def test_plot_training_without_training(mock_plt):
    mock_model = DummyModel()

    ClassificationVisualizer.plot_training(mock_model)

    mock_plt.figure.assert_not_called()