import os
import numpy as np
import pytest
import tempfile
from unittest.mock import MagicMock

from music_genre_classifier.services.mlp_service import MLPClassifier

def test_init_requires_model_or_config():
    with pytest.raises(ValueError):
        MLPClassifier()


def test_init_cannot_receive_both_model_and_config():
    mock_model = MagicMock()
    mock_config = MagicMock()

    with pytest.raises(ValueError):
        MLPClassifier(config=mock_config, model=mock_model)


def test_init_with_model_only():
    mock_model = MagicMock()
    classifier = MLPClassifier(model=mock_model)

    assert classifier.model == mock_model
    assert classifier.scaler is not None

def test_train_calls_fit():
    mock_model = MagicMock()
    mock_scaler = MagicMock()

    mock_scaler.fit_transform.return_value = [[1, 2]]

    classifier = MLPClassifier(model=mock_model, scaler=mock_scaler)

    X = [[0.1, 0.2]]
    y = [1]

    classifier.train(X, y)

    mock_scaler.fit_transform.assert_called_once_with(X)
    mock_model.fit.assert_called_once()

def test_evaluate_calls_score():
    mock_model = MagicMock()
    mock_scaler = MagicMock()

    mock_scaler.transform.return_value = [[1, 2]]
    mock_model.score.return_value = 0.85

    classifier = MLPClassifier(model=mock_model, scaler=mock_scaler)

    result = classifier.evaluate([[0.1, 0.2]], [1])

    mock_scaler.transform.assert_called_once()
    mock_model.score.assert_called_once()

    assert result == 0.85

def test_predict_single():
    mock_model = MagicMock()
    mock_scaler = MagicMock()

    mock_scaler.transform.return_value = [[1, 2]]
    mock_model.predict.return_value = np.array([2])

    classifier = MLPClassifier(model=mock_model, scaler=mock_scaler)

    result = classifier.predict([0.1, 0.2])

    mock_scaler.transform.assert_called_once()
    mock_model.predict.assert_called_once()

    assert result == 2

def test_predict_batch():
    mock_model = MagicMock()
    mock_scaler = MagicMock()

    mock_scaler.transform.return_value = [[1, 2], [3, 4]]
    mock_model.predict.return_value = np.array([0, 1])

    classifier = MLPClassifier(model=mock_model, scaler=mock_scaler)

    X = [[0.1, 0.2], [0.3, 0.4]]
    result = classifier.predict_batch(X)

    mock_scaler.transform.assert_called_once_with(X)
    mock_model.predict.assert_called_once()

    assert np.array_equal(result, np.array([0, 1]))

def test_save_creates_file():
    mock_model = MagicMock()
    mock_scaler = MagicMock()

    classifier = MLPClassifier(model=mock_model, scaler=mock_scaler)

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        file_path = tmp.name

    with patch("music_genre_classifier.mlp.mlp_classifier.joblib.dump") as mock_dump:
        classifier.save(file_path)

        mock_dump.assert_called_once_with(
            {
                "model": mock_model,
                "scaler": mock_scaler,
            },
            file_path,
        )

from unittest.mock import patch

def test_load_restores_model():
    mock_model = MagicMock()
    mock_scaler = MagicMock()

    with patch("music_genre_classifier.mlp.mlp_classifier.joblib.load") as mock_load:
        mock_load.return_value = {
            "model": mock_model,
            "scaler": mock_scaler,
        }

        classifier = MLPClassifier.load("fake_path")

        assert classifier.model == mock_model
        assert classifier.scaler == mock_scaler

def test_save_and_load_predict_consistency():
    mock_model = MagicMock()
    mock_scaler = MagicMock()

    mock_scaler.transform.return_value = [[5, 5]]
    mock_model.predict.return_value = np.array([1])

    classifier = MLPClassifier(model=mock_model, scaler=mock_scaler)

    with patch("music_genre_classifier.mlp.mlp_classifier.joblib.dump"), \
         patch("music_genre_classifier.mlp.mlp_classifier.joblib.load") as mock_load:

        mock_load.return_value = {
            "model": mock_model,
            "scaler": mock_scaler,
        }

        classifier.save("fake_path")
        loaded = MLPClassifier.load("fake_path")

        result = loaded.predict([[1, 2]])

        assert result == 1