"""Tests for classification metrics module."""
import numpy as np
import pytest as pytest

from music_genre_classifier.mlp import classification_metrics as cm

"""Unitary Test cases for ClassificationMetrics class."""
def test_confusion_matrix():
    y_true = [0, 1, 2, 1]
    y_pred = [0, 2, 1, 1]

    metrics = cm.ClassificationMetrics(y_true, y_pred, 3)

    expected = np.array([
        [1, 0, 0],
        [0, 1, 1],
        [0, 1, 0]
    ])

    assert np.array_equal(metrics.conf_matrix, expected)


def test_accuracy():
    y_true = [0, 1, 2, 1]
    y_pred = [0, 2, 1, 1]

    metrics = cm.ClassificationMetrics(y_true, y_pred, 3)

    assert metrics.accuracy() == pytest.approx(0.5)


def test_precision():
    y_true = [0, 1, 2, 1]
    y_pred = [0, 2, 1, 1]

    metrics = cm.ClassificationMetrics(y_true, y_pred, 3)

    expected = [1.0, 0.5, 0.0]
    assert np.allclose(metrics.precision(), expected)


def test_recall():
    y_true = [0, 1, 2, 1]
    y_pred = [0, 2, 1, 1]

    metrics = cm.ClassificationMetrics(y_true, y_pred, 3)

    expected = [1.0, 0.5, 0.0]
    assert np.allclose(metrics.recall(), expected)


def test_f1_score():
    y_true = [0, 1, 2, 1]
    y_pred = [0, 2, 1, 1]

    metrics = cm.ClassificationMetrics(y_true, y_pred, 3)

    expected = [1.0, 0.5, 0.0]
    assert np.allclose(metrics.f1_score(), expected)


def test_macro_f1():
    y_true = [0, 1, 2, 1]
    y_pred = [0, 2, 1, 1]

    metrics = cm.ClassificationMetrics(y_true, y_pred, 3)

    expected = np.mean([1.0, 0.5, 0.0])
    assert metrics.macro_f1() == pytest.approx(expected)


def test_weighted_f1():
    y_true = [0, 1, 2, 1]
    y_pred = [0, 2, 1, 1]

    metrics = cm.ClassificationMetrics(y_true, y_pred, 3)


    supports = np.array([1, 2, 1])
    weights = supports / supports.sum()

    expected = np.sum(weights * np.array([1.0, 0.5, 0.0]))

    assert metrics.weighted_f1() == pytest.approx(expected)

"""Test cases for ClassificationMetrics class."""

def test_perfect_classification():
    y_true = [0, 1, 2]
    y_pred = [0, 1, 2]

    metrics = cm.ClassificationMetrics(y_true, y_pred, 3)

    assert metrics.accuracy() == 1.0
    assert np.allclose(metrics.precision(), [1, 1, 1])
    assert np.allclose(metrics.recall(), [1, 1, 1])
    assert np.allclose(metrics.f1_score(), [1, 1, 1])

def test_division_by_zero_case():
    y_true = [0, 0]
    y_pred = [0, 0]

    metrics = cm.ClassificationMetrics(y_true, y_pred, 2)

    # classe 1 nunca aparece
    assert np.allclose(metrics.precision(), [1.0, 0.0])
    assert np.allclose(metrics.recall(), [1.0, 0.0])
    assert np.allclose(metrics.f1_score(), [1.0, 0.0])

def test_invalid_num_classes_negative():
    y_true = [0, 1]
    y_pred = [0, 1]

    with pytest.raises(ValueError):
        cm.ClassificationMetrics(y_true, y_pred, -1)