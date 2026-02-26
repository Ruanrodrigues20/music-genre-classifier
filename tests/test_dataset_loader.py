import pytest 
import csv
from pathlib import Path 
from music_genre_classifier.data.dataset_loader import DataSetLoader 
from music_genre_classifier.models import GenreType
import pandas as pd
import numpy as np

def test_save_to_csv_with_empty_samples(tmp_path): 
    file_path = tmp_path / "dataset.csv" 
    DataSetLoader.save_to_csv([], file_path) 
    assert not file_path.exists() 

def test_save_to_csv_creates_parent_directory(tmp_path, mocker):
    mocker.patch(
        "music_genre_classifier.data.feature_extractor.FeatureExtractor.get_feature_names",
        return_value=["f1", "f2"],
    )

    sample = mocker.Mock()
    sample.filename = "audio.wav"
    sample.features = [0.1, 0.2]
    sample.label = GenreType.ROCK

    file_path = tmp_path / "non_existing_dir" / "dataset.csv"

    DataSetLoader.save_to_csv([sample], file_path)

    assert file_path.exists()
    assert file_path.parent.exists()

def test_save_to_csv_writes_correct_header(tmp_path, mocker):
    mocker.patch(
        "music_genre_classifier.data.feature_extractor.FeatureExtractor.get_feature_names",
        return_value=["mfcc1", "mfcc2", "mfcc3"],
    )

    sample = mocker.Mock()
    sample.filename = "audio.wav"
    sample.features = [0.1, 0.2, 0.3]
    sample.label = GenreType.ROCK

    file_path = tmp_path / "dataset.csv"

    DataSetLoader.save_to_csv([sample], file_path)

    with open(file_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)

    assert header == ["filename", "mfcc1", "mfcc2", "mfcc3", "label"]

def test_save_to_csv_preserves_column_order(tmp_path, mocker):
    feature_names = ["zcr", "spectral_centroid", "rms"]

    mocker.patch(
        "music_genre_classifier.data.feature_extractor.FeatureExtractor.get_feature_names",
        return_value=feature_names,
    )

    sample = mocker.Mock()
    sample.filename = "track.wav"
    sample.features = [1.0, 2.0, 3.0]
    sample.label = GenreType.ROCK

    file_path = tmp_path / "dataset.csv"

    DataSetLoader.save_to_csv([sample], file_path)

    with open(file_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        row = next(reader)

    assert header == ["filename"] + feature_names + ["label"]
    assert row[1:-1] == ["1.0", "2.0", "3.0"]

def test_save_to_csv_writes_correct_values(tmp_path, mocker):
    mocker.patch(
        "music_genre_classifier.data.feature_extractor.FeatureExtractor.get_feature_names",
        return_value=["f1", "f2"],
    )

    sample = mocker.Mock()
    sample.filename = "song.mp3"
    sample.features = [0.5, 0.9]
    sample.label = GenreType.ROCK

    file_path = tmp_path / "dataset.csv"

    DataSetLoader.save_to_csv([sample], file_path)

    with open(file_path, newline="") as f:
        reader = csv.reader(f)
        next(reader)  
        row = next(reader)

    assert row == ["song.mp3", "0.5", "0.9", "rock"]

def test_load_from_csv_returns_correct_shapes(tmp_path):
    file_path = tmp_path / "test_load.csv"
    data = {
        "filename": ["s1.wav", "s2.wav"],
        "f1": [0.1, 0.5],
        "f2": [0.2, 0.6],
        "label": ["rock", "pop"]
    }
    pd.DataFrame(data).to_csv(file_path, index=False)

    X, y = DataSetLoader.load_from_csv(file_path)
    
    assert X.shape == (2, 2)
    assert len(y) == 2
    assert isinstance(X, np.ndarray)

def test_load_from_csv_mapping_values(tmp_path):
    file_path = tmp_path / "test_mapping.csv"
    data = {
        "filename": ["track.wav"],
        "feature": [1.0],
        "label": ["rock"] 
    }
    pd.DataFrame(data).to_csv(file_path, index=False)

    X, y = DataSetLoader.load_from_csv(file_path)

    assert y[0] == GenreType.ROCK.value

def test_save_to_csv_multiple_samples(tmp_path, mocker):
    mocker.patch("music_genre_classifier.data.feature_extractor.FeatureExtractor.get_feature_names", return_value=["f1"])
    
    s1 = mocker.Mock(filename="1.wav", features=[0.1], label=GenreType.ROCK)
    s2 = mocker.Mock(filename="2.wav", features=[0.2], label=GenreType.POP)
    
    file_path = tmp_path / "multi.csv"
    DataSetLoader.save_to_csv([s1, s2], file_path)
    
    df = pd.read_csv(file_path)
    assert len(df) == 2
    assert df.iloc[1]["filename"] == "2.wav"

def test_load_from_csv_missing_label_column(tmp_path):
    file_path = tmp_path / "invalid.csv"
    pd.DataFrame({"f1": [1.0]}).to_csv(file_path, index=False)

    with pytest.raises(KeyError):
        DataSetLoader.load_from_csv(file_path)