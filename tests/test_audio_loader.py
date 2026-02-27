import pytest
import numpy as np
import io
from pathlib import Path
from music_genre_classifier.data.audio_loader import AudioLoader
from music_genre_classifier.models import GenreType



def test_load_audio_success(mocker):
    """Testa o recorte de áudio com sucesso."""
    sr = 22050
    duration = 100
    y_mock = np.random.uniform(-1, 1, sr * duration)
    
    mocker.patch("librosa.load", return_value=(y_mock, sr))
    mocker.patch("librosa.get_duration", return_value=float(duration))
    
    result = AudioLoader.load_audio(b"fake_audio_content")
    
    from music_genre_classifier.configs import SEGMENT_DURATION
    expected_size = 5 * int(SEGMENT_DURATION * sr)
    assert result is not None
    assert len(result) == expected_size
    assert isinstance(result, np.ndarray)

def test_load_audio_too_short(mocker):
    """Testa áudio que não atinge a duração mínima."""
    mocker.patch("librosa.load", return_value=(np.zeros(100), 22050))
    mocker.patch("librosa.get_duration", return_value=1.0)
    
    result = AudioLoader.load_audio(b"short_audio")
    assert result is None

def test_load_audio_exception(mocker):
    """Testa o bloco except caso o librosa falhe (ex: bytes corrompidos)."""
    mocker.patch("librosa.load", side_effect=Exception("Corrupted file"))
    
    result = AudioLoader.load_audio(b"corrupted_bytes")
    assert result is None


def test_load_dataset_directory_not_found(mocker, caplog):
    """Testa se o loader ignora diretórios de gêneros que não existem."""
    mocker.patch("music_genre_classifier.data.audio_loader.DATASET_DIR", Path("/non/existent/path"))
    
    samples = AudioLoader.load_dataset()
    
    assert len(samples) == 0
    assert "Directory not found" in caplog.text

def test_load_genre_dir_success(mocker, tmp_path):
    """Testa o carregamento de uma pasta de gênero com arquivos válidos."""
    genre_dir = tmp_path / "rock"
    genre_dir.mkdir()
    (genre_dir / "rock.00001.wav").touch()
    
    mocker.patch("librosa.load", return_value=(np.zeros(22050), 22050))
    mocker.patch(
        "music_genre_classifier.data.feature_extractor.FeatureExtractor.extract", 
        return_value=np.array([0.1, 0.2])
    )
    
    samples = AudioLoader._load_genre_dir(genre_dir, GenreType.ROCK)
    
    assert len(samples) == 1
    assert samples[0].filename == "rock.00001.wav"
    assert samples[0].label == GenreType.ROCK
    assert np.array_equal(samples[0].features, np.array([0.1, 0.2]))

def test_load_genre_dir_feature_is_none(mocker, tmp_path, caplog):
    """Testa quando o extrator de features falha em um arquivo específico."""
    genre_dir = tmp_path / "pop"
    genre_dir.mkdir()
    (genre_dir / "pop.00001.wav").touch()
    
    mocker.patch("librosa.load", return_value=(np.zeros(22050), 22050))
    mocker.patch(
        "music_genre_classifier.data.feature_extractor.FeatureExtractor.extract", 
        return_value=None
    )
    
    samples = AudioLoader._load_genre_dir(genre_dir, GenreType.POP)
    
    assert len(samples) == 0
    assert "Feature is None" in caplog.text

def test_load_genre_dir_exception_on_load(mocker, tmp_path, caplog):
    """Testa se um arquivo com erro (ex: erro de leitura) não para o loop total."""
    genre_dir = tmp_path / "rock"
    genre_dir.mkdir()
    (genre_dir / "rock.00001.wav").touch()
    (genre_dir / "rock.00002.wav").touch()
    
    mocker.patch("librosa.load", side_effect=[Exception("Read Error"), (np.zeros(10), 22050)])
    mocker.patch("music_genre_classifier.data.feature_extractor.FeatureExtractor.extract", return_value=[1])
    
    samples = AudioLoader._load_genre_dir(genre_dir, GenreType.ROCK)
    
    assert len(samples) == 1
    assert "Failed processing" in caplog.text

def test_load_dataset_full_integration(mocker, tmp_path):
    """Cobre a linha 40 garantindo que o loop de gêneros chama o carregador de arquivos."""
     
    genre_name = GenreType.get_name(GenreType.ROCK)
    fake_dataset_dir = tmp_path / "data"
    genre_dir = fake_dataset_dir / "rock"
    genre_dir.mkdir(parents=True)
    
    (genre_dir / "rock.00001.wav").touch() 

    mocker.patch("music_genre_classifier.data.audio_loader.DATASET_DIR", fake_dataset_dir)
    mocker.patch("librosa.load", return_value=(np.zeros(10), 22050))
    mocker.patch("librosa.get_duration", return_value=30.0)
    mocker.patch(
        "music_genre_classifier.data.feature_extractor.FeatureExtractor.extract", 
        return_value=np.array([0.1])
    )

    samples = AudioLoader.load_dataset()

    assert len(samples) >= 1