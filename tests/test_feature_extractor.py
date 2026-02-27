import pytest
import numpy as np
from music_genre_classifier.data.feature_extractor import FeatureExtractor

def test_feature_names_matches_extraction_size():
    """Garante que a lista de nomes tem o mesmo tamanho que o vetor extraído."""
    sr = 22050
    y = np.random.uniform(-1, 1, sr * 5).astype(np.float32)
    
    features = FeatureExtractor.extract(y, sr)
    names = FeatureExtractor.get_feature_names()
    
    assert features is not None
    assert len(features) == len(names), f"Vetor tem {len(features)} mas nomes tem {len(names)}"

def test_extract_returns_float32():
    """Modelos de ML geralmente exigem float32 para performance e compatibilidade."""
    sr = 22050
    y = np.random.uniform(-1, 1, sr * 3)
    features = FeatureExtractor.extract(y, sr)
    
    assert features.dtype == np.float32

def test_extract_short_audio_returns_none(caplog):
    """Verifica se áudios menores que 1 segundo são rejeitados."""
    sr = 22050
    y = np.zeros(sr - 1)
    
    result = FeatureExtractor.extract(y, sr)
    
    assert result is None
    assert "Áudio curto ou vazio" in caplog.text

@pytest.mark.filterwarnings("ignore:Trying to estimate tuning")
def test_extract_silence_does_not_crash():
    """Garante que divisões por zero (Crest Factor, Pulse Clarity) são tratadas."""
    sr = 22050
    y = np.zeros(sr * 3)
    
    features = FeatureExtractor.extract(y, sr)
    
    assert features is not None
    assert not np.isnan(features).any(), "O vetor contém valores NaN!"
    assert not np.isinf(features).any(), "O vetor contém valores Infinitos!"

def test_stats_calculation():
    """Valida se a função interna de estatísticas retorna [média, variância]."""
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    stats = FeatureExtractor._stats(data)
    
    assert len(stats) == 2
    assert np.isclose(stats[0][0], 2.0)
    assert np.isclose(stats[1][0], 0.6666667, atol=1e-5)

def test_extract_exception_handling(mocker, caplog):
    """Cobre o bloco 'except' forçando um erro no processamento do librosa."""
    sr = 22050
    y = np.random.uniform(-1, 1, sr * 2)
    
    mocker.patch("librosa.util.normalize", side_effect=RuntimeError("Erro Matemático Forçado"))
    
    result = FeatureExtractor.extract(y, sr)
    
    assert result is None
    assert "Erro in extractor features" in caplog.text

def test_extract_pulse_clarity_short_onset_env(mocker):
    sr = 22050
    y = np.random.uniform(-1, 1, sr * 2).astype(np.float32)

    mocker.patch(
        "librosa.onset.onset_strength",
        return_value=np.array([0.1, 0.2], dtype=np.float32),
    )

    features = FeatureExtractor.extract(y, sr)

    assert features is not None
    assert not np.isnan(features).any()