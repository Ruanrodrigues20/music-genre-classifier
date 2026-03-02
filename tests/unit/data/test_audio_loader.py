from pathlib import Path

import numpy as np

from music_genre_classifier.configs import SAMPLE_RATE, SEGMENT_DURATION
from music_genre_classifier.data.audio_loader import AudioLoader
from music_genre_classifier.models import AudioSample, GenreType

# --- TESTES PARA load_audio (Processamento de Bytes/Buffer) ---


def test_load_audio_success(mocker):
    """
    Testa se o recorte e concatenação de áudio via bytes funcionam.
    
    Decisões:
    - Criamos um áudio falso de 100 segundos.
    - O AudioLoader extrai 5 segmentos de SEGMENT_DURATION cada.
    - Verificamos se o tamanho final é exatamente: 5 * SEGMENT_DURATION * SAMPLE_RATE.
    """
    duration = 100
    # Gera ruído branco para simular áudio
    y_mock = np.random.uniform(-1, 1, SAMPLE_RATE * duration)

    # Mockamos librosa para não depender de arquivos externos ou processamento pesado
    mocker.patch("librosa.load", return_value=(y_mock, SAMPLE_RATE))
    mocker.patch("librosa.get_duration", return_value=float(duration))

    result = AudioLoader.load_audio(b"fake_audio_content")

    # A lista starts_sec no AudioLoader possui 5 pontos de corte
    num_segments = 5
    expected_size = num_segments * int(SEGMENT_DURATION * SAMPLE_RATE)

    assert result is not None
    assert len(result) == expected_size
    assert isinstance(result, np.ndarray)

def test_load_audio_too_short(mocker):
    """
    Objetivo: Validar que áudios menores que a duração mínima exigida retornam None.
    Lógica: Se min_duration = SEGMENT_DURATION * NUM_SEGMENTS, qualquer áudio menor deve falhar.
    """
    # Mock de áudio de apenas 1 segundo
    mocker.patch("librosa.load", return_value=(np.zeros(SAMPLE_RATE), SAMPLE_RATE))
    mocker.patch("librosa.get_duration", return_value=1.0)

    result = AudioLoader.load_audio(b"short_audio")
    assert result is None

def test_load_audio_exception(mocker):
    """
    Objetivo: Garantir que a aplicação não quebre (crash) se o librosa encontrar 
    um arquivo corrompido, retornando None em vez de propagar o erro.
    """
    mocker.patch("librosa.load", side_effect=Exception("Corrupted file"))

    result = AudioLoader.load_audio(b"corrupted_bytes")
    assert result is None

# --- TESTES PARA _load_genre_dir (Processamento de Pastas) ---

def test_load_genre_dir_success(mocker, tmp_path):
    """
    Objetivo: Testar o processamento de uma pasta de gênero específica.
    Decisão: Usamos 'tmp_path' do pytest para criar um sistema de arquivos temporário 
    real, garantindo que o glob() e o Path funcionem corretamente.
    """
    genre_dir = tmp_path / "rock"
    genre_dir.mkdir()
    (genre_dir / "rock.00001.wav").touch() # Cria arquivo vazio para o glob encontrar

    mocker.patch("librosa.load", return_value=(np.zeros(SAMPLE_RATE), SAMPLE_RATE))
    # Simulamos que o extrator retorna um vetor de características fixo
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
    """
    Objetivo: Verificar se o sistema ignora arquivos onde a extração de features falha.
    """
    genre_dir = tmp_path / "pop"
    genre_dir.mkdir()
    (genre_dir / "pop.00001.wav").touch()

    mocker.patch("librosa.load", return_value=(np.zeros(SAMPLE_RATE), SAMPLE_RATE))
    mocker.patch(
        "music_genre_classifier.data.feature_extractor.FeatureExtractor.extract",
        return_value=None
    )

    samples = AudioLoader._load_genre_dir(genre_dir, GenreType.POP)

    assert len(samples) == 0
    assert "Feature is None" in caplog.text

def test_load_genre_dir_exception_on_load(mocker, tmp_path, caplog):
    """
    Objetivo: Testar a resiliência do loop. Se um arquivo der erro, o próximo deve ser processado.
    Decisão: Criamos dois arquivos e fazemos o primeiro falhar via side_effect.
    """
    genre_dir = tmp_path / "rock"
    genre_dir.mkdir()
    (genre_dir / "rock.00001.wav").touch()
    (genre_dir / "rock.00002.wav").touch()

    # O primeiro arquivo lança erro, o segundo carrega normalmente
    mocker.patch("librosa.load", side_effect=[Exception("Read Error"), (np.zeros(10), SAMPLE_RATE)])
    mocker.patch("music_genre_classifier.data.feature_extractor.FeatureExtractor.extract", return_value=[1])

    samples = AudioLoader._load_genre_dir(genre_dir, GenreType.ROCK)

    # Apenas 1 sample deve ter tido sucesso
    assert len(samples) == 1
    assert "Failed processing" in caplog.text

# --- TESTES PARA load_dataset (Orquestração Total) ---

def test_load_dataset_directory_not_found(mocker, caplog):
    """
    Objetivo: Garantir que se o caminho do dataset estiver errado, o código 
    apenas logue um aviso e retorne uma lista vazia, sem travar o app.
    """
    # Patcheamos o DATASET_DIR dentro do módulo onde ele é usado
    mocker.patch("music_genre_classifier.data.audio_loader.DATASET_DIR", Path("/non/existent/path"))

    samples = AudioLoader.load_dataset()

    assert len(samples) == 0
    assert "Directory not found" in caplog.text

def test_load_dataset_full_integration(mocker, tmp_path):
    """
    Objetivo: Testar o fluxo completo: iterar por gêneros -> ler pastas -> carregar arquivos.
    Decisão: Simulamos a estrutura de pastas esperada pelo Enum GenreType.
    """
    # Criamos pastas para cada gênero definido no modelo
    fake_dataset_dir = tmp_path / "data"
    fake_dataset_dir.mkdir()

    for genre in GenreType:
        name = GenreType.get_name(genre)
        g_dir = fake_dataset_dir / name
        g_dir.mkdir()
        (g_dir / f"{name}.00001.wav").touch()

    mocker.patch("music_genre_classifier.data.audio_loader.DATASET_DIR", fake_dataset_dir)
    mocker.patch("librosa.load", return_value=(np.zeros(100), SAMPLE_RATE))
    mocker.patch("music_genre_classifier.data.feature_extractor.FeatureExtractor.extract", return_value=np.array([1, 2]))

    samples = AudioLoader.load_dataset()

    # Deve carregar um sample para cada gênero existente no Enum
    assert len(samples) == len(list(GenreType))
    assert isinstance(samples[0], AudioSample)
