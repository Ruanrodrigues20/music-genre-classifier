import csv

import numpy as np
import pandas as pd
import pytest

from music_genre_classifier.data.dataset_loader import DataSetLoader
from music_genre_classifier.models import GenreType

# --- TESTES PARA save_to_csv ---

def test_save_to_csv_with_empty_samples(tmp_path):
    """
    Objetivo: Garantir que a função não crie um arquivo se a lista de amostras estiver vazia.
    Decisão: Se não há dados, não deve haver arquivo no disco.
    """
    file_path = tmp_path / "dataset.csv"
    DataSetLoader.save_to_csv([], file_path)
    assert not file_path.exists()

def test_save_to_csv_creates_parent_directory(tmp_path, mocker):
    """
    Objetivo: Verificar se o método cria pastas automaticamente caso o caminho informado não exista.
    Decisão: Usamos o mocker para simular os nomes das colunas e tmp_path para o caminho.
    """
    mocker.patch(
        "music_genre_classifier.data.feature_extractor.FeatureExtractor.get_feature_names",
        return_value=["f1", "f2"],
    )

    # Mock de um objeto AudioSample
    sample = mocker.Mock()
    sample.filename = "audio.wav"
    sample.features = [0.1, 0.2]
    sample.label = GenreType.ROCK

    # Definimos um caminho que inclui uma pasta inexistente ('non_existing_dir')
    file_path = tmp_path / "non_existing_dir" / "dataset.csv"

    DataSetLoader.save_to_csv([sample], file_path)

    assert file_path.exists()
    assert file_path.parent.exists() # Garante que o mkdir foi chamado internamente

def test_save_to_csv_writes_correct_header(tmp_path, mocker):
    """
    Objetivo: Validar a estrutura do cabeçalho do CSV.
    Decisão: O cabeçalho deve ser sempre ['filename'] + features + ['label'].
    """
    mocker.patch(
        "music_genre_classifier.data.feature_extractor.FeatureExtractor.get_feature_names",
        return_value=["mfcc1", "mfcc2", "mfcc3"],
    )

    sample = mocker.Mock(filename="a.wav", features=[1, 2, 3], label=GenreType.ROCK)
    file_path = tmp_path / "dataset.csv"

    DataSetLoader.save_to_csv([sample], file_path)

    with open(file_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)

    assert header == ["filename", "mfcc1", "mfcc2", "mfcc3", "label"]

def test_save_to_csv_writes_correct_values(tmp_path, mocker):
    """
    Objetivo: Garantir que os dados inseridos no CSV condizem com o objeto AudioSample.
    Decisão: O rótulo (label) deve ser salvo como string em minúsculo (ex: 'rock').
    """
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
        next(reader) # Pula o cabeçalho
        row = next(reader)

    # Verificamos se a string salva no CSV é o nome do gênero em minúsculo
    assert row == ["song.mp3", "0.5", "0.9", "rock"]

# --- TESTES PARA load_from_csv ---

def test_load_from_csv_returns_correct_shapes(tmp_path):
    """
    Objetivo: Validar se o carregamento retorna matrizes NumPy com as dimensões esperadas.
    Decisão: X deve conter apenas as colunas numéricas (features), e y deve ter o mesmo número de linhas.
    """
    file_path = tmp_path / "test_load.csv"
    # Criamos um CSV artificial usando pandas
    data = {
        "filename": ["s1.wav", "s2.wav"],
        "f1": [0.1, 0.5],
        "f2": [0.2, 0.6],
        "label": ["rock", "pop"]
    }
    pd.DataFrame(data).to_csv(file_path, index=False)

    X, y = DataSetLoader.load_from_csv(file_path)

    # Com 2 amostras e 2 features, o shape deve ser (2, 2)
    assert X.shape == (2, 2)
    assert len(y) == 2
    assert isinstance(X, np.ndarray)

def test_load_from_csv_mapping_values(tmp_path):
    """
    Objetivo: Verificar a conversão de rótulo (string) para o valor inteiro do Enum.
    Decisão: O ML geralmente exige números, então o loader deve converter 'rock' -> GenreType.ROCK.value.
    """
    file_path = tmp_path / "test_mapping.csv"
    data = {
        "filename": ["track.wav"],
        "feature": [1.0],
        "label": ["rock"]
    }
    pd.DataFrame(data).to_csv(file_path, index=False)

    X, y = DataSetLoader.load_from_csv(file_path)

    # Verifica se o mapeamento string -> int (via Enum) funcionou
    assert y[0] == GenreType.ROCK.value

def test_load_from_csv_missing_label_column(tmp_path):
    """
    Objetivo: Garantir que o código falhe explicitamente se o arquivo CSV estiver mal formatado.
    Decisão: Se a coluna 'label' não existe, o carregamento de dados para treino é impossível.
    """
    file_path = tmp_path / "invalid.csv"
    # Criamos um DataFrame sem a coluna 'label' ou 'filename'
    pd.DataFrame({"f1": [1.0]}).to_csv(file_path, index=False)

    # Esperamos que o pandas ou o código disparem um erro de chave (KeyError)
    with pytest.raises(KeyError):
        DataSetLoader.load_from_csv(file_path)
