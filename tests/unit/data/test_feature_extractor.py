import numpy as np
import pytest

from music_genre_classifier.data.feature_extractor import FeatureExtractor


def test_feature_names_matches_extraction_size():
    """
    Objetivo: Garantir a integridade entre os dados extraídos e os nomes das colunas.
    Explicação: Como o modelo de ML precisa de colunas fixas, se o vetor de features 
    tiver 500 elementos, a lista de nomes deve ter exatamente 500. Se houver divergência,
    o treinamento do modelo ou a criação do CSV falharão.
    """
    sr = 22050
    y = np.random.uniform(-1, 1, sr * 5).astype(np.float32)

    features = FeatureExtractor.extract(y, sr)
    names = FeatureExtractor.get_feature_names()

    assert features is not None
    assert len(features) == len(names), f"Vetor tem {len(features)} mas nomes tem {len(names)}"

def test_extract_returns_float32():
    """
    Objetivo: Validar a precisão e o tipo do dado.
    Explicação: Bibliotecas como Scikit-Learn e TensorFlow/PyTorch são otimizadas para float32. 
    Garantir esse tipo evita bugs de memória e inconsistências durante a inferência.
    """
    sr = 22050
    y = np.random.uniform(-1, 1, sr * 3)
    features = FeatureExtractor.extract(y, sr)

    assert features.dtype == np.float32

def test_extract_short_audio_returns_none(caplog):
    """
    Objetivo: Testar a regra de negócio de duração mínima.
    Explicação: Tentamos extrair de um áudio de quase 1 segundo (sr - 1 amostras). 
    O código deve identificar que é insuficiente para as janelas de FFT do Librosa, 
    retornar None e registrar o aviso no log.
    """
    sr = 22050
    y = np.zeros(sr - 1)

    result = FeatureExtractor.extract(y, sr)

    assert result is None
    assert "Áudio curto ou vazio" in caplog.text

@pytest.mark.filterwarnings("ignore:Trying to estimate tuning")
def test_extract_silence_does_not_crash():
    """
    Objetivo: Validar a estabilidade matemática (Divisão por Zero).
    Explicação: Em áudios de silêncio (só zeros), cálculos como o Crest Factor 
    ou o Pulse Clarity podem tentar dividir por zero. O teste garante que o 
    código lida com isso usando o `EPS` (épsilon) definido na classe, 
    não gerando valores NaN (Not a Number) ou Infinitos.
    """
    sr = 22050
    y = np.zeros(sr * 3)

    features = FeatureExtractor.extract(y, sr)

    assert features is not None
    assert not np.isnan(features).any(), "O vetor contém valores NaN!"
    assert not np.isinf(features).any(), "O vetor contém valores Infinitos!"

def test_stats_calculation():
    """
    Objetivo: Validar a função auxiliar de estatísticas.
    Explicação: Verifica se o método `_stats` calcula corretamente a média e 
    variância em um array 2D. É um teste de unidade puro para garantir que a 
    base dos cálculos está correta.
    """
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    stats = FeatureExtractor._stats(data)

    assert len(stats) == 2
    # Média de [1,2,3] é 2.0 | Variância é 0.66...
    assert np.isclose(stats[0][0], 2.0)
    assert np.isclose(stats[1][0], 0.6666667, atol=1e-5)

def test_extract_exception_handling(mocker, caplog):
    """
    Objetivo: Testar a captura de exceções genéricas.
    Explicação: Usamos o `mocker` para forçar uma falha dentro de uma função interna 
    do Librosa (normalize). Isso prova que o bloco `try/except` do extrator 
    consegue capturar erros inesperados sem travar o pipeline.
    """
    sr = 22050
    y = np.random.uniform(-1, 1, sr * 2)

    mocker.patch("librosa.util.normalize", side_effect=RuntimeError("Erro Matemático Forçado"))

    result = FeatureExtractor.extract(y, sr)

    assert result is None
    assert "Erro in extractor features" in caplog.text

def test_extract_pulse_clarity_short_onset_env(mocker):
    """
    Objetivo: Testar caso de borda no cálculo do Pulse Clarity.
    Explicação: Se o envelope de início (onset envelope) for muito curto, a 
    autocorrelação pode falhar. Este teste simula um retorno curto para garantir 
    que o código cai no bloco `else` e retorna 0.0 em vez de estourar erro de índice.
    """
    sr = 22050
    y = np.random.uniform(-1, 1, sr * 2).astype(np.float32)

    mocker.patch(
        "librosa.onset.onset_strength",
        return_value=np.array([0.1, 0.2], dtype=np.float32),
    )

    features = FeatureExtractor.extract(y, sr)

    assert features is not None
    assert not np.isnan(features).any()
