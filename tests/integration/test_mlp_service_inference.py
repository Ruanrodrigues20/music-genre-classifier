import shutil
import pytest

from music_genre_classifier.services.mlp_service import MlpService
from music_genre_classifier.models import GenreType


def test_mlp_service_predict_with_pretrained_model(tmp_path, mocker):
    """
    Teste de integração de inferência do MlpService.

    Garante que:
    - Um modelo real (.joblib) pode ser carregado corretamente
    - O pipeline completo de inferência funciona:
        AudioLoader -> FeatureExtractor -> MLPClassifier
    - O resultado é um gênero válido

    Não testa:
    - Treinamento
    - Qualidade/acurácia do modelo
    - Métricas ou plots
    """

    # =====================================================
    # 1. Copia o modelo treinado para um diretório temporário
    # =====================================================
    model_src = "tests/integration/assets/model.joblib"
    model_dst = tmp_path / "model.joblib"

    shutil.copy(model_src, model_dst)

    # =====================================================
    # 2. Patch do MODEL_PATH para usar o modelo temporário
    # =====================================================
    mocker.patch(
        "music_genre_classifier.services.mlp_service.MODEL_PATH",
        model_dst,
    )

    # =====================================================
    # 3. Inicializa o service (modelo real é carregado)
    # =====================================================
    service = MlpService()

    assert service.model is not None, "Modelo não foi carregado"

    # =====================================================
    # 4. Executa inferência real com áudio real
    # =====================================================
    with open("tests/integration/assets/sample.wav", "rb") as f:
        audio_bytes = f.read()

    genre = service.predict_genre(audio_bytes)

    # =====================================================
    # 5. Valida o resultado
    # =====================================================
    assert isinstance(genre, str)
    assert genre in {g.name.lower() for g in GenreType}
    print(f"\n🎧 Gênero previsto pelo modelo: {genre}")
    
def test_predict_with_too_short_audio_raises_error(tmp_path, mocker):
    import numpy as np
    import soundfile as sf

    shutil.copy("tests/integration/assets/model.joblib", tmp_path / "model.joblib")

    mocker.patch(
        "music_genre_classifier.services.mlp_service.MODEL_PATH",
        tmp_path / "model.joblib",
    )

    service = MlpService()

    sr = 22050
    y = np.zeros(int(sr * 0.01))

    wav_path = tmp_path / "too_short.wav"
    sf.write(wav_path, y, sr)

    with open(wav_path, "rb") as f:
        audio_bytes = f.read()

    with pytest.raises(Exception):
        service.predict_genre(audio_bytes)

def test_predict_with_non_wav_file_raises_error(tmp_path, mocker):
    shutil.copy("tests/integration/assets/model.joblib", tmp_path / "model.joblib")

    mocker.patch(
        "music_genre_classifier.services.mlp_service.MODEL_PATH",
        tmp_path / "model.joblib",
    )

    service = MlpService()

    fake_file = b"isso nao eh um wav"

    with pytest.raises(Exception):
        service.predict_genre(fake_file)