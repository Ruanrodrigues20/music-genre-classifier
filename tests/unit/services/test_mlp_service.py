import json

import numpy as np
import pytest

from music_genre_classifier.models import GenreType, MLPConfig

# =========================================================
# FIXTURES
# =========================================================

@pytest.fixture
def fake_dataset():
    """
    Cria um dataset artificial consistente para os testes de treino.

    - X: matriz de features (20 amostras, 10 features)
    - y: labels balanceados (0 e 1)

    O uso de float32 simula dados reais vindos de extração de áudio.
    """
    X = np.random.rand(20, 10).astype(np.float32)
    y = np.array([0, 1] * 10)
    return X, y


@pytest.fixture
def fake_paths(mocker):
    """
    Mocka os paths globais DATASET_CSV e MODEL_PATH.

    Objetivo:
    - Evitar acesso ao filesystem real
    - Controlar o fluxo lógico do método train()
      (ex: forçar criação de dataset e modelo)

    Ambos retornam exists() == False para simular
    primeira execução do sistema.
    """
    fake_dataset_csv = mocker.Mock()
    fake_dataset_csv.exists.return_value = False

    fake_model_path = mocker.Mock()
    fake_model_path.exists.return_value = False

    mocker.patch(
        "music_genre_classifier.services.mlp_service.DATASET_CSV",
        fake_dataset_csv,
    )
    mocker.patch(
        "music_genre_classifier.services.mlp_service.MODEL_PATH",
        fake_model_path,
    )

    return fake_dataset_csv, fake_model_path


# =========================================================
# TESTS
# =========================================================

def test_train_creates_dataset_and_trains_model(mocker, fake_dataset, fake_paths):
    """
    Testa o fluxo completo do método train():

    Cenário:
    - Dataset ainda não existe
    - Modelo ainda não existe

    Verificações:
    - Dataset é criado e salvo
    - Dados são carregados corretamente
    - Split de treino/teste é realizado
    - Modelo é instanciado com a config correta
    - Treino recebe os dados corretos
    - Avaliação é chamada
    - Modelo é salvo
    """
    X, y = fake_dataset

    # ---------- mocks de dataset ----------
    # Simula leitura de arquivos de áudio
    load_audio_mock = mocker.patch(
        "music_genre_classifier.services.mlp_service.AudioLoader.load_dataset",
        return_value="fake_samples",
    )

    # Simula persistência do CSV
    save_csv_mock = mocker.patch(
        "music_genre_classifier.services.mlp_service.DataSetLoader.save_to_csv"
    )

    # Simula leitura do CSV já processado
    load_csv_mock = mocker.patch(
        "music_genre_classifier.services.mlp_service.DataSetLoader.load_from_csv",
        return_value=(X, y),
    )

    # ---------- mock do split ----------
    # Controla explicitamente os dados de treino/teste
    split_mock = mocker.patch(
        "music_genre_classifier.services.mlp_service.train_test_split",
        return_value=(X[:15], X[15:], y[:15], y[15:]),
    )

    # ---------- mock do modelo ----------
    model_instance = mocker.Mock()
    mlp_cls_mock = mocker.patch(
        "music_genre_classifier.services.mlp_service.MLPClassifier",
        return_value=model_instance,
    )

    # ---------- mock da avaliação ----------
    # Isola o teste do comportamento interno de métricas/plots
    eval_mock = mocker.patch(
        "music_genre_classifier.services.mlp_service.MlpService._MlpService__evaluate_and_save"
    )

    # ---------- execução ----------
    from music_genre_classifier.services.mlp_service import MlpService

    service = MlpService()
    service.train()

    # =====================================================
    # ASSERTS
    # =====================================================

    # Dataset
    load_audio_mock.assert_called_once()
    save_csv_mock.assert_called_once_with("fake_samples")
    load_csv_mock.assert_called_once()

    # Split
    split_mock.assert_called_once()

    # Modelo criado com a configuração do serviço
    mlp_cls_mock.assert_called_once_with(config=service.config)

    # -------- treino (numpy-safe) --------
    model_instance.train.assert_called_once()
    train_args, _ = model_instance.train.call_args

    np.testing.assert_allclose(train_args[0], X[:15])
    np.testing.assert_array_equal(train_args[1], y[:15])

    # -------- avaliação (numpy-safe) --------
    eval_mock.assert_called_once()
    eval_args, _ = eval_mock.call_args

    np.testing.assert_allclose(eval_args[0], X[15:])
    np.testing.assert_array_equal(eval_args[1], y[15:])

    # Save
    model_instance.save.assert_called_once()


def test_predict_genre_returns_lowercase_genre(mocker):
    """
    Garante que:
    - O áudio é carregado corretamente
    - Features são extraídas
    - O modelo é chamado
    - O retorno final é uma string em lowercase
      representando o gênero musical
    """
    audio_bytes = b"fake_audio"

    load_audio_mock = mocker.patch(
        "music_genre_classifier.services.mlp_service.AudioLoader.load_audio",
        return_value="full_audio",
    )

    extract_mock = mocker.patch(
        "music_genre_classifier.services.mlp_service.FeatureExtractor.extract",
        return_value="features",
    )

    model_mock = mocker.Mock()
    model_mock.predict.return_value = GenreType.ROCK.value

    from music_genre_classifier.services.mlp_service import MlpService

    service = MlpService()
    service.model = model_mock

    result = service.predict_genre(audio_bytes)

    load_audio_mock.assert_called_once_with(audio_bytes)
    extract_mock.assert_called_once_with("full_audio")
    model_mock.predict.assert_called_once_with("features")

    assert result == "rock"


def test_get_config_returns_default_when_file_not_exists(mocker):
    """
    Se o arquivo de configuração não existir,
    o serviço deve usar MLPConfig com valores default.
    """
    fake_path = mocker.Mock()
    fake_path.exists.return_value = False

    mocker.patch(
        "music_genre_classifier.services.mlp_service.MODEL_CONFIG",
        fake_path,
    )

    from music_genre_classifier.services.mlp_service import MlpService

    service = MlpService()

    assert isinstance(service.config, MLPConfig)


def test_get_config_loads_from_json(mocker):
    """
    Testa o carregamento correto da configuração
    a partir de um arquivo JSON.
    """
    fake_config = {
        "hidden_layer_sizes": [64, 32],
        "learning_rate_init": 0.01,
        "max_iter": 50,
    }

    fake_model_config = mocker.Mock()
    fake_model_config.exists.return_value = True

    mocker.patch(
        "music_genre_classifier.services.mlp_service.MODEL_CONFIG",
        fake_model_config,
    )

    # Importante: evita tentar carregar um modelo existente
    fake_model_path = mocker.Mock()
    fake_model_path.exists.return_value = False

    mocker.patch(
        "music_genre_classifier.services.mlp_service.MODEL_PATH",
        fake_model_path,
    )

    mocker.patch(
        "builtins.open",
        mocker.mock_open(read_data=json.dumps(fake_config)),
    )

    from music_genre_classifier.services.mlp_service import MlpService

    service = MlpService()

    assert service.config.hidden_layer_sizes == [64, 32]
    assert service.config.learning_rate_init == 0.01
    assert service.config.max_iter == 50


def test_evaluate_and_save_calls_metrics_and_visualizer(mocker):
    """
    Testa se o método privado __evaluate_and_save:

    - Chama predict_batch no modelo
    - Instancia corretamente as métricas
    - Chama o visualizador de métricas
    """
    X_test = np.random.rand(5, 10)
    y_test = np.array([0, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0])

    num_classes = len(GenreType)

    model_mock = mocker.Mock()
    model_mock.predict_batch.return_value = y_pred

    metrics_instance = mocker.Mock()
    metrics_instance.accuracy.return_value = 0.8
    metrics_instance.macro_f1.return_value = 0.75
    metrics_instance.precision.return_value = np.ones(num_classes)
    metrics_instance.recall.return_value = np.ones(num_classes)
    metrics_instance.f1_score.return_value = np.ones(num_classes)
    metrics_instance.conf_matrix = np.zeros((num_classes, num_classes))

    metrics_cls_mock = mocker.patch(
        "music_genre_classifier.services.mlp_service.ClassificationMetrics",
        return_value=metrics_instance,
    )

    visualizer_mock = mocker.patch(
        "music_genre_classifier.services.mlp_service.ClassificationVisualizer.plot_all_metrics"
    )

    from music_genre_classifier.services.mlp_service import MlpService

    service = MlpService()
    service.model = model_mock

    service._MlpService__evaluate_and_save(X_test, y_test)

    model_mock.predict_batch.assert_called_once_with(X_test)

    metrics_cls_mock.assert_called_once_with(
        y_true=y_test,
        y_pred=y_pred,
        num_classes=num_classes,
    )

    visualizer_mock.assert_called_once()


def test_preprocess_calls_run_for_all_genres(mocker):
    """
    Garante que:
    - Um Preprocess é criado para cada gênero
    - Cada instância recebe o gênero correto
    - run() é chamado uma vez por gênero
    """
    preprocess_instance = mocker.Mock()
    preprocess_cls_mock = mocker.patch(
        "music_genre_classifier.services.mlp_service.Preprocess",
        return_value=preprocess_instance,
    )

    from music_genre_classifier.models import GenreType
    from music_genre_classifier.services.mlp_service import MlpService

    service = MlpService()
    service.preprocess()

    assert preprocess_cls_mock.call_count == len(GenreType)

    called_genres = [call.args[0] for call in preprocess_cls_mock.call_args_list]
    expected_genres = [GenreType.get_name(g) for g in GenreType]

    assert called_genres == expected_genres
    assert preprocess_instance.run.call_count == len(GenreType)
