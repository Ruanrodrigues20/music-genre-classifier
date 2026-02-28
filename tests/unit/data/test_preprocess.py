import numpy as np
import pytest
from pathlib import Path

# =========================================================
# FIXTURES
# =========================================================

@pytest.fixture
def fake_audio():
    """
    Gera um sinal de áudio fictício usando ruído branco.
    Utilizado para simular a carga de um arquivo sem precisar de um .mp3 real.
    """
    sr = 22050
    # 40 segundos de áudio para garantir que os cálculos de segmentos passem
    y = np.random.rand(sr * 40)
    return y, sr


@pytest.fixture
def preprocess(mocker, tmp_path):
    """
    Cria uma instância de Preprocess usando um diretório temporário real.
    Isso evita que os testes criem pastas ou arquivos no seu projeto real.
    """
    from music_genre_classifier.data.preprocess import Preprocess

    # Definimos um gênero para o teste
    p = Preprocess("rock")

    # Sobrescrevemos o music_dir para apontar para a pasta temporária do pytest
    p.music_dir = tmp_path / "rock"
    p.music_dir.mkdir(parents=True, exist_ok=True)

    return p


# =========================================================
# TESTES DO __init__
# =========================================================

def test_init_sets_genre_and_music_dir(mocker):
    """
    Verifica se o construtor atribui corretamente o gênero e monta
    o caminho do diretório baseada na configuração DATASET_DIR.
    """
    # Mockamos a constante de configuração para não depender do ambiente externo
    mocker.patch(
        "music_genre_classifier.data.preprocess.DATASET_DIR",
        Path("/fake/dataset"),
    )

    from music_genre_classifier.data.preprocess import Preprocess
    p = Preprocess("rock")

    assert p.genre == "rock"
    assert p.music_dir == Path("/fake/dataset") / "rock"


# =========================================================
# TESTES extract_30s_audio
# =========================================================

def test_extract_30s_audio_skips_if_too_short(mocker, preprocess):
    """
    Testa se a função ignora arquivos que não possuem a duração mínima necessária.
    Isso previne erros de indexação (out of bounds) ao tentar segmentar.
    """
    # Simula um áudio de apenas 5 segundos (curto demais)
    mocker.patch(
        "music_genre_classifier.data.preprocess.librosa.load",
        return_value=(np.zeros(1000), 22050),
    )
    mocker.patch(
        "music_genre_classifier.data.preprocess.librosa.get_duration",
        return_value=5,
    )

    # Espionamos a função de escrita de arquivo
    write_mock = mocker.patch("music_genre_classifier.data.preprocess.sf.write")

    preprocess.extract_30s_audio(Path("song.mp3"))

    # Verifica que o arquivo NÃO foi salvo
    write_mock.assert_not_called()


def test_extract_30s_audio_creates_30s_file(mocker, preprocess, fake_audio):
    """
    Verifica se a lógica de fatiamento (slicing) e concatenação funciona,
    garantindo que o arquivo final seja salvo com o nome correto.
    """
    y, sr = fake_audio

    mocker.patch(
        "music_genre_classifier.data.preprocess.librosa.load",
        return_value=(y, sr),
    )
    mocker.patch(
        "music_genre_classifier.data.preprocess.librosa.get_duration",
        return_value=40,
    )
    
    # Mockamos o sf.write para não gerar arquivos reais no disco durante este teste
    write_mock = mocker.patch("music_genre_classifier.data.preprocess.sf.write")

    audio_path = Path("song.mp3")
    preprocess.extract_30s_audio(audio_path)

    # Valida se o sf.write foi chamado
    write_mock.assert_called_once()

    # Extrai os argumentos passados para o sf.write
    args, _ = write_mock.call_args
    out_path, audio_written, used_sr = args

    # Verificações de saída
    assert str(out_path).endswith("song_30s.wav") # Caminho de saída correto
    assert used_sr == sr                          # Sample rate preservado
    assert isinstance(audio_written, np.ndarray)  # O dado enviado é um array numpy


# =========================================================
# TESTES process_directory
# =========================================================

def test_process_directory_calls_extract_only_for_mp3(mocker, preprocess):
    """
    Garante que o processador filtre apenas arquivos .mp3.
    Se houver um .wav ou .txt na pasta, ele deve ignorar.
    """
    # Criamos arquivos físicos reais na pasta temporária
    mp3 = preprocess.music_dir / "a.mp3"
    wav = preprocess.music_dir / "b.wav"
    mp3.touch()
    wav.touch()

    # Mockamos apenas o método interno para verificar chamadas
    extract_mock = mocker.patch.object(preprocess, "extract_30s_audio")

    preprocess.process_directory()

    # Deve ter sido chamado apenas para o mp3
    extract_mock.assert_called_once_with(mp3)


# =========================================================
# TESTES rename
# =========================================================

def test_rename_renames_files_in_order(preprocess):
    """
    Verifica se a renomeação final segue o padrão 'generoX.wav' 
    e se os arquivos originais '_30s.wav' foram de fato movidos.
    """
    # Criamos arquivos que seriam o resultado do extract_30s_audio
    f1 = preprocess.music_dir / "b_30s.wav"
    f2 = preprocess.music_dir / "a_30s.wav"
    f1.touch()
    f2.touch()

    preprocess.rename()

    # O sorted() no código garante que 'a' venha antes de 'b'
    # Logo: a_30s.wav -> rock0.wav | b_30s.wav -> rock1.wav
    assert (preprocess.music_dir / "rock0.wav").exists()
    assert (preprocess.music_dir / "rock1.wav").exists()
    
    # Verifica que os nomes antigos não existem mais
    assert not f1.exists()


# =========================================================
# TESTES run (Pipeline Completo)
# =========================================================

def test_run_returns_if_no_mp3_files(mocker, preprocess):
    """
    Verifica o 'early return' do método run. Se não houver MP3,
    ele não deve nem tentar processar ou renomear.
    """
    process_mock = mocker.patch.object(preprocess, "process_directory")
    rename_mock = mocker.patch.object(preprocess, "rename")

    preprocess.run()

    process_mock.assert_not_called()
    rename_mock.assert_not_called()


def test_run_calls_process_and_rename(mocker, preprocess):
    """
    Verifica se o pipeline principal executa as etapas na ordem correta
    quando existem arquivos para processar.
    """
    # Simula a existência de um arquivo MP3
    mp3 = preprocess.music_dir / "song.mp3"
    mp3.touch()

    process_mock = mocker.patch.object(preprocess, "process_directory")
    rename_mock = mocker.patch.object(preprocess, "rename")

    preprocess.run()

    # Valida que as duas etapas do pipeline foram acionadas
    process_mock.assert_called_once()
    rename_mock.assert_called_once()
