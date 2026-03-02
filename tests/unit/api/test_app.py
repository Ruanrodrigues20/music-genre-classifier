import io

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(mocker):
    """
    Cliente FastAPI com MlpService mockado.
    """
    from music_genre_classifier.api.app import app, get_mlp_service

    fake_service = mocker.Mock()
    fake_service.predict_genre.return_value = "rock"

    app.dependency_overrides[get_mlp_service] = lambda: fake_service

    client = TestClient(app)
    yield client

    app.dependency_overrides.clear()


# =========================================================
# PÁGINAS HTML
# =========================================================

@pytest.mark.parametrize(
    "route",
    ["/", "/about", "/inference", "/results"],
)
def test_html_pages_return_200(client, route):
    response = client.get(route)

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


# =========================================================
# /predict — SUCESSO
# =========================================================

def test_predict_success(client):
    fake_audio = io.BytesIO(b"fake audio bytes")

    response = client.post(
        "/predict",
        files={"file": ("test.wav", fake_audio, "audio/wav")},
    )

    assert response.status_code == 200
    assert response.json() == {"genre": "rock"}


# =========================================================
# /predict — ERRO
# =========================================================

def test_predict_returns_500_on_exception(mocker):
    from music_genre_classifier.api.app import app, get_mlp_service

    fake_service = mocker.Mock()
    fake_service.predict_genre.side_effect = Exception("boom")

    app.dependency_overrides[get_mlp_service] = lambda: fake_service

    client = TestClient(app)

    response = client.post(
        "/predict",
        files={"file": ("test.wav", io.BytesIO(b"x"), "audio/wav")},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "boom"

    app.dependency_overrides.clear()
