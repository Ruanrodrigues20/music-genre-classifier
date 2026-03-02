from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from music_genre_classifier.api.dto.predict_response import PredictResponse
from music_genre_classifier.configs import RESULTS_DIR
from music_genre_classifier.services import MlpService

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI()
mlp_service = MlpService()
templates = Jinja2Templates(directory=BASE_DIR / "templates")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR / "static"),
    name="static",
)

app.mount(
    "/results",
    StaticFiles(directory=RESULTS_DIR),
    name="results",
)


def get_mlp_service():
    return mlp_service


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "pages/home.html",
        {"request": request},
    )


@app.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    return templates.TemplateResponse(
        "pages/about.html",
        {"request": request},
    )


@app.get("/inference", response_class=HTMLResponse)
async def inference(request: Request):
    return templates.TemplateResponse(
        "pages/inference.html",
        {"request": request},
    )


@app.get("/results", response_class=HTMLResponse)
async def results(request: Request):
    return templates.TemplateResponse(
        "pages/results.html",
        {"request": request},
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    service: MlpService = Depends(get_mlp_service),
):
    try:
        audio_bytes = await file.read()
        genre = service.predict_genre(audio_bytes)

        return PredictResponse(genre=genre)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
