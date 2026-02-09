from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()

# Templates (HTML)
templates = Jinja2Templates(directory="src/music_genre_classifier/api/static")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    return {"genre": "genre_placeholder"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=8000, reload=True)
