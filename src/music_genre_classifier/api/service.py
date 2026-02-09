import tempfile
from music_genre_classifier.data.preprocess import load_audio
from music_genre_classifier.features.extractor import extract_features
from music_genre_classifier.features.classifier import predict_genre


async def classify_audio(upload_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await upload_file.read())
        tmp_path = tmp.name

    y, sr = load_audio(tmp_path)
    features = extract_features(y, sr)
    genre = predict_genre(features)

    return genre
