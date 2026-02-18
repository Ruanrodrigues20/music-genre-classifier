import librosa
import io
from pathlib import Path
from typing import Tuple

from music_genre_classifier.configs import SR


class AudioLoader:
    @staticmethod
    def from_path(path: str | Path) -> Tuple:
        return librosa.load(str(path), sr=SR, mono=True)

    @staticmethod
    def from_bytes(audio_bytes: bytes) -> Tuple:
        return librosa.load(io.BytesIO(audio_bytes), sr=SR, mono=True)
