from music_genre_classifier.utils.preprocess import Preprocess
from pathlib import Path
import re

__all__ = ["Preprocess"]


def extract_number(path: Path):
    # pega o número do nome: pop74.wav → 74
    return int(re.search(r"\d+", path.stem).group())
