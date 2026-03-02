import re
from pathlib import Path


def extract_number(path: Path):
    return int(re.search(r"\d+", path.stem).group())
