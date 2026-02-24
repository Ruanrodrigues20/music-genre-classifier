from pathlib import Path
import re


def extract_number(path: Path):
    return int(re.search(r"\d+", path.stem).group())
