from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "dataset"
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_PATH = DATA_DIR / "model.joblib"
DATASET_CSV = DATA_DIR / "dataset.csv"
MODEL_CONFIG = DATA_DIR / "model_config.json"


if not DATA_DIR.exists():
    DATA_DIR.mkdir()

if not RESULTS_DIR.exists():
    RESULTS_DIR.mkdir()
