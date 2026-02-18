from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "dataset"
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_PATH = DATA_DIR / "model.joblib"
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"


if not DATA_DIR.exists():
    DATA_DIR.mkdir()

if not RESULTS_DIR.exists():
    RESULTS_DIR.mkdir()
