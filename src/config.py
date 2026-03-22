from pathlib import Path

import yaml


ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"
OUTPUTS_DIR = ROOT / "outputs"
SUBMISSIONS_DIR = ROOT / "submissions"


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
