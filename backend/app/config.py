import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = BACKEND_DIR.parent

MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
PLOTS_DIR = ROOT_DIR / "plots"
SAMPLE_DATA_DIR = ROOT_DIR / "sample_data"

UPLOADS_DIR = BACKEND_DIR / "uploads"
JOBS_DATA_DIR = BACKEND_DIR / "jobs_data"
REPORTS_DIR = BACKEND_DIR / "reports"
DB_PATH = BACKEND_DIR / "app.db"

# Ensure runtime directories exist
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
JOBS_DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

ZERO_VARIANCE_DROPPED = [
    "Bwd PSH Flags",
    "Bwd URG Flags",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate"
]

DEFAULT_DBSCAN_CONFIG = {
    "eps": 2.552247135571616,
    "min_samples": 10,
    "max_sample_rows": 50000,
    "metric": "euclidean"
}
