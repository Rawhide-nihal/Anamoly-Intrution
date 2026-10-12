import sqlite3
import json
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, Any, List, Optional
from ..config import DB_PATH

@contextmanager
def get_db():
    conn = sqlite3.connect(str(DB_PATH), timeout=20.0)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

# For backwards compatibility with existing call sites
get_connection = get_db

def init_db():
    with get_db() as conn:
        cursor = conn.cursor()
        # Datasets table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS datasets (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                filepath TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                row_count INTEGER NOT NULL,
                column_count INTEGER NOT NULL,
                columns_json TEXT NOT NULL,
                has_labels BOOLEAN NOT NULL,
                label_distribution_json TEXT,
                created_at TEXT NOT NULL,
                is_sample BOOLEAN DEFAULT 0
            )
        """)
        # Jobs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                progress INTEGER NOT NULL,
                current_stage TEXT NOT NULL,
                dataset_id TEXT NOT NULL,
                model_id TEXT NOT NULL,
                algorithm TEXT NOT NULL,
                params_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                completed_at TEXT,
                error_message TEXT,
                execution_time_s REAL,
                results_summary_json TEXT
            )
        """)
        # Training / Reproduction Jobs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_jobs (
                job_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                progress INTEGER NOT NULL,
                current_stage TEXT NOT NULL,
                dataset_id TEXT NOT NULL,
                variant TEXT NOT NULL,
                params_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                completed_at TEXT,
                error_message TEXT,
                model_path TEXT,
                training_rows INTEGER,
                feature_count INTEGER,
                execution_time_s REAL,
                evaluation_metrics_json TEXT
            )
        """)
        conn.commit()

init_db()
