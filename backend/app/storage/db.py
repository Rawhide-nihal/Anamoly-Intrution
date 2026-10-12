import sqlite3
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from ..config import DB_PATH

def get_connection():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_connection() as conn:
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
        conn.commit()

init_db()
