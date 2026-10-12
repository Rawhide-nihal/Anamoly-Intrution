import os
import re
import uuid
import json
import shutil
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from ..config import UPLOADS_DIR, SAMPLE_DATA_DIR, ZERO_VARIANCE_DROPPED, RESULTS_DIR
from ..storage.db import get_connection
from ..schemas.base import ValidationReport, DatasetInfo

class DatasetService:
    def __init__(self):
        self._seed_sample_datasets()

    def _seed_sample_datasets(self):
        sample_file = SAMPLE_DATA_DIR / "sample_traffic_cicids2017.csv"
        if sample_file.exists():
            with get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id FROM datasets WHERE id = 'sample_cicids2017'")
                if not cursor.fetchone():
                    df = pd.read_csv(sample_file)
                    cols = [str(c).strip() for c in df.columns]
                    has_labels = "Label" in cols
                    dist = df["Label"].value_counts().to_dict() if has_labels else None
                    now = datetime.utcnow().isoformat()
                    cursor.execute("""
                        INSERT INTO datasets (id, filename, filepath, size_bytes, row_count, column_count, columns_json, has_labels, label_distribution_json, created_at, is_sample)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        "sample_cicids2017",
                        "sample_traffic_cicids2017.csv",
                        str(sample_file),
                        sample_file.stat().st_size,
                        len(df),
                        len(cols),
                        json.dumps(cols),
                        has_labels,
                        json.dumps(dist) if dist else None,
                        now,
                        1
                    ))
                    conn.commit()

    def list_datasets(self) -> List[Dict[str, Any]]:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM datasets ORDER BY created_at DESC")
            rows = cursor.fetchall()
            out = []
            for r in rows:
                d = dict(r)
                d["columns"] = json.loads(d["columns_json"])
                d["label_distribution"] = json.loads(d["label_distribution_json"]) if d["label_distribution_json"] else None
                out.append(d)
            return out

    def get_dataset(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
            r = cursor.fetchone()
            if r:
                d = dict(r)
                d["columns"] = json.loads(d["columns_json"])
                d["label_distribution"] = json.loads(d["label_distribution_json"]) if d["label_distribution_json"] else None
                return d
        return None

    def save_upload_stream(self, upload_file, max_bytes: int = 200 * 1024 * 1024) -> str:
        """
        Streams file to disk in 1MB chunks with strict size limits and sanitization.
        """
        raw_name = Path(upload_file.filename).name
        # Sanitize filename
        safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", raw_name)
        if not safe_name.lower().endswith(".csv"):
            raise ValueError("Only CSV files are supported.")

        dataset_id = str(uuid.uuid4())
        target_path = UPLOADS_DIR / f"{dataset_id}_{safe_name}"

        total_bytes = 0
        chunk_size = 1024 * 1024 # 1MB chunks

        try:
            with open(target_path, "wb") as f_out:
                while True:
                    chunk = upload_file.file.read(chunk_size)
                    if not chunk:
                        break
                    total_bytes += len(chunk)
                    if total_bytes > max_bytes:
                        raise ValueError(f"Uploaded file exceeds maximum limit of {max_bytes // (1024*1024)}MB.")
                    f_out.write(chunk)

            if total_bytes == 0:
                raise ValueError("Uploaded file is completely empty (0 bytes).")

            # Validate CSV structure & read headers with sample
            try:
                df_sample = pd.read_csv(target_path, nrows=5)
            except Exception as e:
                raise ValueError(f"Invalid or corrupted CSV format: {str(e)}")

            if df_sample.empty and total_bytes > 0:
                raise ValueError("CSV contains no tabular data rows.")

            cols = [str(c).strip() for c in df_sample.columns]
            if len(cols) == 0:
                raise ValueError("CSV has no recognized column headers.")

            # Fast row counting without reading entire table into memory
            row_count = 0
            with open(target_path, "r", encoding="utf-8", errors="replace") as f_count:
                for line in f_count:
                    if line.strip():
                        row_count += 1
            # Exclude header
            total_rows = max(0, row_count - 1)

            # Check if Label column exists
            has_labels = "Label" in cols
            dist = None
            if has_labels:
                try:
                    df_labels = pd.read_csv(target_path, usecols=["Label"])
                    dist = df_labels["Label"].astype(str).value_counts().to_dict()
                except Exception:
                    dist = None

            now = datetime.utcnow().isoformat()
            with get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO datasets (id, filename, filepath, size_bytes, row_count, column_count, columns_json, has_labels, label_distribution_json, created_at, is_sample)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    dataset_id,
                    safe_name,
                    str(target_path),
                    total_bytes,
                    total_rows,
                    len(cols),
                    json.dumps(cols),
                    has_labels,
                    json.dumps(dist) if dist else None,
                    now,
                    0
                ))
                conn.commit()

            return dataset_id

        except Exception as e:
            # Clean up partial or failed file on disk
            if target_path.exists():
                try:
                    target_path.unlink()
                except Exception:
                    pass
            raise e

    def validate_dataset(self, dataset_id: str) -> ValidationReport:
        ds = self.get_dataset(dataset_id)
        if not ds:
            raise ValueError(f"Dataset {dataset_id} not found")

        filepath = ds["filepath"]
        df = pd.read_csv(filepath)
        cols = [str(c).strip() for c in df.columns]
        df.columns = cols

        total_rows = len(df)
        total_columns = len(cols)

        # Read expected feature lists
        feat_path = RESULTS_DIR / "phase2c_feature_lists.json"
        with open(feat_path, "r", encoding="utf-8") as f:
            expected_data = json.load(f)
        expected_with_port = set(expected_data["with_port"])
        expected_without_port = set(expected_data["without_port"])

        current_cols_set = set(cols)
        matching_with_port = list(expected_with_port.intersection(current_cols_set))
        missing_with_port = list(expected_with_port - current_cols_set)

        matching_without_port = list(expected_without_port.intersection(current_cols_set))
        missing_without_port = list(expected_without_port - current_cols_set)

        extra_features = [c for c in cols if c not in expected_with_port and c != "Label" and c not in ZERO_VARIANCE_DROPPED]
        zero_var_found = [c for c in ZERO_VARIANCE_DROPPED if c in cols]

        # Non-finite values
        num_cols = df.select_dtypes(include=[np.number]).columns
        inf_count = int(np.isinf(df[num_cols]).values.sum()) if len(num_cols) > 0 else 0
        nan_count = int(df.isna().values.sum())
        dup_count = int(df.duplicated().sum())

        has_labels = "Label" in cols
        labels_list = list(df["Label"].astype(str).unique()) if has_labels else None

        compatible_models = []
        if len(missing_with_port) == 0:
            compatible_models.append("if_with_port")
            compatible_models.append("dbscan_clustering")
        if len(missing_without_port) == 0:
            compatible_models.append("if_without_port")

        warnings = []
        errors = []

        if total_rows == 0:
            errors.append("Dataset contains 0 rows.")
        if len(compatible_models) == 0:
            errors.append(f"Missing required flow features for Isolation Forest: {', '.join(missing_with_port[:5])} (and {len(missing_with_port)-5} more)")
        if inf_count > 0:
            warnings.append(f"Found {inf_count} infinite values; will be safely converted to NaN and imputed with training median.")
        if dup_count > 0:
            warnings.append(f"Found {dup_count} duplicate rows. Exact row alignment is preserved for inference.")
        if not has_labels:
            warnings.append("No 'Label' column found. Supervised benchmark metrics (Precision, Recall, ROC-AUC) will be unavailable; unsupervised anomaly detection will proceed.")

        return ValidationReport(
            valid=len(errors) == 0,
            total_rows=total_rows,
            total_columns=total_columns,
            missing_required_features=missing_with_port,
            extra_features=extra_features,
            matching_features=matching_with_port,
            zero_variance_found=zero_var_found,
            infinite_values_count=inf_count,
            nan_values_count=nan_count,
            duplicate_rows_count=dup_count,
            has_labels=has_labels,
            labels=labels_list,
            compatible_models=compatible_models,
            warnings=warnings,
            errors=errors
        )

    def get_preview(self, dataset_id: str, limit: int = 50) -> Dict[str, Any]:
        ds = self.get_dataset(dataset_id)
        if not ds:
            raise ValueError(f"Dataset {dataset_id} not found")
        df = pd.read_csv(ds["filepath"], nrows=limit)
        df = df.replace([np.inf, -np.inf], None)
        df = df.where(pd.notnull(df), None)
        return {
            "columns": [str(c).strip() for c in df.columns],
            "rows": df.to_dict(orient="records"),
            "total_rows": ds["row_count"],
            "preview_count": len(df)
        }

dataset_service = DatasetService()
