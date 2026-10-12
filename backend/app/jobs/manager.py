import json
import uuid
import time
import threading
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from ..config import JOBS_DATA_DIR, ZERO_VARIANCE_DROPPED
from ..storage.db import get_connection
from ..ml.service import ml_service
from ..schemas.base import AnalysisJobStatus, AnalysisResultSummary

class JobManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._cleanup_interrupted_jobs()

    def _cleanup_interrupted_jobs(self):
        """
        On server startup, mark any previously 'running' or 'validating' jobs as 'interrupted'.
        """
        try:
            with get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE jobs
                    SET status = 'failed', current_stage = 'Interrupted by server restart',
                        error_message = 'Process terminated unexpectedly during server lifecycle restart.'
                    WHERE status IN ('queued', 'validating', 'preparing_data', 'running_analysis', 'generating_output')
                """)
                conn.commit()
        except Exception as e:
            print("Failed to clean up interrupted jobs:", e)

    def create_job(self, dataset_id: str, model_id: str, algorithm: str, params: Dict[str, Any]) -> str:
        job_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO jobs (job_id, status, progress, current_stage, dataset_id, model_id, algorithm, params_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (job_id, "queued", 0, "Queued", dataset_id, model_id, algorithm, json.dumps(params), now))
            conn.commit()

        # Start execution in background thread
        thread = threading.Thread(target=self._run_job_worker, args=(job_id, dataset_id, model_id, algorithm, params), daemon=True)
        thread.start()
        return job_id

    def update_job_status(self, job_id: str, status: str, progress: int, stage: str, error_msg: Optional[str] = None):
        with self._lock:
            with get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE jobs
                    SET status = ?, progress = ?, current_stage = ?, error_message = ?
                    WHERE job_id = ?
                """, (status, progress, stage, error_msg, job_id))
                conn.commit()

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
            row = cursor.fetchone()
            if row:
                return dict(row)
        return None

    def list_jobs(self) -> list:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT 50")
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def _run_job_worker(self, job_id: str, dataset_id: str, model_id: str, algorithm: str, params: Dict[str, Any]):
        t0 = time.time()
        try:
            self.update_job_status(job_id, "validating", 10, "Validating dataset and schema")
            
            # Fetch dataset info from db
            with get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
                d_row = cursor.fetchone()
            if not d_row:
                raise ValueError("Dataset not found in storage")

            filepath = d_row["filepath"]
            filename = d_row["filename"]
            df = pd.read_csv(filepath)
            df.columns = [str(c).strip() for c in df.columns]

            has_labels = "Label" in df.columns
            ground_truth_labels_series = None
            if has_labels:
                ground_truth_labels_series = df["Label"].astype(str).str.replace("", "-", regex=False).str.replace(r"\s+", " ", regex=True).str.strip()

            self.update_job_status(job_id, "preparing_data", 30, "Preparing data tensors and transformations")

            scores = None
            cluster_labels = None
            cluster_stats = None

            if algorithm == "isolation_forest":
                self.update_job_status(job_id, "running_analysis", 60, "Executing Isolation Forest scoring")
                preds, scores, source_indices = ml_service.run_isolation_forest_inference(df, model_id=model_id)
                actual_analyzed_rows = len(preds)
            elif algorithm == "dbscan":
                self.update_job_status(job_id, "running_analysis", 60, "Performing transductive DBSCAN density clustering")
                eps = float(params.get("dbscan_eps", 2.5522))
                min_samples = int(params.get("dbscan_min_samples", 10))
                sample_limit = int(params.get("sample_limit", 50000))
                preds, cluster_labels, source_indices, cluster_stats = ml_service.run_dbscan_clustering(
                    df, eps=eps, min_samples=min_samples, sample_limit=sample_limit
                )
                actual_analyzed_rows = len(preds)
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")

            self.update_job_status(job_id, "generating_output", 85, "Compiling metrics and analytics")

            # Calculate predictions and metrics
            normal_count = int(np.sum(preds == 0))
            anomaly_count = int(np.sum(preds == 1))
            anomaly_percentage = float(anomaly_count / actual_analyzed_rows * 100) if actual_analyzed_rows > 0 else 0.0

            metrics = None
            category_flag_rates = None

            # EXACT ROW ALIGNMENT FOR GROUND TRUTH EVALUATION:
            # Match strictly against ground_truth_labels_series.iloc[source_indices]
            if has_labels and ground_truth_labels_series is not None:
                aligned_labels = ground_truth_labels_series.iloc[source_indices].to_numpy()
                y_true_binary = (aligned_labels != "BENIGN").astype(np.int8)

                tp = int(np.sum((y_true_binary == 1) & (preds == 1)))
                tn = int(np.sum((y_true_binary == 0) & (preds == 0)))
                fp = int(np.sum((y_true_binary == 0) & (preds == 1)))
                fn = int(np.sum((y_true_binary == 1) & (preds == 0)))
                prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
                rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
                f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
                fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
                fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

                roc_auc_val = None
                if scores is not None and len(np.unique(y_true_binary)) > 1:
                    r = pd.Series(scores).rank(method="average").to_numpy()
                    n1 = int((y_true_binary == 1).sum())
                    n0 = len(y_true_binary) - n1
                    if n1 > 0 and n0 > 0:
                        roc_auc_val = float((r[y_true_binary == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

                metrics = {
                    "TP": tp, "TN": tn, "FP": fp, "FN": fn,
                    "precision": prec, "recall": rec, "f1": f1,
                    "fpr": fpr, "fnr": fnr, "roc_auc": roc_auc_val
                }

                # Attack category breakdown aligned with source_indices
                cat_rates = {}
                for cat in np.unique(aligned_labels):
                    if cat != "BENIGN":
                        mask = (aligned_labels == cat)
                        flagged = int(np.sum(preds[mask] == 1))
                        total_c = int(np.sum(mask))
                        cat_rates[cat] = {
                            "total": total_c,
                            "flagged": flagged,
                            "detection_rate": float(flagged / total_c) if total_c > 0 else 0.0
                        }
                category_flag_rates = cat_rates

            # Score distribution histogram for IF
            score_dist = None
            if scores is not None and len(scores) > 0:
                hist, bin_edges = np.histogram(scores, bins=20)
                score_dist = {
                    "counts": [int(x) for x in hist],
                    "bins": [float(x) for x in bin_edges]
                }

            execution_time = float(time.time() - t0)

            summary = {
                "job_id": job_id,
                "dataset_name": filename,
                "algorithm": algorithm,
                "model_name": model_id,
                "total_rows": actual_analyzed_rows,
                "original_dataset_rows": len(df),
                "is_sampled": bool(cluster_stats.get("is_sampled")) if cluster_stats else False,
                "sampling_info": cluster_stats.get("sampling_method") if cluster_stats else None,
                "normal_count": normal_count,
                "anomaly_count": anomaly_count,
                "anomaly_percentage": anomaly_percentage,
                "execution_time_s": execution_time,
                "has_ground_truth": has_labels,
                "metrics": metrics,
                "category_flag_rates": category_flag_rates,
                "cluster_stats": cluster_stats,
                "score_distribution": score_dist,
                "is_historical_benchmark": False
            }

            # Persist per-row predictions to disk with exact source_indices
            out_file = JOBS_DATA_DIR / f"{job_id}_results.npz"
            save_dict = {
                "preds": preds,
                "source_indices": source_indices
            }
            if scores is not None:
                save_dict["scores"] = scores
            if cluster_labels is not None:
                save_dict["clusters"] = cluster_labels
            np.savez_compressed(out_file, **save_dict)

            # Mark completed in DB
            completed_now = datetime.utcnow().isoformat()
            with self._lock:
                with get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        UPDATE jobs
                        SET status = ?, progress = ?, current_stage = ?, completed_at = ?,
                            execution_time_s = ?, results_summary_json = ?
                        WHERE job_id = ?
                    """, ("completed", 100, "Completed", completed_now, execution_time, json.dumps(summary), job_id))
                    conn.commit()

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.update_job_status(job_id, "failed", 0, "Execution failed", error_msg=str(e))

job_manager = JobManager()
