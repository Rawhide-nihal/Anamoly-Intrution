import json
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
from ..config import MODELS_DIR, RESULTS_DIR, ZERO_VARIANCE_DROPPED

class MLService:
    def __init__(self):
        self.models_cache: Dict[str, Any] = {}
        self.feature_lists: Dict[str, Any] = {}
        self._load_feature_lists()

    def _load_feature_lists(self):
        feat_path = RESULTS_DIR / "phase2c_feature_lists.json"
        if feat_path.exists():
            with open(feat_path, "r", encoding="utf-8") as f:
                self.feature_lists = json.load(f)
        else:
            self.feature_lists = {"with_port": [], "without_port": []}

    def get_models_inventory(self) -> list:
        # Check actual files on disk
        if_with_port_ready = (
            (MODELS_DIR / "isolation_forest.pkl").exists() and
            (MODELS_DIR / "imputer.pkl").exists() and
            (MODELS_DIR / "scaler.pkl").exists()
        )
        if_no_port_ready = (
            (MODELS_DIR / "isolation_forest_no_port.pkl").exists() and
            (MODELS_DIR / "imputer_no_port.pkl").exists() and
            (MODELS_DIR / "scaler_no_port.pkl").exists()
        )

        return [
            {
                "id": "if_with_port",
                "name": "Isolation Forest (Baseline with Destination Port)",
                "algorithm": "Isolation Forest",
                "variant": "with_port",
                "feature_count": 70,
                "includes_destination_port": True,
                "status": "ready" if if_with_port_ready else "missing",
                "path": "models/isolation_forest.pkl",
                "description": "Trained exclusively on 1,467,538 BENIGN flows with 70 numeric features (Destination Port included). Out-of-sample inference.",
                "contamination": "auto",
                "n_estimators": 200,
                "supported_tasks": ["inference", "scoring", "evaluation"]
            },
            {
                "id": "if_without_port",
                "name": "Isolation Forest (Destination Port Ablated)",
                "algorithm": "Isolation Forest",
                "variant": "without_port",
                "feature_count": 69,
                "includes_destination_port": False,
                "status": "ready" if if_no_port_ready else "missing",
                "path": "models/isolation_forest_no_port.pkl",
                "description": "Trained on BENIGN flows with 69 numeric features (Destination Port excluded). Demonstrates port independence.",
                "contamination": "auto",
                "n_estimators": 200,
                "supported_tasks": ["inference", "scoring", "evaluation"]
            },
            {
                "id": "dbscan_clustering",
                "name": "DBSCAN Density Clustering (Unsupervised)",
                "algorithm": "DBSCAN",
                "variant": "transductive",
                "feature_count": 70,
                "includes_destination_port": True,
                "status": "ready",
                "path": "Unsupervised Transductive Clustering",
                "description": "Genuine density-based spatial clustering. Identifies noise points (-1) as anomalies. Offers research preset or dynamic k-distance calibration.",
                "contamination": "unsupervised",
                "n_estimators": None,
                "supported_tasks": ["clustering", "noise_detection", "density_analysis"]
            }
        ]

    def load_if_artifacts(self, model_id: str = "if_with_port") -> Tuple[Any, Any, Any, list]:
        if model_id not in self.models_cache:
            if model_id == "if_with_port":
                model_file = MODELS_DIR / "isolation_forest.pkl"
                imputer_file = MODELS_DIR / "imputer.pkl"
                scaler_file = MODELS_DIR / "scaler.pkl"
                expected_features = self.feature_lists.get("with_port", [])
            elif model_id == "if_without_port":
                model_file = MODELS_DIR / "isolation_forest_no_port.pkl"
                imputer_file = MODELS_DIR / "imputer_no_port.pkl"
                scaler_file = MODELS_DIR / "scaler_no_port.pkl"
                expected_features = self.feature_lists.get("without_port", [])
            else:
                raise ValueError(f"Unknown Isolation Forest model ID: {model_id}")

            if not model_file.exists() or not imputer_file.exists() or not scaler_file.exists():
                raise FileNotFoundError(f"Model artifacts for '{model_id}' are incomplete or missing.")

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = joblib.load(model_file)
                imputer = joblib.load(imputer_file)
                scaler = joblib.load(scaler_file)

            self.models_cache[model_id] = (model, imputer, scaler, expected_features)

        return self.models_cache[model_id]

    def preprocess_for_inference(self, df: pd.DataFrame, expected_features: list, imputer: Any, scaler: Any) -> np.ndarray:
        """
        Preprocesses dataframe using frozen preprocessors without refitting.
        Preserves exact row order.
        """
        df_cols = [str(c).strip() for c in df.columns]
        df_clean = df.copy()
        df_clean.columns = df_cols

        # Convert inf/-inf to nan
        num_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[num_cols] = df_clean[num_cols].replace([np.inf, -np.inf], np.nan)

        # Select expected features in exact required order
        X = df_clean[expected_features].copy()

        # Transform using frozen imputer and scaler
        X_imp = imputer.transform(X)
        X_scaled = scaler.transform(X_imp).astype(np.float32)
        return X_scaled

    def run_isolation_forest_inference(self, df: pd.DataFrame, model_id: str = "if_with_port") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (predictions, anomaly_scores, source_indices)
        predictions: 0 for normal, 1 for anomaly (score > 0)
        anomaly_score: -decision_function(X) (higher = more anomalous)
        source_indices: np.arange(len(df)) representing exact row alignment
        """
        model, imputer, scaler, expected_features = self.load_if_artifacts(model_id)
        X_scaled = self.preprocess_for_inference(df, expected_features, imputer, scaler)

        # Scientific convention from repository:
        # decision_function: positive for inliers, negative for outliers
        # anomaly_score = -decision_function(X)
        scores = -model.decision_function(X_scaled)
        preds = (scores > 0).astype(np.int8)
        source_indices = np.arange(len(df), dtype=np.int64)

        return preds, scores, source_indices

    def compute_k_distance_elbow(self, df: pd.DataFrame, k: int = 10, sample_size: int = 2000) -> Dict[str, Any]:
        """
        Computes k-distance curve on a representative sample to estimate eps.
        """
        from sklearn.neighbors import NearestNeighbors
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler

        n_rows = len(df)
        np.random.seed(42)
        eval_size = min(n_rows, sample_size)
        indices = np.random.choice(n_rows, size=eval_size, replace=False)
        df_sub = df.iloc[indices].copy()

        df_sub.columns = [str(c).strip() for c in df_sub.columns]
        cols_to_drop = [c for c in ZERO_VARIANCE_DROPPED if c in df_sub.columns]
        if "Label" in df_sub.columns:
            cols_to_drop.append("Label")
        X = df_sub.drop(columns=cols_to_drop, errors="ignore")
        num_cols = X.select_dtypes(include=[np.number]).columns
        X = X[num_cols].replace([np.inf, -np.inf], np.nan)

        imp = SimpleImputer(strategy="median")
        sc = StandardScaler()
        X_scaled = sc.fit_transform(imp.fit_transform(X)).astype(np.float32)

        k_val = min(k, len(X_scaled) - 1)
        nn = NearestNeighbors(n_neighbors=k_val)
        nn.fit(X_scaled)
        distances, _ = nn.kneighbors(X_scaled)
        k_distances = np.sort(distances[:, -1])

        # Suggested eps at ~90th percentile of k-distance
        suggested_eps = float(np.percentile(k_distances, 90))

        # Sample 50 points along the curve for visualization
        step = max(1, len(k_distances) // 50)
        sampled_curve = [
            {"index": int(i), "distance": float(k_distances[i])}
            for i in range(0, len(k_distances), step)
        ]

        return {
            "suggested_eps": round(suggested_eps, 4),
            "k": k_val,
            "curve": sampled_curve,
            "sample_size": eval_size
        }

    def run_dbscan_clustering(
        self,
        df: pd.DataFrame,
        eps: float = 2.5522,
        min_samples: int = 10,
        sample_limit: int = 50000,
        seed: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Performs genuine transductive DBSCAN density clustering.
        Returns:
            preds: (N,) binary array (1 for noise/anomaly, 0 for clustered normal)
            cluster_labels: (N,) cluster assignments (-1 for noise)
            source_indices: (N,) original row indices from df
            stats: metadata detailing sampling and cluster distributions
        """
        from sklearn.cluster import DBSCAN
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler

        n_rows = len(df)
        sampled = False

        if n_rows > sample_limit:
            sampled = True
            np.random.seed(seed)
            # Sample stable row indices
            source_indices = np.sort(np.random.choice(n_rows, size=sample_limit, replace=False)).astype(np.int64)
            df_work = df.iloc[source_indices].copy()
        else:
            source_indices = np.arange(n_rows, dtype=np.int64)
            df_work = df.copy()

        # Clean features
        df_work.columns = [str(c).strip() for c in df_work.columns]
        cols_to_drop = [c for c in ZERO_VARIANCE_DROPPED if c in df_work.columns]
        if "Label" in df_work.columns:
            cols_to_drop.append("Label")
        X = df_work.drop(columns=cols_to_drop, errors="ignore")

        # Keep numeric only
        num_cols = X.select_dtypes(include=[np.number]).columns
        X = X[num_cols].replace([np.inf, -np.inf], np.nan)

        # Standardize for clustering
        imp = SimpleImputer(strategy="median")
        sc = StandardScaler()
        X_scaled = sc.fit_transform(imp.fit_transform(X)).astype(np.float32)

        # Execute scikit-learn DBSCAN
        db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean", n_jobs=-1)
        cluster_labels = db.fit_predict(X_scaled)

        # In DBSCAN, -1 is noise (outlier anomaly)
        preds = (cluster_labels == -1).astype(np.int8)

        unique_clusters, counts = np.unique(cluster_labels[cluster_labels >= 0], return_counts=True)
        n_clusters = len(unique_clusters)
        noise_count = int(np.sum(cluster_labels == -1))

        # Memory estimation in MB: N x N x 4 bytes
        estimated_matrix_mb = (len(df_work) ** 2 * 4) / (1024 * 1024)

        stats = {
            "n_clusters": n_clusters,
            "noise_count": noise_count,
            "noise_percentage": float(noise_count / len(cluster_labels) * 100),
            "is_sampled": sampled,
            "sampling_method": "Uniform random seeded sampling" if sampled else "Full dataset (no sampling)",
            "sampling_seed": seed if sampled else None,
            "sample_size": len(df_work),
            "original_size": n_rows,
            "eps": eps,
            "min_samples": min_samples,
            "estimated_distance_matrix_mb": round(estimated_matrix_mb, 2),
            "cluster_sizes": {int(c): int(cnt) for c, cnt in zip(unique_clusters[:20], counts[:20])}
        }

        return preds, cluster_labels, source_indices, stats

ml_service = MLService()
