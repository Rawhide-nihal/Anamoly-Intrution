from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

class DatasetInfo(BaseModel):
    id: str
    filename: str
    filepath: str
    size_bytes: int
    row_count: int
    column_count: int
    columns: List[str]
    has_labels: bool
    label_distribution: Optional[Dict[str, int]] = None
    created_at: str
    is_sample: bool = False

class ValidationReport(BaseModel):
    valid: bool
    total_rows: int
    total_columns: int
    missing_required_features: List[str]
    extra_features: List[str]
    matching_features: List[str]
    zero_variance_found: List[str]
    infinite_values_count: int
    nan_values_count: int
    duplicate_rows_count: int
    has_labels: bool
    labels: Optional[List[str]] = None
    compatible_models: List[str]
    warnings: List[str]
    errors: List[str]

class ModelInfo(BaseModel):
    id: str
    name: str
    algorithm: str
    variant: str
    feature_count: int
    includes_destination_port: bool
    status: str
    path: str
    description: str
    contamination: Optional[str] = None
    n_estimators: Optional[int] = None
    supported_tasks: List[str]

class AnalysisJobCreate(BaseModel):
    dataset_id: str
    model_id: str
    algorithm: str # "isolation_forest" or "dbscan"
    # DBSCAN parameters (if applicable)
    dbscan_eps: Optional[float] = 2.5522
    dbscan_min_samples: Optional[int] = 10
    sample_limit: Optional[int] = 50000

class AnalysisJobStatus(BaseModel):
    job_id: str
    status: str # "queued", "validating", "preparing_data", "running_analysis", "completed", "failed", "cancelled"
    progress: int # 0 to 100
    current_stage: str
    dataset_id: str
    model_id: str
    algorithm: str
    created_at: str
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    execution_time_s: Optional[float] = None

class AnalysisResultSummary(BaseModel):
    job_id: str
    dataset_name: str
    algorithm: str
    model_name: str
    total_rows: int
    normal_count: int
    anomaly_count: int
    anomaly_percentage: float
    execution_time_s: float
    has_ground_truth: bool
    metrics: Optional[Dict[str, Any]] = None # precision, recall, f1, fpr, fnr, roc_auc if labels present
    category_flag_rates: Optional[Dict[str, Any]] = None
    cluster_stats: Optional[Dict[str, Any]] = None # For DBSCAN: cluster counts, noise %
    score_distribution: Optional[Dict[str, Any]] = None
    is_historical_benchmark: bool = False

class PredictionRow(BaseModel):
    row_index: int
    prediction: int # 0: Normal, 1: Anomaly / Noise
    prediction_label: str # "Normal" or "Anomaly"
    score: Optional[float] = None # For IF: -decision_function
    cluster_id: Optional[int] = None # For DBSCAN: cluster ID or -1 for noise
    ground_truth: Optional[str] = None
    features_preview: Dict[str, Any]

class PaginatedPredictions(BaseModel):
    job_id: str
    total_rows: int
    page: int
    page_size: int
    total_pages: int
    filter_type: str # "all", "normal", "anomaly"
    rows: List[PredictionRow]
