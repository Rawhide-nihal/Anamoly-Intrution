export interface DatasetInfo {
  id: string;
  filename: string;
  filepath: string;
  size_bytes: number;
  row_count: number;
  column_count: number;
  columns: string[];
  has_labels: boolean;
  label_distribution?: Record<string, number>;
  created_at: string;
  is_sample: boolean;
}

export interface ValidationReport {
  valid: boolean;
  total_rows: number;
  total_columns: number;
  missing_required_features: string[];
  extra_features: string[];
  matching_features: string[];
  zero_variance_found: string[];
  infinite_values_count: number;
  nan_values_count: number;
  duplicate_rows_count: number;
  has_labels: boolean;
  labels?: string[];
  compatible_models: string[];
  warnings: string[];
  errors: string[];
}

export interface ModelInfo {
  id: string;
  name: string;
  algorithm: string;
  variant: string;
  feature_count: number;
  includes_destination_port: boolean;
  status: string;
  path: string;
  description: string;
  contamination?: string;
  n_estimators?: number;
  supported_tasks: string[];
}

export interface AnalysisJobStatus {
  job_id: string;
  status: 'queued' | 'validating' | 'preparing_data' | 'running_analysis' | 'generating_output' | 'completed' | 'failed' | 'cancelled';
  progress: number;
  current_stage: string;
  dataset_id: string;
  model_id: string;
  algorithm: string;
  created_at: string;
  completed_at?: string;
  error_message?: string;
  execution_time_s?: number;
}

export interface AnalysisResultSummary {
  job_id: string;
  dataset_name: string;
  algorithm: string;
  model_name: string;
  total_rows: number;
  original_dataset_rows?: number;
  is_sampled?: boolean;
  sampling_info?: string;
  normal_count: number;
  anomaly_count: number;
  anomaly_percentage: number;
  execution_time_s: number;
  has_ground_truth: boolean;
  metrics?: {
    TP: number;
    TN: number;
    FP: number;
    FN: number;
    precision: number;
    recall: number;
    f1: number;
    fpr: number;
    fnr: number;
    roc_auc?: number;
  };
  category_flag_rates?: Record<string, { total: number; flagged: number; detection_rate: number }>;
  cluster_stats?: {
    n_clusters: number;
    noise_count: number;
    noise_percentage: number;
    is_sampled: boolean;
    sampling_method: string;
    sampling_seed?: number;
    sample_size: number;
    original_size: number;
    eps: number;
    min_samples: number;
    estimated_distance_matrix_mb?: number;
    cluster_sizes: Record<string, number>;
  };
  score_distribution?: {
    counts: number[];
    bins: number[];
  };
  is_historical_benchmark: boolean;
}

export interface PredictionRow {
  row_index: number;
  prediction: number;
  prediction_label: string;
  score?: number;
  cluster_id?: number;
  ground_truth?: string;
  features_preview: Record<string, any>;
}

export interface PaginatedPredictions {
  job_id: string;
  total_rows: number;
  page: number;
  page_size: number;
  total_pages: number;
  filter_type: string;
  rows: PredictionRow[];
  is_sampled?: boolean;
  original_dataset_rows?: number;
}

export interface ResearchStatus {
  all_present: boolean;
  found_count: number;
  total_expected: number;
  found_files: string[];
  missing_files: string[];
  message: string;
}

export interface KDistanceElbowResponse {
  suggested_eps: number;
  k: number;
  curve: Array<{ index: number; distance: number }>;
  sample_size: number;
}

export interface TrainingJobStatus {
  job_id: string;
  status: 'queued' | 'running' | 'completed' | 'failed';
  progress: number;
  current_stage: string;
  dataset_id: string;
  variant: string;
  created_at: string;
  completed_at?: string;
  error_message?: string;
  model_path?: string;
  training_rows?: number;
  feature_count?: number;
  execution_time_s?: number;
  evaluation_metrics?: {
    training_rows: number;
    feature_count: number;
    features_used: string[];
    n_estimators: number;
    contamination: number;
    random_state: number;
    internal_flagged_count: number;
    internal_anomaly_rate: number;
    mean_anomaly_score: number;
    max_anomaly_score: number;
    min_anomaly_score: number;
  };
}

