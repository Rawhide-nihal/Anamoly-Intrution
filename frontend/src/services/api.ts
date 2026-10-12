import {
  DatasetInfo,
  ValidationReport,
  ModelInfo,
  AnalysisJobStatus,
  AnalysisResultSummary,
  PaginatedPredictions,
  ResearchStatus
} from '../types';

const API_BASE = '/api';

export const api = {
  // System Health
  async getHealth() {
    const res = await fetch(`${API_BASE}/health`);
    return res.json();
  },

  // Datasets
  async getDatasets(): Promise<DatasetInfo[]> {
    const res = await fetch(`${API_BASE}/datasets`);
    return res.json();
  },

  async uploadDataset(file: File): Promise<{ dataset_id: string; message: string }> {
    const formData = new FormData();
    formData.append('file', file);
    const res = await fetch(`${API_BASE}/datasets/upload`, {
      method: 'POST',
      body: formData
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Upload failed');
    }
    return res.json();
  },

  async validateDataset(datasetId: string): Promise<ValidationReport> {
    const res = await fetch(`${API_BASE}/datasets/${datasetId}/validate`);
    if (!res.ok) throw new Error('Validation failed');
    return res.json();
  },

  async previewDataset(datasetId: string, limit = 50) {
    const res = await fetch(`${API_BASE}/datasets/${datasetId}/preview?limit=${limit}`);
    if (!res.ok) throw new Error('Failed to preview dataset');
    return res.json();
  },

  async getKDistanceCurve(datasetId: string, k = 10): Promise<import('../types').KDistanceElbowResponse> {
    const res = await fetch(`${API_BASE}/datasets/${datasetId}/k-distance?k=${k}`);
    if (!res.ok) throw new Error('Failed to compute k-distance curve');
    return res.json();
  },

  // Models
  async getModels(): Promise<ModelInfo[]> {
    const res = await fetch(`${API_BASE}/models`);
    return res.json();
  },

  // Jobs
  async createJob(params: {
    dataset_id: string;
    model_id: string;
    algorithm: string;
    dbscan_eps?: number;
    dbscan_min_samples?: number;
    sample_limit?: number;
  }): Promise<{ job_id: string; status: string }> {
    const res = await fetch(`${API_BASE}/jobs`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params)
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Failed to start analysis job');
    }
    return res.json();
  },

  async getJobs(): Promise<AnalysisJobStatus[]> {
    const res = await fetch(`${API_BASE}/jobs`);
    return res.json();
  },

  async getJobStatus(jobId: string): Promise<AnalysisJobStatus> {
    const res = await fetch(`${API_BASE}/jobs/${jobId}`);
    if (!res.ok) throw new Error('Job not found');
    return res.json();
  },

  async getJobResults(jobId: string): Promise<AnalysisResultSummary> {
    const res = await fetch(`${API_BASE}/jobs/${jobId}/results`);
    if (!res.ok) throw new Error('Results not ready');
    return res.json();
  },

  async getPredictions(
    jobId: string,
    page = 1,
    pageSize = 50,
    filterType = 'all'
  ): Promise<PaginatedPredictions> {
    const res = await fetch(
      `${API_BASE}/jobs/${jobId}/predictions?page=${page}&page_size=${pageSize}&filter_type=${filterType}`
    );
    if (!res.ok) throw new Error('Failed to fetch predictions');
    return res.json();
  },

  getExportCsvUrl(jobId: string): string {
    return `${API_BASE}/jobs/${jobId}/export/csv`;
  },

  // Research
  async getResearchStatus(): Promise<ResearchStatus> {
    const res = await fetch(`${API_BASE}/research/status`);
    return res.json();
  },

  async getPhase3Comparison(): Promise<any> {
    const res = await fetch(`${API_BASE}/research/phase3-comparison`);
    return res.json();
  },

  async getPhaseMetrics(phase: string): Promise<any> {
    const res = await fetch(`${API_BASE}/research/metrics/${phase}`);
    return res.json();
  },

  async getResearchReports(): Promise<Record<string, string>> {
    const res = await fetch(`${API_BASE}/research/reports`);
    return res.json();
  },

  async getResearchPlots(): Promise<Array<{ phase: string; filename: string; path: string; title: string }>> {
    const res = await fetch(`${API_BASE}/research/plots`);
    return res.json();
  }
};
