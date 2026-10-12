import React, { useEffect, useState } from 'react';
import { useSearchParams, Link } from 'react-router-dom';
import {
  TableProperties,
  Download,
  Filter,
  CheckCircle2,
  AlertTriangle,
  FileText,
  Clock,
  Layers,
  BarChart2,
  Activity
} from 'lucide-react';
import { api } from '../services/api';
import { AnalysisResultSummary, PaginatedPredictions, AnalysisJobStatus } from '../types';

export const ResultsPage: React.FC = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const [jobs, setJobs] = useState<AnalysisJobStatus[]>([]);
  const [selectedJobId, setSelectedJobId] = useState<string>('');
  const [resultsSummary, setResultsSummary] = useState<AnalysisResultSummary | null>(null);
  const [predictionsData, setPredictionsData] = useState<PaginatedPredictions | null>(null);

  const [page, setPage] = useState<number>(1);
  const [filterType, setFilterType] = useState<string>('all');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    loadCompletedJobs();
  }, []);

  const loadCompletedJobs = async () => {
    try {
      const allJobs = await api.getJobs();
      const completed = allJobs.filter((j) => j.status === 'completed');
      setJobs(completed);

      const queryJobId = searchParams.get('job_id');
      if (queryJobId && completed.some((j) => j.job_id === queryJobId)) {
        setSelectedJobId(queryJobId);
        fetchJobDetails(queryJobId, 1, 'all');
      } else if (completed.length > 0) {
        setSelectedJobId(completed[0].job_id);
        fetchJobDetails(completed[0].job_id, 1, 'all');
      }
    } catch (e) {
      console.error(e);
    }
  };

  const fetchJobDetails = async (jobId: string, pageNum: number, filter: string) => {
    setLoading(true);
    try {
      const [summary, preds] = await Promise.all([
        api.getJobResults(jobId),
        api.getPredictions(jobId, pageNum, 25, filter)
      ]);
      setResultsSummary(summary);
      setPredictionsData(preds);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  const handleSelectJob = (id: string) => {
    setSelectedJobId(id);
    setSearchParams({ job_id: id });
    setPage(1);
    setFilterType('all');
    fetchJobDetails(id, 1, 'all');
  };

  const handleFilterChange = (newFilter: string) => {
    setFilterType(newFilter);
    setPage(1);
    if (selectedJobId) {
      fetchJobDetails(selectedJobId, 1, newFilter);
    }
  };

  const handlePageChange = (newPage: number) => {
    setPage(newPage);
    if (selectedJobId) {
      fetchJobDetails(selectedJobId, newPage, filterType);
    }
  };

  return (
    <div className="space-y-8">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white tracking-tight">Analysis & Prediction Results</h1>
          <p className="text-xs text-slate-400">
            Per-row flow predictions, anomaly score distribution, and density cluster assignments
          </p>
        </div>

        {selectedJobId && (
          <a
            href={api.getExportCsvUrl(selectedJobId)}
            download
            className="inline-flex items-center space-x-2 px-4 py-2 rounded-lg bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-semibold text-xs shadow transition-all self-start"
          >
            <Download className="w-4 h-4" />
            <span>Export Full CSV Results</span>
          </a>
        )}
      </div>

      {/* Job Selector Dropdown / Pills */}
      {jobs.length > 0 && (
        <div className="flex items-center space-x-3 overflow-x-auto pb-2">
          <span className="text-xs text-slate-400 font-mono shrink-0">Analysis Runs:</span>
          {jobs.map((j) => (
            <button
              key={j.job_id}
              onClick={() => handleSelectJob(j.job_id)}
              className={`px-3 py-1.5 rounded-lg text-xs font-mono shrink-0 transition-all border ${
                selectedJobId === j.job_id
                  ? 'bg-cyan-500/15 border-cyan-500/40 text-cyan-300 font-semibold'
                  : 'bg-slate-900/60 border-slate-800 text-slate-400 hover:border-slate-700'
              }`}
            >
              {j.algorithm === 'isolation_forest' ? 'iForest' : 'DBSCAN'} • {j.job_id.slice(0, 6)} ({new Date(j.created_at).toLocaleTimeString()})
            </button>
          ))}
        </div>
      )}

      {/* Results Summary Cards */}
      {resultsSummary ? (
        <div className="space-y-6">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="p-5 rounded-xl bg-slate-900/40 border border-slate-800 space-y-1.5">
              <span className="text-slate-400 text-xs">Dataset Evaluated</span>
              <div className="text-sm font-bold text-cyan-400 font-mono truncate">
                {resultsSummary.dataset_name}
              </div>
              <div className="text-[11px] text-slate-500">
                {resultsSummary.total_rows.toLocaleString()} flows analyzed
                {resultsSummary.is_sampled && (
                  <span className="text-amber-400 ml-1">
                    (Sampled from {resultsSummary.original_dataset_rows?.toLocaleString()} rows)
                  </span>
                )}
              </div>
            </div>

            <div className="p-5 rounded-xl bg-slate-900/40 border border-slate-800 space-y-1.5">
              <span className="text-slate-400 text-xs">Normal Traffic</span>
              <div className="text-xl font-bold text-emerald-400 font-mono">
                {resultsSummary.normal_count.toLocaleString()}
              </div>
              <div className="text-[11px] text-slate-500">
                {(100 - resultsSummary.anomaly_percentage).toFixed(2)}% of total
              </div>
            </div>

            <div className="p-5 rounded-xl bg-slate-900/40 border border-slate-800 space-y-1.5">
              <span className="text-slate-400 text-xs">Anomalies Detected</span>
              <div className="text-xl font-bold text-rose-400 font-mono">
                {resultsSummary.anomaly_count.toLocaleString()}
              </div>
              <div className="text-[11px] text-slate-500">
                {resultsSummary.anomaly_percentage.toFixed(2)}% anomaly rate
              </div>
            </div>

            <div className="p-5 rounded-xl bg-slate-900/40 border border-slate-800 space-y-1.5">
              <span className="text-slate-400 text-xs">Execution Time</span>
              <div className="text-xl font-bold text-slate-200 font-mono">
                {resultsSummary.execution_time_s.toFixed(2)}s
              </div>
              <div className="text-[11px] text-slate-500">
                {resultsSummary.algorithm.toUpperCase()} algorithm
              </div>
            </div>
          </div>

          {/* Supervised Metrics If Labels Present */}
          {resultsSummary.has_ground_truth && resultsSummary.metrics && (
            <div className="p-6 rounded-xl bg-slate-900/40 border border-slate-800 space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="font-semibold text-slate-100 text-sm flex items-center space-x-2">
                  <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                  <span>Supervised Evaluation Metrics (Ground Truth 'Label' Present)</span>
                </h3>
                <span className="text-xs text-slate-500 font-mono">Verified Calculation</span>
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 text-center text-xs">
                <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800/80">
                  <div className="text-slate-400 text-[10px]">Precision</div>
                  <div className="font-bold text-cyan-400 font-mono text-base">
                    {(resultsSummary.metrics.precision * 100).toFixed(2)}%
                  </div>
                </div>
                <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800/80">
                  <div className="text-slate-400 text-[10px]">Recall</div>
                  <div className="font-bold text-cyan-400 font-mono text-base">
                    {(resultsSummary.metrics.recall * 100).toFixed(2)}%
                  </div>
                </div>
                <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800/80">
                  <div className="text-slate-400 text-[10px]">F1 Score</div>
                  <div className="font-bold text-cyan-400 font-mono text-base">
                    {(resultsSummary.metrics.f1 * 100).toFixed(2)}%
                  </div>
                </div>
                <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800/80">
                  <div className="text-slate-400 text-[10px]">False Positive Rate</div>
                  <div className="font-bold text-slate-300 font-mono text-base">
                    {(resultsSummary.metrics.fpr * 100).toFixed(2)}%
                  </div>
                </div>
                <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800/80">
                  <div className="text-slate-400 text-[10px]">False Negative Rate</div>
                  <div className="font-bold text-slate-300 font-mono text-base">
                    {(resultsSummary.metrics.fnr * 100).toFixed(2)}%
                  </div>
                </div>
                <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800/80">
                  <div className="text-slate-400 text-[10px]">ROC-AUC</div>
                  <div className="font-bold text-emerald-400 font-mono text-base">
                    {resultsSummary.metrics.roc_auc != null
                      ? resultsSummary.metrics.roc_auc.toFixed(4)
                      : 'N/A'}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* DBSCAN Specific Stats */}
          {resultsSummary.cluster_stats && (
            <div className="p-6 rounded-xl bg-blue-950/20 border border-blue-900/40 space-y-3 text-xs">
              <h3 className="font-semibold text-blue-300 text-sm flex items-center space-x-2">
                <Layers className="w-4 h-4 text-blue-400" />
                <span>DBSCAN Clustering Dynamics</span>
              </h3>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-slate-300">
                <div>Total Clusters Discovered: <strong className="text-blue-400 font-mono">{resultsSummary.cluster_stats.n_clusters}</strong></div>
                <div>Noise Flows (Anomalies): <strong className="text-rose-400 font-mono">{resultsSummary.cluster_stats.noise_count.toLocaleString()}</strong></div>
                <div>Noise Percentage: <strong className="text-rose-400 font-mono">{resultsSummary.cluster_stats.noise_percentage.toFixed(2)}%</strong></div>
                <div>Epsilon / MinSamples: <strong className="text-slate-200 font-mono">{resultsSummary.cluster_stats.eps} / {resultsSummary.cluster_stats.min_samples}</strong></div>
              </div>
            </div>
          )}

          {/* Row-Level Predictions Table */}
          <div className="p-6 rounded-xl bg-slate-900/40 border border-slate-800 space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
              <h3 className="font-semibold text-slate-100 text-sm flex items-center space-x-2">
                <TableProperties className="w-4 h-4 text-cyan-400" />
                <span>Row-Level Prediction Inspection</span>
              </h3>

              {/* Filters */}
              <div className="flex items-center space-x-2 text-xs">
                <span className="text-slate-500">Filter:</span>
                {(['all', 'normal', 'anomaly'] as const).map((ft) => (
                  <button
                    key={ft}
                    onClick={() => handleFilterChange(ft)}
                    className={`px-3 py-1 rounded-md capitalize font-mono text-xs transition-all ${
                      filterType === ft
                        ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                        : 'bg-slate-800 text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    {ft}
                  </button>
                ))}
              </div>
            </div>

            {loading ? (
              <div className="p-8 text-center text-xs text-slate-400 font-mono">
                Loading predictions...
              </div>
            ) : predictionsData && predictionsData.rows.length > 0 ? (
              <div className="space-y-4">
                <div className="overflow-x-auto border border-slate-800 rounded-lg">
                  <table className="w-full text-left text-xs border-collapse font-mono">
                    <thead className="bg-slate-900 border-b border-slate-800 text-slate-400 text-[11px]">
                      <tr>
                        <th className="py-2.5 px-3">Source Dataset Row #</th>
                        <th className="py-2.5 px-3">Prediction</th>
                        {resultsSummary.algorithm === 'isolation_forest' && (
                          <th className="py-2.5 px-3">Anomaly Score (-df)</th>
                        )}
                        {resultsSummary.algorithm === 'dbscan' && (
                          <th className="py-2.5 px-3">Cluster ID</th>
                        )}
                        {resultsSummary.has_ground_truth && (
                          <th className="py-2.5 px-3">True Label</th>
                        )}
                        <th className="py-2.5 px-3">Port / Duration Preview</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/40 bg-slate-950/40">
                      {predictionsData.rows.map((row) => (
                        <tr key={row.row_index} className="hover:bg-slate-800/30">
                          <td className="py-2 px-3 text-slate-500 font-mono">{row.row_index}</td>
                          <td className="py-2 px-3">
                            <span
                              className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                                row.prediction === 1
                                  ? 'bg-rose-500/10 text-rose-400 border border-rose-500/20'
                                  : 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                              }`}
                            >
                              {row.prediction_label.toUpperCase()}
                            </span>
                          </td>
                          {resultsSummary.algorithm === 'isolation_forest' && (
                            <td className="py-2 px-3 text-cyan-300">
                              {row.score !== undefined && row.score !== null ? row.score.toFixed(6) : '-'}
                            </td>
                          )}
                          {resultsSummary.algorithm === 'dbscan' && (
                            <td className="py-2 px-3 text-blue-300">
                              {row.cluster_id === -1 ? 'Noise (-1)' : `Cluster ${row.cluster_id}`}
                            </td>
                          )}
                          {resultsSummary.has_ground_truth && (
                            <td className="py-2 px-3 text-slate-300">
                              {row.ground_truth || '-'}
                            </td>
                          )}
                          <td className="py-2 px-3 text-slate-400 text-[11px]">
                            {Object.entries(row.features_preview)
                              .slice(0, 2)
                              .map(([k, v]) => `${k}: ${v}`)
                              .join(' | ')}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                {/* Pagination Controls */}
                <div className="flex items-center justify-between text-xs text-slate-400 pt-2 font-mono">
                  <div>
                    Page {predictionsData.page} of {predictionsData.total_pages} ({predictionsData.total_rows.toLocaleString()} matching rows)
                  </div>
                  <div className="flex space-x-2">
                    <button
                      disabled={page <= 1}
                      onClick={() => handlePageChange(page - 1)}
                      className="px-3 py-1 rounded bg-slate-800 hover:bg-slate-700 disabled:opacity-40 disabled:cursor-not-allowed"
                    >
                      Previous
                    </button>
                    <button
                      disabled={page >= predictionsData.total_pages}
                      onClick={() => handlePageChange(page + 1)}
                      className="px-3 py-1 rounded bg-slate-800 hover:bg-slate-700 disabled:opacity-40 disabled:cursor-not-allowed"
                    >
                      Next
                    </button>
                  </div>
                </div>
              </div>
            ) : (
              <div className="p-8 text-center text-xs text-slate-500">
                No matching prediction records found for this filter.
              </div>
            )}
          </div>
        </div>
      ) : (
        <div className="p-12 text-center text-xs text-slate-500 border border-dashed border-slate-800 rounded-xl">
          No analysis job selected or available. Run a detection from the 'Anomaly Detection' page.
        </div>
      )}
    </div>
  );
};
