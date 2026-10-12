import React, { useEffect, useState } from 'react';
import {
  Cpu,
  Play,
  RotateCw,
  CheckCircle2,
  AlertCircle,
  Clock,
  Layers,
  ShieldCheck,
  FileSpreadsheet,
  Settings,
  Activity,
  ArrowRight
} from 'lucide-react';
import { api } from '../services/api';
import { DatasetInfo, TrainingJobStatus } from '../types';

export const TrainingPage: React.FC = () => {
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState<string>('');
  const [variant, setVariant] = useState<string>('with_port');
  const [nEstimators, setNEstimators] = useState<number>(200);
  const [contamination, setContamination] = useState<number>(0.01);
  const [randomState, setRandomState] = useState<number>(42);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [activeJob, setActiveJob] = useState<TrainingJobStatus | null>(null);
  const [trainingHistory, setTrainingHistory] = useState<TrainingJobStatus[]>([]);

  useEffect(() => {
    loadInitialData();
  }, []);

  const loadInitialData = async () => {
    try {
      const [dsList, history] = await Promise.all([
        api.getDatasets(),
        api.getTrainingJobs()
      ]);
      setDatasets(dsList);
      if (dsList.length > 0 && !selectedDatasetId) {
        setSelectedDatasetId(dsList[0].id);
      }
      setTrainingHistory(history);
    } catch (err: any) {
      console.error(err);
    }
  };

  // Poll active training job
  useEffect(() => {
    if (!activeJobId) return;

    const interval = setInterval(async () => {
      try {
        const job = await api.getTrainingJobStatus(activeJobId);
        setActiveJob(job);
        if (job.status === 'completed' || job.status === 'failed') {
          setActiveJobId(null);
          // Refresh history
          const updated = await api.getTrainingJobs();
          setTrainingHistory(updated);
        }
      } catch (e) {
        console.error(e);
      }
    }, 1500);

    return () => clearInterval(interval);
  }, [activeJobId]);

  const handleStartTraining = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    if (!selectedDatasetId) {
      setError('Please select a compatible dataset.');
      return;
    }

    setLoading(true);
    try {
      const res = await api.createTrainingJob({
        dataset_id: selectedDatasetId,
        variant,
        n_estimators: nEstimators,
        contamination,
        random_state: randomState
      });
      setActiveJobId(res.job_id);
      setActiveJob({
        job_id: res.job_id,
        status: 'queued',
        progress: 0,
        current_stage: 'Queued in worker pool',
        dataset_id: selectedDatasetId,
        variant,
        created_at: new Date().toISOString()
      });
    } catch (err: any) {
      setError(err.message || 'Failed to submit training job');
    } finally {
      setLoading(false);
    }
  };

  const selectedDataset = datasets.find((d) => d.id === selectedDatasetId);

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-white tracking-tight flex items-center space-x-2">
          <Cpu className="w-6 h-6 text-cyan-400" />
          <span>Model Training & Research Reproduction</span>
        </h1>
        <p className="text-xs text-slate-400">
          Execute controlled retraining runs of Isolation Forest baselines on benign network flows with reproducible hyperparameters
        </p>
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-red-950/40 border border-red-500/30 flex items-center space-x-3 text-red-300 text-xs">
          <AlertCircle className="w-5 h-5 shrink-0 text-red-400" />
          <span>{error}</span>
        </div>
      )}

      {/* Active Job Progress Banner */}
      {activeJob && (
        <div className="p-6 rounded-xl bg-cyan-950/30 border border-cyan-500/40 space-y-4 shadow-lg shadow-cyan-950/20">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <RotateCw className="w-5 h-5 text-cyan-400 animate-spin" />
              <div>
                <h3 className="font-semibold text-slate-100 text-sm">
                  Training Isolation Forest ({activeJob.variant === 'with_port' ? 'Baseline with Port' : 'Port Ablated'})
                </h3>
                <p className="text-xs text-cyan-300 font-mono">{activeJob.current_stage}</p>
              </div>
            </div>
            <span className="text-xs font-mono px-2 py-0.5 rounded bg-cyan-500/20 text-cyan-300 border border-cyan-500/30">
              {activeJob.status.toUpperCase()} ({activeJob.progress}%)
            </span>
          </div>

          <div className="w-full bg-slate-900 rounded-full h-2 overflow-hidden border border-slate-800">
            <div
              className="bg-gradient-to-r from-cyan-500 to-blue-500 h-2 rounded-full transition-all duration-300"
              style={{ width: `${Math.max(activeJob.progress, 5)}%` }}
            />
          </div>

          {activeJob.status === 'completed' && activeJob.evaluation_metrics && (
            <div className="pt-2 border-t border-cyan-500/20 grid grid-cols-2 md:grid-cols-4 gap-4 text-xs font-mono">
              <div>
                <div className="text-slate-400">Trained Flows:</div>
                <div className="text-slate-200 font-semibold">{activeJob.evaluation_metrics.training_rows.toLocaleString()}</div>
              </div>
              <div>
                <div className="text-slate-400">Features:</div>
                <div className="text-slate-200 font-semibold">{activeJob.evaluation_metrics.feature_count} cols</div>
              </div>
              <div>
                <div className="text-slate-400">Internal Anomaly Rate:</div>
                <div className="text-emerald-400 font-semibold">{activeJob.evaluation_metrics.internal_anomaly_rate}%</div>
              </div>
              <div>
                <div className="text-slate-400">Elapsed Time:</div>
                <div className="text-cyan-300 font-semibold">{activeJob.execution_time_s?.toFixed(2)}s</div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Main Form and Guidance Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Form */}
        <div className="lg:col-span-2 p-6 rounded-xl bg-slate-900/40 border border-slate-800 space-y-6">
          <div className="flex items-center space-x-2 border-b border-slate-800 pb-3">
            <Settings className="w-4 h-4 text-cyan-400" />
            <h2 className="font-semibold text-slate-100 text-sm">Hyperparameter & Dataset Configuration</h2>
          </div>

          <form onSubmit={handleStartTraining} className="space-y-6">
            {/* Dataset Selector */}
            <div className="space-y-2">
              <label className="text-xs font-medium text-slate-300 block">Select Compatible Training Dataset</label>
              <select
                value={selectedDatasetId}
                onChange={(e) => setSelectedDatasetId(e.target.value)}
                className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-cyan-500"
              >
                {datasets.map((ds) => (
                  <option key={ds.id} value={ds.id}>
                    {ds.filename} ({ds.row_count.toLocaleString()} rows, {ds.column_count} cols)
                  </option>
                ))}
              </select>
              {selectedDataset && (
                <div className="flex items-center space-x-3 text-[11px] text-slate-400 pt-1">
                  <span>Has Ground Truth Labels: <strong className="text-slate-200">{selectedDataset.has_labels ? 'Yes' : 'No'}</strong></span>
                  <span>•</span>
                  <span>Deduplication Ready: <strong className="text-slate-200">Yes</strong></span>
                </div>
              )}
            </div>

            {/* Model Variant */}
            <div className="space-y-2">
              <label className="text-xs font-medium text-slate-300 block">Research Architecture Variant</label>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <label
                  className={`p-4 rounded-xl border cursor-pointer transition-all flex flex-col justify-between ${
                    variant === 'with_port'
                      ? 'bg-cyan-950/20 border-cyan-500 text-slate-100'
                      : 'bg-slate-950/40 border-slate-800 text-slate-400 hover:border-slate-700'
                  }`}
                >
                  <div className="space-y-1">
                    <div className="flex items-center justify-between">
                      <span className="font-semibold text-xs text-slate-200">Baseline with Port</span>
                      <input
                        type="radio"
                        name="variant"
                        value="with_port"
                        checked={variant === 'with_port'}
                        onChange={() => setVariant('with_port')}
                        className="text-cyan-500 focus:ring-0"
                      />
                    </div>
                    <p className="text-[11px] text-slate-400">
                      70 features. Preserves Destination Port (Phase 2A reproduction).
                    </p>
                  </div>
                </label>

                <label
                  className={`p-4 rounded-xl border cursor-pointer transition-all flex flex-col justify-between ${
                    variant === 'without_port'
                      ? 'bg-cyan-950/20 border-cyan-500 text-slate-100'
                      : 'bg-slate-950/40 border-slate-800 text-slate-400 hover:border-slate-700'
                  }`}
                >
                  <div className="space-y-1">
                    <div className="flex items-center justify-between">
                      <span className="font-semibold text-xs text-slate-200">Destination Port Ablated</span>
                      <input
                        type="radio"
                        name="variant"
                        value="without_port"
                        checked={variant === 'without_port'}
                        onChange={() => setVariant('without_port')}
                        className="text-cyan-500 focus:ring-0"
                      />
                    </div>
                    <p className="text-[11px] text-slate-400">
                      69 features. Excludes Destination Port to verify protocol dynamics (Phase 2C).
                    </p>
                  </div>
                </label>
              </div>
            </div>

            {/* Hyperparameters */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              <div className="space-y-1">
                <label className="text-xs text-slate-300 font-medium">Estimators (Trees)</label>
                <input
                  type="number"
                  min="50"
                  max="500"
                  step="10"
                  value={nEstimators}
                  onChange={(e) => setNEstimators(Number(e.target.value))}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-cyan-500"
                />
                <span className="text-[10px] text-slate-500">Research default: 200</span>
              </div>

              <div className="space-y-1">
                <label className="text-xs text-slate-300 font-medium">Contamination</label>
                <input
                  type="number"
                  min="0.001"
                  max="0.2"
                  step="0.005"
                  value={contamination}
                  onChange={(e) => setContamination(Number(e.target.value))}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-cyan-500"
                />
                <span className="text-[10px] text-slate-500">Auto threshold approx: 0.01</span>
              </div>

              <div className="space-y-1">
                <label className="text-xs text-slate-300 font-medium">Random State Seed</label>
                <input
                  type="number"
                  min="0"
                  max="9999"
                  value={randomState}
                  onChange={(e) => setRandomState(Number(e.target.value))}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-cyan-500"
                />
                <span className="text-[10px] text-slate-500">Fixed seed: 42</span>
              </div>
            </div>

            <div className="pt-2">
              <button
                type="submit"
                disabled={loading || activeJobId !== null}
                className="w-full py-2.5 px-4 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white rounded-lg text-xs font-semibold shadow-md shadow-cyan-600/20 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2 transition-all"
              >
                {loading || activeJobId !== null ? (
                  <>
                    <RotateCw className="w-4 h-4 animate-spin" />
                    <span>Training in Progress...</span>
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4" />
                    <span>Launch Model Training Task</span>
                  </>
                )}
              </button>
            </div>
          </form>
        </div>

        {/* Academic Protocols & Safeguards Notice */}
        <div className="p-6 rounded-xl bg-slate-900/40 border border-slate-800 space-y-4 text-xs">
          <div className="flex items-center space-x-2 text-cyan-400 font-semibold border-b border-slate-800 pb-3">
            <ShieldCheck className="w-4 h-4" />
            <span>Academic Integrity & Preservation</span>
          </div>

          <div className="space-y-3 text-slate-400 leading-relaxed text-[11px]">
            <p>
              <strong className="text-slate-200">1. Frozen Benchmark Preservation:</strong> Training executions produce versioned artifacts under <code className="text-cyan-400">jobs_data/reproduced_models/</code>. Original paper models in <code className="text-cyan-400">models/</code> are strictly write-protected.
            </p>
            <p>
              <strong className="text-slate-200">2. Unsupervised Benign Filtering:</strong> When labels are detected, the training worker automatically extracts only BENIGN rows to reproduce true out-of-sample anomaly detection without data leakage.
            </p>
            <p>
              <strong className="text-slate-200">3. Resource Management:</strong> The job executes inside a bounded <code className="text-cyan-400">ThreadPoolExecutor</code> worker queue with explicit memory garbage collection.
            </p>
          </div>
        </div>
      </div>

      {/* Reproduction & Training History Table */}
      <div className="p-6 rounded-xl bg-slate-900/40 border border-slate-800 space-y-4">
        <h3 className="font-semibold text-slate-200 text-xs flex items-center space-x-2">
          <Activity className="w-4 h-4 text-cyan-400" />
          <span>Completed Training Runs & Reproduced Models ({trainingHistory.length})</span>
        </h3>

        {trainingHistory.length === 0 ? (
          <div className="text-center py-8 text-xs text-slate-500">
            No training jobs have been executed yet in this session. Configure and launch a run above.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs text-slate-300">
              <thead className="bg-slate-950/60 text-slate-400 font-mono text-[11px] border-b border-slate-800">
                <tr>
                  <th className="py-2.5 px-3">Job ID</th>
                  <th className="py-2.5 px-3">Variant</th>
                  <th className="py-2.5 px-3">Status</th>
                  <th className="py-2.5 px-3">Trained Flows</th>
                  <th className="py-2.5 px-3">Anomaly Rate</th>
                  <th className="py-2.5 px-3">Time</th>
                  <th className="py-2.5 px-3">Completed At</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 font-mono text-[11px]">
                {trainingHistory.map((tj) => (
                  <tr key={tj.job_id} className="hover:bg-slate-850/40">
                    <td className="py-2.5 px-3 text-cyan-400">{tj.job_id.slice(0, 8)}...</td>
                    <td className="py-2.5 px-3 text-slate-300">
                      {tj.variant === 'with_port' ? 'Baseline (70 feat)' : 'Ablated (69 feat)'}
                    </td>
                    <td className="py-2.5 px-3">
                      <span
                        className={`px-2 py-0.5 rounded text-[10px] ${
                          tj.status === 'completed'
                            ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                            : tj.status === 'failed'
                            ? 'bg-red-500/10 text-red-400 border border-red-500/20'
                            : 'bg-cyan-500/10 text-cyan-400 border border-cyan-500/20'
                        }`}
                      >
                        {tj.status.toUpperCase()}
                      </span>
                    </td>
                    <td className="py-2.5 px-3 text-slate-300">
                      {tj.training_rows ? tj.training_rows.toLocaleString() : '—'}
                    </td>
                    <td className="py-2.5 px-3 text-emerald-400">
                      {tj.evaluation_metrics ? `${tj.evaluation_metrics.internal_anomaly_rate}%` : '—'}
                    </td>
                    <td className="py-2.5 px-3 text-slate-400">
                      {tj.execution_time_s ? `${tj.execution_time_s.toFixed(2)}s` : '—'}
                    </td>
                    <td className="py-2.5 px-3 text-slate-500">
                      {tj.completed_at ? new Date(tj.completed_at).toLocaleTimeString() : '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};
export default TrainingPage;
