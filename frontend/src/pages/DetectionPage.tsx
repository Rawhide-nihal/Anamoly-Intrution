import React, { useEffect, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import {
  Activity,
  Cpu,
  Layers,
  Sliders,
  Play,
  CheckCircle2,
  AlertTriangle,
  Clock,
  ShieldAlert,
  HelpCircle,
  Database
} from 'lucide-react';
import { api } from '../services/api';
import { DatasetInfo, ModelInfo, AnalysisJobStatus } from '../types';

export const DetectionPage: React.FC = () => {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();

  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState<string>('');
  const [selectedAlgorithm, setSelectedAlgorithm] = useState<string>('isolation_forest');
  const [selectedModelId, setSelectedModelId] = useState<string>('if_with_port');

  // DBSCAN params
  const [dbscanEps, setDbscanEps] = useState<number>(2.5522);
  const [dbscanMinSamples, setDbscanMinSamples] = useState<number>(10);
  const [sampleLimit, setSampleLimit] = useState<number>(50000);

  // Job status
  const [activeJob, setActiveJob] = useState<AnalysisJobStatus | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  useEffect(() => {
    loadOptions();
  }, []);

  const loadOptions = async () => {
    try {
      const [dsList, mList] = await Promise.all([api.getDatasets(), api.getModels()]);
      setDatasets(dsList);
      setModels(mList);

      const queryDs = searchParams.get('dataset_id');
      if (queryDs && dsList.some((d) => d.id === queryDs)) {
        setSelectedDatasetId(queryDs);
      } else if (dsList.length > 0) {
        setSelectedDatasetId(dsList[0].id);
      }
    } catch (e) {
      console.error(e);
    }
  };

  // Poll active job
  useEffect(() => {
    if (!activeJob || activeJob.status === 'completed' || activeJob.status === 'failed') {
      return;
    }
    const interval = setInterval(async () => {
      try {
        const status = await api.getJobStatus(activeJob.job_id);
        setActiveJob(status);
        if (status.status === 'completed') {
          clearInterval(interval);
          navigate(`/results?job_id=${status.job_id}`);
        } else if (status.status === 'failed') {
          clearInterval(interval);
          setErrorMsg(status.error_message || 'Analysis failed');
        }
      } catch (e) {
        console.error(e);
      }
    }, 1500);

    return () => clearInterval(interval);
  }, [activeJob, navigate]);

  const handleStartAnalysis = async () => {
    if (!selectedDatasetId) {
      setErrorMsg('Please select a dataset');
      return;
    }

    setIsSubmitting(true);
    setErrorMsg(null);

    try {
      const res = await api.createJob({
        dataset_id: selectedDatasetId,
        model_id: selectedAlgorithm === 'dbscan' ? 'dbscan_clustering' : selectedModelId,
        algorithm: selectedAlgorithm,
        dbscan_eps: dbscanEps,
        dbscan_min_samples: dbscanMinSamples,
        sample_limit: sampleLimit
      });

      const initialStatus = await api.getJobStatus(res.job_id);
      setActiveJob(initialStatus);
    } catch (err: any) {
      setErrorMsg(err.message || 'Failed to start job');
    } finally {
      setIsSubmitting(false);
    }
  };

  const selectedDataset = datasets.find((d) => d.id === selectedDatasetId);

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-white tracking-tight">Execute Anomaly Detection</h1>
        <p className="text-xs text-slate-400">
          Run out-of-sample Isolation Forest inference or transductive DBSCAN density clustering on network traffic flows
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Configuration Form */}
        <div className="lg:col-span-2 space-y-6">
          {/* Step 1: Dataset Selection */}
          <div className="p-6 rounded-xl bg-slate-900/40 border border-slate-800 space-y-4">
            <h3 className="font-semibold text-slate-100 text-sm flex items-center space-x-2">
              <span className="w-5 h-5 rounded-full bg-cyan-500/20 text-cyan-400 text-xs flex items-center justify-center font-mono font-bold">1</span>
              <span>Select Network Traffic Dataset</span>
            </h3>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {datasets.map((ds) => (
                <div
                  key={ds.id}
                  onClick={() => setSelectedDatasetId(ds.id)}
                  className={`cursor-pointer p-3.5 rounded-lg border text-xs transition-all ${
                    selectedDatasetId === ds.id
                      ? 'bg-cyan-950/20 border-cyan-500 text-slate-100'
                      : 'bg-slate-900/60 border-slate-800 hover:border-slate-700 text-slate-400'
                  }`}
                >
                  <div className="font-mono text-cyan-400 font-medium truncate">{ds.filename}</div>
                  <div className="text-[11px] text-slate-400 mt-1">
                    {ds.row_count.toLocaleString()} flows • {ds.has_labels ? 'Labeled' : 'Unlabeled'}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Step 2: Algorithm Selection */}
          <div className="p-6 rounded-xl bg-slate-900/40 border border-slate-800 space-y-4">
            <h3 className="font-semibold text-slate-100 text-sm flex items-center space-x-2">
              <span className="w-5 h-5 rounded-full bg-cyan-500/20 text-cyan-400 text-xs flex items-center justify-center font-mono font-bold">2</span>
              <span>Select Detection Algorithm</span>
            </h3>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div
                onClick={() => {
                  setSelectedAlgorithm('isolation_forest');
                  setSelectedModelId('if_with_port');
                }}
                className={`cursor-pointer p-4 rounded-xl border text-xs space-y-2 transition-all ${
                  selectedAlgorithm === 'isolation_forest'
                    ? 'bg-cyan-950/30 border-cyan-500 text-slate-100 ring-1 ring-cyan-500'
                    : 'bg-slate-900/40 border-slate-800 text-slate-400 hover:border-slate-700'
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="font-bold text-slate-100 text-sm">Isolation Forest</span>
                  <span className="px-2 py-0.5 rounded bg-cyan-500/10 text-cyan-400 text-[10px] font-mono border border-cyan-500/20">
                    Out-of-Sample
                  </span>
                </div>
                <p className="text-[11px] text-slate-400 leading-relaxed">
                  Evaluates flows against 200 isolation trees trained on 1.46M benign flows. Outputs continuous anomaly scores <code className="text-cyan-300">-decision_function(X)</code>.
                </p>
              </div>

              <div
                onClick={() => setSelectedAlgorithm('dbscan')}
                className={`cursor-pointer p-4 rounded-xl border text-xs space-y-2 transition-all ${
                  selectedAlgorithm === 'dbscan'
                    ? 'bg-blue-950/30 border-blue-500 text-slate-100 ring-1 ring-blue-500'
                    : 'bg-slate-900/40 border-slate-800 text-slate-400 hover:border-slate-700'
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="font-bold text-slate-100 text-sm">DBSCAN Clustering</span>
                  <span className="px-2 py-0.5 rounded bg-blue-500/10 text-blue-400 text-[10px] font-mono border border-blue-500/20">
                    Transductive Density
                  </span>
                </div>
                <p className="text-[11px] text-slate-400 leading-relaxed">
                  Density-based spatial clustering. Clusters flows into density groups; non-dense noise points (<code className="text-blue-300">cluster -1</code>) are flagged as anomalies.
                </p>
              </div>
            </div>

            {/* Algorithm-Specific Sub-Options */}
            {selectedAlgorithm === 'isolation_forest' ? (
              <div className="p-4 rounded-lg bg-slate-950/50 border border-slate-800/80 space-y-3">
                <div className="text-xs font-semibold text-slate-300">Model Feature Variant:</div>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
                  <label className="flex items-center space-x-2.5 cursor-pointer">
                    <input
                      type="radio"
                      name="if_model"
                      checked={selectedModelId === 'if_with_port'}
                      onChange={() => setSelectedModelId('if_with_port')}
                      className="text-cyan-500 focus:ring-0"
                    />
                    <span className="text-slate-300 font-medium">With Destination Port (70 features)</span>
                  </label>

                  <label className="flex items-center space-x-2.5 cursor-pointer">
                    <input
                      type="radio"
                      name="if_model"
                      checked={selectedModelId === 'if_without_port'}
                      onChange={() => setSelectedModelId('if_without_port')}
                      className="text-cyan-500 focus:ring-0"
                    />
                    <span className="text-slate-300 font-medium">Port Ablated (69 features)</span>
                  </label>
                </div>
              </div>
            ) : (
              <div className="p-4 rounded-lg bg-slate-950/50 border border-slate-800/80 space-y-3 text-xs">
                <div className="text-xs font-semibold text-slate-300">DBSCAN Parameters (Elbow Calibrated):</div>
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                  <div>
                    <label className="text-[11px] text-slate-400 block mb-1">Epsilon (eps):</label>
                    <input
                      type="number"
                      step="0.01"
                      value={dbscanEps}
                      onChange={(e) => setDbscanEps(parseFloat(e.target.value) || 2.55)}
                      className="w-full px-2.5 py-1.5 rounded bg-slate-900 border border-slate-700 text-slate-200 font-mono text-xs"
                    />
                  </div>

                  <div>
                    <label className="text-[11px] text-slate-400 block mb-1">Min Samples:</label>
                    <input
                      type="number"
                      value={dbscanMinSamples}
                      onChange={(e) => setDbscanMinSamples(parseInt(e.target.value) || 10)}
                      className="w-full px-2.5 py-1.5 rounded bg-slate-900 border border-slate-700 text-slate-200 font-mono text-xs"
                    />
                  </div>

                  <div>
                    <label className="text-[11px] text-slate-400 block mb-1">Max Sample Limit:</label>
                    <input
                      type="number"
                      value={sampleLimit}
                      onChange={(e) => setSampleLimit(parseInt(e.target.value) || 50000)}
                      className="w-full px-2.5 py-1.5 rounded bg-slate-900 border border-slate-700 text-slate-200 font-mono text-xs"
                    />
                  </div>
                </div>
                <p className="text-[10px] text-slate-500">
                  DBSCAN requires pairwise distance computation. Datasets over {sampleLimit.toLocaleString()} rows are uniformly sampled (seed 42) with explicit notice.
                </p>
              </div>
            )}
          </div>

          {/* Trigger Button & Errors */}
          {errorMsg && (
            <div className="p-3.5 rounded-lg bg-rose-500/10 border border-rose-500/20 text-xs text-rose-300 flex items-center space-x-2">
              <AlertTriangle className="w-4 h-4 shrink-0" />
              <span>{errorMsg}</span>
            </div>
          )}

          <button
            onClick={handleStartAnalysis}
            disabled={isSubmitting || (activeJob !== null && activeJob.status !== 'completed' && activeJob.status !== 'failed')}
            className="w-full py-3.5 px-6 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-slate-950 font-bold text-sm shadow-lg shadow-cyan-500/20 disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center justify-center space-x-2"
          >
            <Play className="w-4 h-4 fill-slate-950" />
            <span>Analyze Traffic Flows</span>
          </button>
        </div>

        {/* Execution Status / Info Panel */}
        <div className="space-y-6">
          <div className="p-6 rounded-xl bg-slate-900/40 border border-slate-800 space-y-4">
            <h3 className="font-semibold text-slate-100 text-sm flex items-center space-x-2">
              <Activity className="w-4 h-4 text-cyan-400" />
              <span>Job Execution Monitor</span>
            </h3>

            {activeJob ? (
              <div className="space-y-4 text-xs font-mono">
                <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800 space-y-2">
                  <div className="flex justify-between text-slate-400 text-[11px]">
                    <span>STATUS:</span>
                    <span className="text-cyan-400 font-bold">{activeJob.status.toUpperCase()}</span>
                  </div>
                  <div className="w-full bg-slate-800 rounded-full h-2 overflow-hidden">
                    <div
                      className="bg-cyan-500 h-2 transition-all duration-300"
                      style={{ width: `${activeJob.progress}%` }}
                    />
                  </div>
                  <div className="text-[11px] text-slate-300 font-sans">{activeJob.current_stage}</div>
                </div>

                <div className="text-[11px] text-slate-400 space-y-1">
                  <div>Job ID: <span className="text-slate-200">{activeJob.job_id.slice(0, 8)}...</span></div>
                  <div>Algorithm: <span className="text-slate-200">{activeJob.algorithm}</span></div>
                </div>
              </div>
            ) : (
              <div className="p-6 text-center text-xs text-slate-500 border border-dashed border-slate-800 rounded-lg">
                Ready to execute. Configure options and click 'Analyze Traffic Flows'.
              </div>
            )}
          </div>

          <div className="p-5 rounded-xl bg-slate-900/30 border border-slate-800 space-y-3 text-xs text-slate-400">
            <div className="font-semibold text-slate-300 flex items-center space-x-1.5">
              <HelpCircle className="w-4 h-4 text-cyan-400" />
              <span>Prediction Conventions</span>
            </div>
            <div className="space-y-1.5 leading-relaxed text-[11px]">
              <div>• <strong>0 (Normal)</strong>: Regular network flow adhering to baseline traffic distribution.</div>
              <div>• <strong>1 (Anomaly)</strong>: Statistical outlier (IF score &gt; 0) or unclustered noise point (DBSCAN label -1).</div>
              <div className="text-slate-500 pt-1">
                Note: Flagged anomaly indicates structural flow deviation, not guaranteed confirmed attack category.
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
