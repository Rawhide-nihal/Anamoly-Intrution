import React, { useEffect, useState } from 'react';
import {
  FileSpreadsheet,
  Download,
  FileText,
  Printer,
  CheckCircle2,
  AlertCircle
} from 'lucide-react';
import { api } from '../services/api';
import { AnalysisJobStatus, AnalysisResultSummary } from '../types';

export const ReportsPage: React.FC = () => {
  const [jobs, setJobs] = useState<AnalysisJobStatus[]>([]);
  const [selectedJobId, setSelectedJobId] = useState<string>('');
  const [summary, setSummary] = useState<AnalysisResultSummary | null>(null);

  useEffect(() => {
    loadJobs();
  }, []);

  const loadJobs = async () => {
    try {
      const allJobs = await api.getJobs();
      const completed = allJobs.filter((j) => j.status === 'completed');
      setJobs(completed);
      if (completed.length > 0) {
        setSelectedJobId(completed[0].job_id);
        fetchSummary(completed[0].job_id);
      }
    } catch (e) {
      console.error(e);
    }
  };

  const fetchSummary = async (jobId: string) => {
    try {
      const res = await api.getJobResults(jobId);
      setSummary(res);
    } catch (e) {
      console.error(e);
    }
  };

  const handleSelectJob = (id: string) => {
    setSelectedJobId(id);
    fetchSummary(id);
  };

  const handlePrint = () => {
    window.print();
  };

  return (
    <div className="space-y-8">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white tracking-tight">Technical Reports & Export</h1>
          <p className="text-xs text-slate-400">
            Generate formal cybersecurity audit summaries, performance reports, and downloadable data
          </p>
        </div>

        <div className="flex items-center space-x-3">
          <button
            onClick={handlePrint}
            className="inline-flex items-center space-x-2 px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 font-medium text-xs border border-slate-700 transition-all"
          >
            <Printer className="w-4 h-4" />
            <span>Print / Save PDF</span>
          </button>

          {selectedJobId && (
            <a
              href={api.getExportCsvUrl(selectedJobId)}
              download
              className="inline-flex items-center space-x-2 px-4 py-2 rounded-lg bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-semibold text-xs shadow transition-all"
            >
              <Download className="w-4 h-4" />
              <span>Export CSV</span>
            </a>
          )}
        </div>
      </div>

      {/* Select Job */}
      {jobs.length > 0 && (
        <div className="flex items-center space-x-3 overflow-x-auto pb-2">
          <span className="text-xs text-slate-400 font-mono shrink-0">Select Analysis Job:</span>
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
              {j.algorithm} • {j.job_id.slice(0, 8)}
            </button>
          ))}
        </div>
      )}

      {/* Printable Report Document Card */}
      {summary ? (
        <div className="p-8 rounded-2xl bg-slate-900/40 border border-slate-800 space-y-6 text-slate-300 font-sans print:bg-white print:text-black print:border-none">
          {/* Header */}
          <div className="border-b border-slate-800 pb-6 flex justify-between items-start">
            <div>
              <div className="text-xs uppercase font-mono tracking-widest text-cyan-400 print:text-blue-600">
                Official Intrusion Detection Evaluation Report
              </div>
              <h2 className="text-xl font-bold text-white print:text-black mt-1">
                Anomaly-Based Network Intrusion Detection Summary
              </h2>
              <div className="text-xs text-slate-400 print:text-gray-600 mt-1">
                KLH University • Department of Computer Science and Engineering • Team 21
              </div>
            </div>

            <div className="text-right text-xs font-mono text-slate-400 print:text-gray-600">
              <div>Job ID: {summary.job_id.slice(0, 12)}</div>
              <div>Generated: {new Date().toLocaleDateString()}</div>
            </div>
          </div>

          {/* Configuration Grid */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs font-mono">
            <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800/80 print:border-gray-300">
              <span className="text-slate-500 block text-[10px]">DATASET</span>
              <span className="font-semibold text-slate-200 print:text-black">{summary.dataset_name}</span>
            </div>
            <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800/80 print:border-gray-300">
              <span className="text-slate-500 block text-[10px]">ALGORITHM</span>
              <span className="font-semibold text-cyan-400 print:text-blue-600">{summary.algorithm.toUpperCase()}</span>
            </div>
            <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800/80 print:border-gray-300">
              <span className="text-slate-500 block text-[10px]">ANALYZED FLOWS</span>
              <span className="font-semibold text-slate-200 print:text-black">
                {summary.total_rows.toLocaleString()}
                {summary.is_sampled && (
                  <span className="text-amber-400 text-[10px] block">
                    (Sampled from {summary.original_dataset_rows?.toLocaleString()})
                  </span>
                )}
              </span>
            </div>
            <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800/80 print:border-gray-300">
              <span className="text-slate-500 block text-[10px]">RUNTIME</span>
              <span className="font-semibold text-slate-200 print:text-black">{summary.execution_time_s.toFixed(2)}s</span>
            </div>
          </div>

          {/* Results Analysis */}
          <div className="space-y-3">
            <h3 className="font-bold text-white print:text-black text-sm">Classification Findings</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
              <div className="p-4 rounded-lg bg-slate-950/60 border border-slate-800">
                <div className="text-slate-400">Normal Traffic Flows</div>
                <div className="text-xl font-bold text-emerald-400 font-mono mt-1">
                  {summary.normal_count.toLocaleString()}
                </div>
                <div className="text-[11px] text-slate-500 mt-0.5">
                  {(100 - summary.anomaly_percentage).toFixed(2)}% of flow sample
                </div>
              </div>

              <div className="p-4 rounded-lg bg-slate-950/60 border border-slate-800">
                <div className="text-slate-400">Flagged Anomalous Flows</div>
                <div className="text-xl font-bold text-rose-400 font-mono mt-1">
                  {summary.anomaly_count.toLocaleString()}
                </div>
                <div className="text-[11px] text-slate-500 mt-0.5">
                  {summary.anomaly_percentage.toFixed(2)}% anomaly rate
                </div>
              </div>

              <div className="p-4 rounded-lg bg-slate-950/60 border border-slate-800">
                <div className="text-slate-400">Ground Truth Status</div>
                <div className="text-xl font-bold text-slate-200 font-mono mt-1">
                  {summary.has_ground_truth ? 'VERIFIED' : 'UNLABELED'}
                </div>
                <div className="text-[11px] text-slate-500 mt-0.5">
                  {summary.has_ground_truth ? 'Supervised metrics available' : 'Pure unsupervised scoring'}
                </div>
              </div>
            </div>
          </div>

          {/* Methodological Notes */}
          <div className="p-4 rounded-lg bg-slate-950/80 border border-slate-800/80 text-xs space-y-1.5 text-slate-400">
            <div className="font-bold text-slate-300">Methodological Notes & Interpretation Guidelines</div>
            <p className="leading-relaxed">
              Anomaly scores represent statistical isolation from the baseline normal model profile. Predicted anomalies indicate deviations from ordinary behavior and should be correlated with surrounding flow signatures before attributing malicious attack intent.
            </p>
          </div>
        </div>
      ) : (
        <div className="p-8 text-center text-xs text-slate-500 border border-dashed border-slate-800 rounded-xl">
          No analysis job selected. Please run an anomaly detection analysis first.
        </div>
      )}
    </div>
  );
};
