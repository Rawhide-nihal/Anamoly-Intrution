import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  Activity,
  ShieldCheck,
  AlertOctagon,
  Percent,
  Clock,
  Layers,
  FileText,
  UploadCloud,
  CheckCircle2,
  TrendingUp,
  Database
} from 'lucide-react';
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  Tooltip,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Legend
} from 'recharts';
import { api } from '../services/api';
import { AnalysisJobStatus, AnalysisResultSummary } from '../types';

export const DashboardPage: React.FC = () => {
  const [jobs, setJobs] = useState<AnalysisJobStatus[]>([]);
  const [selectedJobResult, setSelectedJobResult] = useState<AnalysisResultSummary | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadDashboardData();
  }, []);

  const loadDashboardData = async () => {
    try {
      setLoading(true);
      const jobsList = await api.getJobs();
      setJobs(jobsList);
      const completed = jobsList.find((j) => j.status === 'completed');
      if (completed) {
        const res = await api.getJobResults(completed.job_id);
        setSelectedJobResult(res);
      }
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  const pieData = selectedJobResult
    ? [
        { name: 'Normal Traffic', value: selectedJobResult.normal_count, color: '#10b981' },
        { name: 'Predicted Anomalies', value: selectedJobResult.anomaly_count, color: '#ef4444' }
      ]
    : [
        { name: 'Normal (Historical IF)', value: 581525, color: '#10b981' },
        { name: 'Anomaly (Historical IF)', value: 473299, color: '#ef4444' }
      ];

  const benchmarkCompData = [
    { metric: 'Precision', iForest: 83.87, dbscan: 10.61 },
    { metric: 'Recall', iForest: 57.89, dbscan: 2.05 },
    { metric: 'F1 Score', iForest: 68.5, dbscan: 3.44 },
    { metric: 'FPR', iForest: 7.54, dbscan: 3.46 }
  ];

  return (
    <div className="space-y-8">
      {/* Top Banner */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-800">
        <div>
          <h1 className="text-2xl font-bold text-white tracking-tight">Detection Analytics Dashboard</h1>
          <p className="text-xs text-slate-400">
            Real-time inference metrics & comparative evaluation against frozen CIC-IDS2017 benchmarks
          </p>
        </div>
        <div className="flex items-center space-x-3">
          <Link
            to="/upload"
            className="inline-flex items-center space-x-2 px-4 py-2 rounded-lg bg-cyan-500 hover:bg-cyan-400 text-slate-950 text-xs font-semibold shadow transition-all"
          >
            <UploadCloud className="w-4 h-4" />
            <span>Upload New Flows</span>
          </Link>
          <Link
            to="/detect"
            className="inline-flex items-center space-x-2 px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium border border-slate-700 transition-all"
          >
            <Activity className="w-4 h-4" />
            <span>Run Detection</span>
          </Link>
        </div>
      </div>

      {/* Metrics Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="p-5 rounded-xl bg-slate-900/60 border border-slate-800/80 space-y-2">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span>Total Flows Analyzed</span>
            <Database className="w-4 h-4 text-cyan-400" />
          </div>
          <div className="text-2xl font-bold text-white font-mono">
            {selectedJobResult
              ? selectedJobResult.total_rows.toLocaleString()
              : '1,054,824'}
          </div>
          <div className="text-[11px] text-slate-500">
            {selectedJobResult ? selectedJobResult.dataset_name : 'Phase 2A Test Partition (Frozen)'}
          </div>
        </div>

        <div className="p-5 rounded-xl bg-slate-900/60 border border-slate-800/80 space-y-2">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span>Normal Traffic Flows</span>
            <ShieldCheck className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="text-2xl font-bold text-emerald-400 font-mono">
            {selectedJobResult
              ? selectedJobResult.normal_count.toLocaleString()
              : '581,525'}
          </div>
          <div className="text-[11px] text-slate-500">
            {selectedJobResult
              ? `${(100 - selectedJobResult.anomaly_percentage).toFixed(1)}% of total`
              : '55.1% of test set'}
          </div>
        </div>

        <div className="p-5 rounded-xl bg-slate-900/60 border border-slate-800/80 space-y-2">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span>Flagged Anomalies</span>
            <AlertOctagon className="w-4 h-4 text-rose-400" />
          </div>
          <div className="text-2xl font-bold text-rose-400 font-mono">
            {selectedJobResult
              ? selectedJobResult.anomaly_count.toLocaleString()
              : '473,299'}
          </div>
          <div className="text-[11px] text-slate-500">
            {selectedJobResult
              ? `${selectedJobResult.anomaly_percentage.toFixed(2)}% anomaly rate`
              : '27.87% benchmark anomaly rate'}
          </div>
        </div>

        <div className="p-5 rounded-xl bg-slate-900/60 border border-slate-800/80 space-y-2">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span>Active Algorithm</span>
            <Layers className="w-4 h-4 text-blue-400" />
          </div>
          <div className="text-xl font-bold text-cyan-300 font-mono truncate">
            {selectedJobResult ? selectedJobResult.algorithm.replace('_', ' ').toUpperCase() : 'ISOLATION FOREST'}
          </div>
          <div className="text-[11px] text-slate-500">
            {selectedJobResult ? `Execution: ${selectedJobResult.execution_time_s.toFixed(2)}s` : 'Frozen Model: 200 trees'}
          </div>
        </div>
      </div>

      {/* Visual Analytics Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Anomaly Distribution Chart */}
        <div className="p-6 rounded-xl bg-slate-900/40 border border-slate-800 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="font-semibold text-slate-200 text-sm">Traffic Classification Distribution</h3>
            <span className="text-xs text-slate-500 font-mono">
              {selectedJobResult ? 'Current Analysis' : 'Frozen Benchmark'}
            </span>
          </div>

          <div className="h-64 flex items-center justify-center">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={65}
                  outerRadius={95}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {pieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip
                  formatter={(val: any) => Number(val).toLocaleString()}
                  contentStyle={{ backgroundColor: '#090d16', borderColor: '#1e293b', borderRadius: '8px', fontSize: '12px' }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>

          <div className="flex justify-center space-x-6 text-xs">
            <div className="flex items-center space-x-2">
              <span className="w-3 h-3 rounded-full bg-emerald-500" />
              <span className="text-slate-400">Normal Flows</span>
            </div>
            <div className="flex items-center space-x-2">
              <span className="w-3 h-3 rounded-full bg-rose-500" />
              <span className="text-slate-400">Anomalous Flows</span>
            </div>
          </div>
        </div>

        {/* Historical Benchmark Overview */}
        <div className="p-6 rounded-xl bg-slate-900/40 border border-slate-800 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="font-semibold text-slate-200 text-sm">Frozen Benchmark Comparison (%)</h3>
            <span className="text-xs text-slate-500 font-mono">Phase 3 Final</span>
          </div>

          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={benchmarkCompData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="metric" stroke="#64748b" fontSize={11} />
                <YAxis stroke="#64748b" fontSize={11} domain={[0, 100]} />
                <Tooltip
                  formatter={(val: any) => `${val}%`}
                  contentStyle={{ backgroundColor: '#090d16', borderColor: '#1e293b', borderRadius: '8px', fontSize: '12px' }}
                />
                <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }} />
                <Bar dataKey="iForest" name="Isolation Forest (Out-of-sample)" fill="#06b6d4" radius={[4, 4, 0, 0]} />
                <Bar dataKey="dbscan" name="DBSCAN (Transductive)" fill="#3b82f6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="text-[11px] text-slate-500 italic text-center">
            Note: Isolation Forest was evaluated on unseen test flows; DBSCAN on a 50k representative sample.
          </div>
        </div>
      </div>

      {/* Recent Analysis Jobs Table */}
      <div className="p-6 rounded-xl bg-slate-900/40 border border-slate-800 space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="font-semibold text-slate-200 text-sm">Recent Analysis Sessions</h3>
          <Link to="/results" className="text-xs text-cyan-400 hover:underline">
            View All Results
          </Link>
        </div>

        {jobs.length === 0 ? (
          <div className="p-8 text-center text-xs text-slate-500 border border-dashed border-slate-800 rounded-lg">
            No analysis jobs run yet. Click 'Run Detection' or 'Upload New Flows' to begin.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr className="border-b border-slate-800 text-slate-400 font-mono">
                  <th className="py-2.5 px-3">Job ID</th>
                  <th className="py-2.5 px-3">Algorithm</th>
                  <th className="py-2.5 px-3">Status</th>
                  <th className="py-2.5 px-3">Progress</th>
                  <th className="py-2.5 px-3">Timestamp</th>
                  <th className="py-2.5 px-3">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 font-sans">
                {jobs.slice(0, 5).map((job) => (
                  <tr key={job.job_id} className="hover:bg-slate-800/30">
                    <td className="py-2.5 px-3 font-mono text-cyan-400">{job.job_id.slice(0, 8)}...</td>
                    <td className="py-2.5 px-3 font-medium text-slate-300">
                      {job.algorithm === 'isolation_forest' ? 'Isolation Forest' : 'DBSCAN'}
                    </td>
                    <td className="py-2.5 px-3">
                      <span
                        className={`px-2 py-0.5 rounded text-[10px] font-mono ${
                          job.status === 'completed'
                            ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                            : job.status === 'failed'
                            ? 'bg-rose-500/10 text-rose-400 border border-rose-500/20'
                            : 'bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 animate-pulse'
                        }`}
                      >
                        {job.status.toUpperCase()}
                      </span>
                    </td>
                    <td className="py-2.5 px-3 font-mono">{job.progress}%</td>
                    <td className="py-2.5 px-3 text-slate-500 font-mono">
                      {new Date(job.created_at).toLocaleTimeString()}
                    </td>
                    <td className="py-2.5 px-3">
                      {job.status === 'completed' && (
                        <Link
                          to={`/results?job_id=${job.job_id}`}
                          className="text-cyan-400 hover:text-cyan-300 underline font-medium"
                        >
                          View Results
                        </Link>
                      )}
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
