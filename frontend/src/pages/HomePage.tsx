import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  ShieldAlert,
  ArrowRight,
  Activity,
  Layers,
  Database,
  CheckCircle2,
  AlertTriangle,
  Award,
  Terminal,
  Cpu,
  BarChart3
} from 'lucide-react';
import { api } from '../services/api';
import { ResearchStatus } from '../types';

export const HomePage: React.FC = () => {
  const [researchStatus, setResearchStatus] = useState<ResearchStatus | null>(null);

  useEffect(() => {
    api.getResearchStatus().then(setResearchStatus).catch(console.error);
  }, []);

  return (
    <div className="space-y-12">
      {/* Hero Section */}
      <div className="relative overflow-hidden rounded-2xl bg-gradient-to-b from-slate-900 to-slate-950 border border-slate-800 p-8 md:p-12 shadow-2xl">
        <div className="absolute top-0 right-0 -mt-12 -mr-12 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl pointer-events-none" />
        <div className="absolute bottom-0 left-0 -mb-12 -ml-12 w-96 h-96 bg-blue-500/10 rounded-full blur-3xl pointer-events-none" />

        <div className="relative z-10 max-w-3xl space-y-6">
          <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/20 text-cyan-400 text-xs font-mono">
            <span className="w-2 h-2 rounded-full bg-cyan-400 animate-ping" />
            <span>Academic Research Project • Team 21</span>
          </div>

          <h1 className="text-3xl md:text-5xl font-extrabold tracking-tight text-white leading-tight">
            Network Intrusion Detection System
          </h1>

          <p className="text-lg md:text-xl text-slate-300 font-light leading-relaxed">
            Anomaly-Based Network Traffic Analysis Using <span className="text-cyan-400 font-medium">Isolation Forest</span> and <span className="text-blue-400 font-medium">DBSCAN</span> on the CIC-IDS2017 Benchmark.
          </p>

          <p className="text-sm text-slate-400 leading-relaxed">
            Evaluating out-of-sample decision boundary scoring versus transductive density clustering across 2.52M network flows, 70 flow features, and 14 distinct cyber attack vectors.
          </p>

          <div className="flex flex-wrap gap-4 pt-4">
            <Link
              to="/upload"
              className="inline-flex items-center space-x-2 px-6 py-3 rounded-lg bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-slate-950 font-semibold shadow-lg shadow-cyan-500/20 transition-all"
            >
              <span>Start Analysis</span>
              <ArrowRight className="w-4 h-4" />
            </Link>

            <Link
              to="/dashboard"
              className="inline-flex items-center space-x-2 px-6 py-3 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-200 font-medium transition-all"
            >
              <span>Explore Dashboard</span>
            </Link>

            <Link
              to="/performance"
              className="inline-flex items-center space-x-2 px-6 py-3 rounded-lg bg-slate-900 hover:bg-slate-800 border border-slate-800 text-cyan-400 font-medium transition-all"
            >
              <span>View Experiment Results</span>
            </Link>
          </div>
        </div>
      </div>

      {/* Dataset & Institution Notice */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="p-6 rounded-xl bg-slate-900/60 border border-slate-800/80 space-y-3">
          <div className="flex items-center space-x-3 text-cyan-400">
            <Award className="w-5 h-5" />
            <h3 className="font-semibold text-slate-100 text-sm">Academic Affiliation</h3>
          </div>
          <div className="text-xs text-slate-400 space-y-1">
            <div className="font-medium text-slate-200">KLH University</div>
            <div>KLH Bachupally Campus, Hyderabad</div>
            <div>Department of Computer Science and Engineering</div>
            <div className="text-cyan-400/90 pt-1">Guide: Mr. Shaik Asif • Team 21</div>
          </div>
        </div>

        <div className="p-6 rounded-xl bg-slate-900/60 border border-slate-800/80 space-y-3">
          <div className="flex items-center space-x-3 text-blue-400">
            <Database className="w-5 h-5" />
            <h3 className="font-semibold text-slate-100 text-sm">Dataset Benchmark</h3>
          </div>
          <div className="text-xs text-slate-400 space-y-1">
            <div className="font-medium text-slate-200">CIC-IDS2017 (Canadian Institute for Cybersecurity)</div>
            <div>2,522,362 deduplicated flows • 70 flow features</div>
            <div>14 attack categories (DoS, DDoS, PortScan, Patator, Web, Bot)</div>
            <div className="text-slate-500 pt-1">
              {researchStatus?.all_present ? (
                <span className="text-emerald-400 flex items-center gap-1">
                  <CheckCircle2 className="w-3.5 h-3.5 inline" /> 8 CSV files verified locally
                </span>
              ) : (
                <span className="text-amber-400/90 flex items-center gap-1">
                  <AlertTriangle className="w-3.5 h-3.5 inline" /> Saved models & benchmarks active
                </span>
              )}
            </div>
          </div>
        </div>

        <div className="p-6 rounded-xl bg-slate-900/60 border border-slate-800/80 space-y-3">
          <div className="flex items-center space-x-3 text-emerald-400">
            <Activity className="w-5 h-5" />
            <h3 className="font-semibold text-slate-100 text-sm">Scientific Reproducibility</h3>
          </div>
          <div className="text-xs text-slate-400 space-y-1">
            <div>Frozen manifests with SHA-256 artifact verification</div>
            <div>Controlled train/test split (seed 42)</div>
            <div>Zero data leakage: preprocessors fitted on benign train only</div>
            <div className="text-emerald-400/90 pt-1">All reported research figures preserved</div>
          </div>
        </div>
      </div>

      {/* Two Core Methodologies */}
      <div className="space-y-4">
        <h2 className="text-xl font-bold text-slate-100 flex items-center space-x-2">
          <Layers className="w-5 h-5 text-cyan-400" />
          <span>Core Detection Methodologies</span>
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Isolation Forest Card */}
          <div className="p-6 rounded-xl bg-slate-900/40 border border-slate-800 space-y-4 hover:border-cyan-500/30 transition-all">
            <div className="flex items-center justify-between">
              <span className="px-2.5 py-1 rounded bg-cyan-500/10 text-cyan-400 font-mono text-xs border border-cyan-500/20">
                Phase 2A • Out-of-Sample
              </span>
              <span className="text-xs text-slate-400 font-mono">Precision: 83.87%</span>
            </div>
            <h3 className="text-lg font-bold text-slate-100">Isolation Forest (iForest)</h3>
            <p className="text-xs text-slate-400 leading-relaxed">
              Trained exclusively on 1,467,538 BENIGN network flows (no attack flows seen during training). Evaluated on 1,054,824 unseen test flows. Anomaly scores computed as <code className="text-cyan-300">-decision_function(X)</code> where higher scores indicate greater isolation abnormality.
            </p>
            <div className="grid grid-cols-3 gap-2 pt-2 text-center text-xs">
              <div className="p-2 rounded bg-slate-950/60 border border-slate-800/60">
                <div className="text-slate-400 text-[10px]">Recall</div>
                <div className="font-semibold text-cyan-400">57.89%</div>
              </div>
              <div className="p-2 rounded bg-slate-950/60 border border-slate-800/60">
                <div className="text-slate-400 text-[10px]">F1 Score</div>
                <div className="font-semibold text-cyan-400">68.50%</div>
              </div>
              <div className="p-2 rounded bg-slate-950/60 border border-slate-800/60">
                <div className="text-slate-400 text-[10px]">ROC-AUC</div>
                <div className="font-semibold text-cyan-400">0.8082</div>
              </div>
            </div>
          </div>

          {/* DBSCAN Card */}
          <div className="p-6 rounded-xl bg-slate-900/40 border border-slate-800 space-y-4 hover:border-blue-500/30 transition-all">
            <div className="flex items-center justify-between">
              <span className="px-2.5 py-1 rounded bg-blue-500/10 text-blue-400 font-mono text-xs border border-blue-500/20">
                Phase 2B • Transductive Density
              </span>
              <span className="text-xs text-slate-400 font-mono">Precision: 10.61%</span>
            </div>
            <h3 className="text-lg font-bold text-slate-100">DBSCAN Density Clustering</h3>
            <p className="text-xs text-slate-400 leading-relaxed">
              Clusters data into high-density regions and flags points outside dense clusters as noise (<code className="text-blue-300">label -1</code>). Parameters (<code className="text-blue-300">eps=2.552, min_samples=10</code>) determined via unsupervised k-distance graph analysis. Evaluated transductively on 50,000 flows.
            </p>
            <div className="grid grid-cols-3 gap-2 pt-2 text-center text-xs">
              <div className="p-2 rounded bg-slate-950/60 border border-slate-800/60">
                <div className="text-slate-400 text-[10px]">Recall</div>
                <div className="font-semibold text-blue-400">2.05%</div>
              </div>
              <div className="p-2 rounded bg-slate-950/60 border border-slate-800/60">
                <div className="text-slate-400 text-[10px]">F1 Score</div>
                <div className="font-semibold text-blue-400">3.44%</div>
              </div>
              <div className="p-2 rounded bg-slate-950/60 border border-slate-800/60">
                <div className="text-slate-400 text-[10px]">Noise Flagged</div>
                <div className="font-semibold text-blue-400">3.22%</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Methodological Qualification Alert */}
      <div className="p-5 rounded-xl bg-cyan-950/30 border border-cyan-800/50 flex items-start space-x-4 text-xs text-cyan-200">
        <AlertTriangle className="w-5 h-5 text-cyan-400 shrink-0 mt-0.5" />
        <div className="space-y-1">
          <div className="font-bold text-cyan-300">Mandatory Methodological Qualification</div>
          <p className="leading-relaxed text-slate-300">
            Isolation Forest was evaluated on an unseen test population after being trained exclusively on BENIGN traffic, whereas DBSCAN was evaluated transductively on a representative mixed sample. Therefore, their raw metric values should be interpreted within their respective experimental designs rather than as a direct apples-to-apples benchmark. High-density attacks (e.g. DoS floods) group into dense clusters under DBSCAN rather than remaining isolated noise points.
          </p>
        </div>
      </div>
    </div>
  );
};
