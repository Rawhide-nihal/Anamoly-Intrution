import React, { useEffect, useState } from 'react';
import {
  BookOpen,
  FileText,
  Layers,
  CheckCircle2,
  Cpu,
  ShieldCheck,
  Terminal,
  ExternalLink
} from 'lucide-react';
import { api } from '../services/api';

export const ResearchPage: React.FC = () => {
  const [reports, setReports] = useState<Record<string, string>>({});
  const [activeReportKey, setActiveReportKey] = useState<string>('phase3_final_report.txt');

  useEffect(() => {
    api.getResearchReports().then(setReports).catch(console.error);
  }, []);

  const phases = [
    {
      id: 'phase1',
      title: 'Phase 1 — Dataset Validation & Partitioning',
      desc: 'Validated deduplication from 2.83M to 2,522,362 flows. 70/30 train/test split on 2.09M benign flows; all 425k attack flows isolated in test.',
      file: 'zero_variance_reconciliation.txt'
    },
    {
      id: 'phase2a',
      title: 'Phase 2A — Isolation Forest Out-of-Sample',
      desc: '200 trees trained on 1,467,538 BENIGN flows only. Scored 1,054,824 unseen flows. Precision: 83.87%, Recall: 57.89%, ROC-AUC: 0.8082.',
      file: 'phase2a_report.txt'
    },
    {
      id: 'phase2b',
      title: 'Phase 2B — DBSCAN Transductive Density Clustering',
      desc: 'Clustered 50,000 representative flows. Discovered 56 clusters with eps=2.552, min_samples=10. Evaluated attack density in clusters vs noise.',
      file: 'phase2b_report.txt'
    },
    {
      id: 'phase2c',
      title: 'Phase 2C — Destination Port Ablation Experiment',
      desc: 'Tested sensitivity to Destination Port. IF recall dropped slightly from 57.89% to 54.78%, confirming models learn true packet dynamics.',
      file: 'phase2c_report.txt'
    },
    {
      id: 'phase3',
      title: 'Phase 3 — Final Research Synthesis & Comparison',
      desc: 'Comprehensive comparative analysis between out-of-sample decision boundary scoring and transductive density clustering in network security.',
      file: 'phase3_final_report.txt'
    }
  ];

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-white tracking-tight">Academic Research Experiments</h1>
        <p className="text-xs text-slate-400">
          Complete historical documentation, experimental phase manifests, and scientific investigation logs
        </p>
      </div>

      {/* Phase Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {phases.map((ph) => (
          <div
            key={ph.id}
            onClick={() => setActiveReportKey(ph.file)}
            className={`cursor-pointer p-5 rounded-xl border transition-all flex flex-col justify-between space-y-3 ${
              activeReportKey === ph.file
                ? 'bg-cyan-950/20 border-cyan-500 text-slate-100 shadow-md shadow-cyan-500/5'
                : 'bg-slate-900/40 border-slate-800 text-slate-300 hover:border-slate-700'
            }`}
          >
            <div className="space-y-1.5">
              <span className="text-[10px] font-mono text-cyan-400 px-1.5 py-0.5 rounded bg-cyan-500/10 border border-cyan-500/20">
                {ph.file}
              </span>
              <h3 className="text-xs font-bold text-slate-100 mt-2">{ph.title}</h3>
              <p className="text-[11px] text-slate-400 leading-relaxed">{ph.desc}</p>
            </div>

            <div className="pt-2 text-[10px] font-mono text-cyan-400 flex items-center space-x-1">
              <span>View Scientific Log</span>
              <span>→</span>
            </div>
          </div>
        ))}
      </div>

      {/* Report Document Viewer */}
      <div className="p-6 rounded-xl bg-slate-900/40 border border-slate-800 space-y-4">
        <div className="flex items-center justify-between border-b border-slate-800 pb-3">
          <div className="flex items-center space-x-2">
            <FileText className="w-4 h-4 text-cyan-400" />
            <h3 className="font-semibold text-slate-200 text-xs font-mono">
              results/{activeReportKey}
            </h3>
          </div>
          <span className="text-[11px] text-slate-500 font-mono">Frozen Research Artifact</span>
        </div>

        <div className="bg-slate-950 p-4 rounded-lg border border-slate-800 max-h-[500px] overflow-y-auto">
          <pre className="text-xs font-mono text-slate-300 whitespace-pre-wrap leading-relaxed">
            {reports[activeReportKey] || 'Loading report content...'}
          </pre>
        </div>
      </div>
    </div>
  );
};
