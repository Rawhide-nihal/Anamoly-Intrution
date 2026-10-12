import React, { useEffect, useState } from 'react';
import {
  BarChart3,
  Award,
  AlertTriangle,
  FileText,
  CheckCircle2,
  TrendingUp,
  Percent,
  Layers,
  ShieldCheck
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend
} from 'recharts';
import { api } from '../services/api';

export const PerformancePage: React.FC = () => {
  const [phase3Data, setPhase3Data] = useState<any | null>(null);

  useEffect(() => {
    api.getPhase3Comparison().then(setPhase3Data).catch(console.error);
  }, []);

  const benchmarkTable = [
    { metric: 'Precision', iForest: '0.8387 (83.87%)', dbscan: '0.1061 (10.61%)', desc: 'True positive rate among all flagged anomalies' },
    { metric: 'Recall', iForest: '0.5789 (57.89%)', dbscan: '0.0205 (2.05%)', desc: 'Coverage of actual attack flows flagged' },
    { metric: 'F1 Score', iForest: '0.6850 (68.50%)', dbscan: '0.0344 (3.44%)', desc: 'Harmonic mean of precision and recall' },
    { metric: 'False Positive Rate (FPR)', iForest: '0.0754 (7.54%)', dbscan: '0.0346 (3.46%)', desc: 'Benign flows incorrectly flagged as anomalies' },
    { metric: 'False Negative Rate (FNR)', iForest: '0.4211 (42.11%)', dbscan: '0.9795 (97.95%)', desc: 'Attack flows missed (categorized as normal)' },
    { metric: 'ROC-AUC', iForest: '0.8082', dbscan: 'Not Applicable', desc: 'Ranking quality across continuous anomaly scores' }
  ];

  const chartData = [
    { metric: 'Precision', iForest: 83.87, dbscan: 10.61 },
    { metric: 'Recall', iForest: 57.89, dbscan: 2.05 },
    { metric: 'F1 Score', iForest: 68.5, dbscan: 3.44 },
    { metric: 'FPR', iForest: 7.54, dbscan: 3.46 }
  ];

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-white tracking-tight">Model Performance & Evaluation</h1>
        <p className="text-xs text-slate-400">
          Historical benchmark metrics derived from frozen Phase 2A and Phase 2B experiment artifacts
        </p>
      </div>

      {/* Mandatory Methodological Disclaimer Alert */}
      <div className="p-6 rounded-xl bg-cyan-950/20 border border-cyan-800/50 space-y-2 text-xs">
        <div className="flex items-center space-x-2 text-cyan-400 font-bold text-sm">
          <AlertTriangle className="w-5 h-5 shrink-0" />
          <span>Mandatory Methodological Qualification</span>
        </div>
        <p className="text-slate-300 leading-relaxed italic pl-7">
          "Isolation Forest was evaluated on an unseen test population after being trained exclusively on BENIGN traffic, whereas DBSCAN was evaluated transductively on a representative mixed sample. Therefore, their raw metric values should be interpreted within their respective experimental designs rather than as a direct apples-to-apples benchmark."
        </p>
      </div>

      {/* Benchmark Metric Comparison Table */}
      <div className="p-6 rounded-xl bg-slate-900/40 border border-slate-800 space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="font-semibold text-slate-100 text-sm flex items-center space-x-2">
            <Award className="w-4 h-4 text-cyan-400" />
            <span>Frozen Historical Research Benchmark</span>
          </h3>
          <span className="text-xs text-slate-500 font-mono">results/phase3_comparison.json</span>
        </div>

        <div className="overflow-x-auto border border-slate-800 rounded-lg">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="bg-slate-900 border-b border-slate-800 text-slate-400 font-mono">
                <th className="py-3 px-4">Metric</th>
                <th className="py-3 px-4 text-cyan-300">Isolation Forest (Unseen Test)</th>
                <th className="py-3 px-4 text-blue-300">DBSCAN (50k Mixed Sample)</th>
                <th className="py-3 px-4 text-slate-400">Interpretation</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-sans">
              {benchmarkTable.map((row, i) => (
                <tr key={i} className="hover:bg-slate-800/30">
                  <td className="py-3 px-4 font-semibold text-slate-200">{row.metric}</td>
                  <td className="py-3 px-4 font-mono font-bold text-cyan-400">{row.iForest}</td>
                  <td className="py-3 px-4 font-mono font-bold text-blue-400">{row.dbscan}</td>
                  <td className="py-3 px-4 text-slate-400 text-[11px]">{row.desc}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Comparative Visualization */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="p-6 rounded-xl bg-slate-900/40 border border-slate-800 space-y-4">
          <h3 className="font-semibold text-slate-100 text-sm">Metric Comparison Overview (%)</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="metric" stroke="#64748b" fontSize={11} />
                <YAxis stroke="#64748b" fontSize={11} domain={[0, 100]} />
                <Tooltip
                  formatter={(val: any) => `${val}%`}
                  contentStyle={{ backgroundColor: '#090d16', borderColor: '#1e293b', borderRadius: '8px', fontSize: '12px' }}
                />
                <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }} />
                <Bar dataKey="iForest" name="Isolation Forest" fill="#06b6d4" radius={[4, 4, 0, 0]} />
                <Bar dataKey="dbscan" name="DBSCAN" fill="#3b82f6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Attack Density Explanation */}
        <div className="p-6 rounded-xl bg-slate-900/40 border border-slate-800 space-y-4 text-xs">
          <h3 className="font-semibold text-slate-100 text-sm flex items-center space-x-2">
            <Layers className="w-4 h-4 text-blue-400" />
            <span>Key Research Insight: Attack Density vs Sparsity</span>
          </h3>
          <p className="text-slate-300 leading-relaxed">
            In network intrusion detection, many cyber attacks (e.g. DoS Hulk, DDoS, PortScan) generate <strong>dense packet floods</strong> with highly correlated flow metrics rather than isolated outliers.
          </p>
          <div className="p-3.5 rounded-lg bg-slate-950/60 border border-slate-800 space-y-2 text-[11px] font-mono">
            <div>• In the 50,000-flow sample, <span className="text-blue-400 font-bold">41.0%</span> of attack flows formed attack-dominated dense clusters.</div>
            <div>• Another <span className="text-blue-400 font-bold">56.1%</span> grouped into mixed clusters with high local density.</div>
            <div>• Only <span className="text-rose-400 font-bold">2.05%</span> were isolated as noise points by DBSCAN.</div>
          </div>
          <p className="text-slate-400 leading-relaxed text-[11px]">
            In contrast, Isolation Forest trained solely on BENIGN flows isolates high-volume DoS packets easily, flagging 89.3% of DoS Hulk traffic despite their density.
          </p>
        </div>
      </div>
    </div>
  );
};
