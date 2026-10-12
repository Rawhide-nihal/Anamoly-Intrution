import React, { useEffect, useState } from 'react';
import {
  LineChart,
  BarChart2,
  Image as ImageIcon,
  Layers,
  Filter,
  Maximize2
} from 'lucide-react';
import { api } from '../services/api';

export const AnalyticsPage: React.FC = () => {
  const [plots, setPlots] = useState<Array<{ phase: string; filename: string; path: string; title: string }>>([]);
  const [selectedPhase, setSelectedPhase] = useState<string>('all');
  const [activeModalImg, setActiveModalImg] = useState<string | null>(null);

  useEffect(() => {
    api.getResearchPlots().then(setPlots).catch(console.error);
  }, []);

  const filteredPlots = selectedPhase === 'all'
    ? plots
    : plots.filter((p) => p.phase === selectedPhase);

  return (
    <div className="space-y-8">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white tracking-tight">Interactive Visual Analytics & Plots</h1>
          <p className="text-xs text-slate-400">
            High-resolution empirical figures generated during research phases 2A, 2B, 2C, and 3
          </p>
        </div>

        {/* Phase Filter Tabs */}
        <div className="flex items-center space-x-1.5 overflow-x-auto pb-1">
          {[
            { id: 'all', label: 'All Figures' },
            { id: 'phase2a', label: 'Phase 2A (iForest)' },
            { id: 'phase2b', label: 'Phase 2B (DBSCAN)' },
            { id: 'phase2c', label: 'Phase 2C (Port Ablation)' },
            { id: 'phase3', label: 'Phase 3 (Comparison)' }
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setSelectedPhase(tab.id)}
              className={`px-3 py-1.5 rounded-lg text-xs font-mono transition-all shrink-0 ${
                selectedPhase === tab.id
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                  : 'bg-slate-900 text-slate-400 hover:text-slate-200 border border-slate-800'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {/* Plots Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {filteredPlots.map((plot, i) => (
          <div
            key={i}
            className="group rounded-xl bg-slate-900/40 border border-slate-800 overflow-hidden hover:border-cyan-500/40 transition-all flex flex-col justify-between"
          >
            <div className="p-4 border-b border-slate-800/80 flex items-center justify-between">
              <div>
                <span className="text-[10px] font-mono uppercase text-cyan-400 px-1.5 py-0.5 rounded bg-cyan-500/10 border border-cyan-500/20">
                  {plot.phase}
                </span>
                <h3 className="font-semibold text-slate-200 text-xs mt-1.5 truncate max-w-[220px]">
                  {plot.title}
                </h3>
              </div>
              <button
                onClick={() => setActiveModalImg(plot.path)}
                className="p-1.5 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-cyan-400 transition-all"
                title="Expand Image"
              >
                <Maximize2 className="w-4 h-4" />
              </button>
            </div>

            <div
              onClick={() => setActiveModalImg(plot.path)}
              className="cursor-pointer bg-slate-950 flex items-center justify-center p-3 h-56 overflow-hidden"
            >
              <img
                src={plot.path}
                alt={plot.title}
                className="max-h-full object-contain rounded group-hover:scale-105 transition-transform duration-300"
                loading="lazy"
              />
            </div>

            <div className="p-3 bg-slate-950/60 border-t border-slate-800/60 text-[11px] font-mono text-slate-500 flex justify-between">
              <span>{plot.filename}</span>
              <span className="text-cyan-400/80">Verified Artifact</span>
            </div>
          </div>
        ))}
      </div>

      {/* Modal for full image preview */}
      {activeModalImg && (
        <div
          onClick={() => setActiveModalImg(null)}
          className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4 cursor-pointer"
        >
          <div className="max-w-4xl max-h-[90vh] bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden p-2 shadow-2xl">
            <img
              src={activeModalImg}
              alt="Research Visualization"
              className="max-h-[85vh] w-auto mx-auto object-contain rounded-lg"
            />
          </div>
        </div>
      )}
    </div>
  );
};
