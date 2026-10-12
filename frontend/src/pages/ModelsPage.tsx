import React, { useEffect, useState } from 'react';
import {
  Boxes,
  Cpu,
  CheckCircle2,
  FileCode,
  ShieldCheck,
  Layers,
  ArrowRight
} from 'lucide-react';
import { Link } from 'react-router-dom';
import { api } from '../services/api';
import { ModelInfo } from '../types';

export const ModelsPage: React.FC = () => {
  const [models, setModels] = useState<ModelInfo[]>([]);

  useEffect(() => {
    api.getModels().then(setModels).catch(console.error);
  }, []);

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-white tracking-tight">Saved Models & Preprocessor Inventory</h1>
        <p className="text-xs text-slate-400">
          Inspection of frozen serialized joblib artifacts, expected tensor shapes, and preprocessors
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {models.map((m) => (
          <div
            key={m.id}
            className="p-6 rounded-xl bg-slate-900/40 border border-slate-800 space-y-4 hover:border-slate-700 transition-all flex flex-col justify-between"
          >
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="px-2 py-0.5 rounded bg-cyan-500/10 border border-cyan-500/20 text-cyan-400 font-mono text-xs">
                  {m.variant}
                </span>
                <span className="flex items-center space-x-1 text-emerald-400 text-xs font-mono">
                  <CheckCircle2 className="w-3.5 h-3.5" />
                  <span>{m.status.toUpperCase()}</span>
                </span>
              </div>

              <h3 className="font-bold text-slate-100 text-sm">{m.name}</h3>
              <p className="text-xs text-slate-400 leading-relaxed">{m.description}</p>
            </div>

            <div className="space-y-2 pt-3 border-t border-slate-800 text-xs font-mono">
              <div className="flex justify-between text-slate-400">
                <span>Features Expected:</span>
                <span className="text-slate-200">{m.feature_count} numeric cols</span>
              </div>
              <div className="flex justify-between text-slate-400">
                <span>Destination Port:</span>
                <span className="text-slate-200">{m.includes_destination_port ? 'Included' : 'Ablated'}</span>
              </div>
              <div className="flex justify-between text-slate-400">
                <span>Supported Tasks:</span>
                <span className="text-cyan-400">{m.supported_tasks.join(', ')}</span>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Preprocessing Artifacts Table */}
      <div className="p-6 rounded-xl bg-slate-900/40 border border-slate-800 space-y-4">
        <h3 className="font-semibold text-slate-200 text-sm flex items-center space-x-2">
          <FileCode className="w-4 h-4 text-cyan-400" />
          <span>Serialized Preprocessor Artifacts (models/)</span>
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs font-mono">
          <div className="p-4 rounded-lg bg-slate-950/60 border border-slate-800/80 space-y-2">
            <div className="font-bold text-cyan-400">models/imputer.pkl</div>
            <div className="text-slate-400 text-[11px]">
              SimpleImputer(strategy='median') fitted on 1,467,538 benign training flows (70 features).
            </div>
          </div>

          <div className="p-4 rounded-lg bg-slate-950/60 border border-slate-800/80 space-y-2">
            <div className="font-bold text-cyan-400">models/scaler.pkl</div>
            <div className="text-slate-400 text-[11px]">
              StandardScaler() fitted on benign training flows. Reuses training mean and variance vectors.
            </div>
          </div>

          <div className="p-4 rounded-lg bg-slate-950/60 border border-slate-800/80 space-y-2">
            <div className="font-bold text-cyan-400">models/imputer_no_port.pkl</div>
            <div className="text-slate-400 text-[11px]">
              Median imputer for the 69-feature ablated baseline (Destination Port removed).
            </div>
          </div>

          <div className="p-4 rounded-lg bg-slate-950/60 border border-slate-800/80 space-y-2">
            <div className="font-bold text-cyan-400">models/scaler_no_port.pkl</div>
            <div className="text-slate-400 text-[11px]">
              StandardScaler for the 69-feature ablated baseline.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
