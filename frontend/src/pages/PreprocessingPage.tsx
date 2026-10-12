import React from 'react';
import {
  FileText,
  Filter,
  Sliders,
  Database,
  ArrowRight,
  ShieldCheck,
  AlertTriangle,
  Cpu
} from 'lucide-react';
import { Link } from 'react-router-dom';

export const PreprocessingPage: React.FC = () => {
  const steps = [
    {
      step: 1,
      title: 'Data Ingestion & Cleaning',
      subtitle: 'Raw Flow Aggregation',
      description: 'Ingests 8 CIC-IDS2017 CSV captures. Strips whitespace from column headers and standardizes multi-class labels into clean hyphenated strings.',
      stat: '2,830,743 raw flows'
    },
    {
      step: 2,
      title: 'Exact Deduplication',
      subtitle: 'Packet Capture Artifact Removal',
      description: 'Removes duplicate flow rows caused by multi-sensor monitoring overlaps, reducing bias toward identical recurrent packets.',
      stat: '2,522,362 unique flows kept'
    },
    {
      step: 3,
      title: 'Zero-Variance Feature Filtering',
      subtitle: '8 Constant Features Dropped',
      description: 'Eliminates 8 features with 0 variance across all 2.52M flows (e.g. Bwd PSH/URG Flags, Bulk rates). Destination Port is explicitly retained for Phase 2A.',
      stat: '70 numeric features remaining'
    },
    {
      step: 4,
      title: 'Non-Finite Conversion & Imputation',
      subtitle: 'Inf to NaN & Median Imputer',
      description: 'Replaces inf/-inf network rates with NaN. Fits SimpleImputer(strategy="median") exclusively on BENIGN training flows. Inference uses frozen medians.',
      stat: 'Zero data leakage'
    },
    {
      step: 5,
      title: 'Standard Scaling',
      subtitle: 'Z-Score Normalization',
      description: 'Fits StandardScaler on BENIGN training flows. Unseen test flows and user uploads are transformed using saved mean and variance vectors.',
      stat: 'X ~ N(0, 1) float32'
    },
    {
      step: 6,
      title: 'Model Input Tensor Preparation',
      subtitle: 'Strict Feature Ordering',
      description: 'Aligns 70 features in exact training order matching joblib preprocessors. Feeds into Isolation Forest or DBSCAN density clusterer.',
      stat: '70-dimensional vector'
    }
  ];

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-white tracking-tight">Preprocessing Pipeline Architecture</h1>
        <p className="text-xs text-slate-400">
          Scientific end-to-end transformation workflow from raw network PCAP flows to standardized model tensors
        </p>
      </div>

      {/* Critical Scientific Note Banner */}
      <div className="p-4 rounded-xl bg-cyan-950/20 border border-cyan-800/40 flex items-start space-x-3 text-xs">
        <ShieldCheck className="w-5 h-5 text-cyan-400 shrink-0 mt-0.5" />
        <div className="space-y-1">
          <div className="font-semibold text-cyan-300">Scientific Integrity Rule (Zero Leakage)</div>
          <p className="text-slate-300 leading-relaxed">
            For online inference, the system uses the <strong>exact fitted preprocessing artifacts</strong> (<code className="text-cyan-400">imputer.pkl</code> and <code className="text-cyan-400">scaler.pkl</code>) fitted on the 1,467,538 benign training flows. The imputer and scaler are <strong>never refit</strong> on newly uploaded inference data, ensuring complete independence and real-world out-of-sample validity.
          </p>
        </div>
      </div>

      {/* Workflow Flow Diagram */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {steps.map((st) => (
          <div
            key={st.step}
            className="p-5 rounded-xl bg-slate-900/40 border border-slate-800 hover:border-slate-700 transition-all flex flex-col justify-between space-y-4"
          >
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="w-7 h-7 rounded-full bg-cyan-500/10 border border-cyan-500/20 text-cyan-400 font-mono text-xs flex items-center justify-center font-bold">
                  {st.step}
                </span>
                <span className="text-[11px] font-mono text-slate-500">{st.subtitle}</span>
              </div>
              <h3 className="text-sm font-bold text-slate-100">{st.title}</h3>
              <p className="text-xs text-slate-400 leading-relaxed">{st.description}</p>
            </div>

            <div className="pt-3 border-t border-slate-800/60 flex items-center justify-between text-xs font-mono">
              <span className="text-slate-500 text-[10px]">Output State</span>
              <span className="text-cyan-400 font-medium">{st.stat}</span>
            </div>
          </div>
        ))}
      </div>

      {/* Zero Variance Dropped Table */}
      <div className="p-6 rounded-xl bg-slate-900/40 border border-slate-800 space-y-4">
        <h3 className="font-semibold text-slate-200 text-sm flex items-center space-x-2">
          <Filter className="w-4 h-4 text-cyan-400" />
          <span>Excluded Zero-Variance Features (8 Dropped)</span>
        </h3>
        <p className="text-xs text-slate-400">
          These 8 columns have zero standard deviation across all flows in the dataset and are systematically dropped:
        </p>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs font-mono">
          {[
            'Bwd PSH Flags',
            'Bwd URG Flags',
            'Fwd Avg Bytes/Bulk',
            'Fwd Avg Packets/Bulk',
            'Fwd Avg Bulk Rate',
            'Bwd Avg Bytes/Bulk',
            'Bwd Avg Packets/Bulk',
            'Bwd Avg Bulk Rate'
          ].map((col) => (
            <div key={col} className="p-2.5 rounded bg-slate-950/60 border border-slate-800/60 text-slate-300 text-center">
              {col}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
