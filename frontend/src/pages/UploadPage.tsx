import React, { useEffect, useState, useRef } from 'react';
import {
  UploadCloud,
  FileCheck,
  AlertCircle,
  CheckCircle2,
  Table,
  Cpu,
  Layers,
  ArrowRight,
  Database
} from 'lucide-react';
import { Link } from 'react-router-dom';
import { api } from '../services/api';
import { DatasetInfo, ValidationReport } from '../types';

export const UploadPage: React.FC = () => {
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState<string>('');
  const [validationReport, setValidationReport] = useState<ValidationReport | null>(null);
  const [previewData, setPreviewData] = useState<any | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isValidating, setIsValidating] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    loadDatasets();
  }, []);

  const loadDatasets = async () => {
    try {
      const list = await api.getDatasets();
      setDatasets(list);
      if (list.length > 0 && !selectedDatasetId) {
        setSelectedDatasetId(list[0].id);
        handleSelectDataset(list[0].id);
      }
    } catch (e) {
      console.error(e);
    }
  };

  const handleSelectDataset = async (id: string) => {
    setSelectedDatasetId(id);
    setIsValidating(true);
    setValidationReport(null);
    setPreviewData(null);
    try {
      const [report, preview] = await Promise.all([
        api.validateDataset(id),
        api.previewDataset(id, 20)
      ]);
      setValidationReport(report);
      setPreviewData(preview);
    } catch (e: any) {
      console.error(e);
    } finally {
      setIsValidating(false);
    }
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!file.name.endsWith('.csv')) {
      setUploadError('Only .csv network flow files are permitted.');
      return;
    }

    setIsUploading(true);
    setUploadError(null);

    try {
      const res = await api.uploadDataset(file);
      await loadDatasets();
      handleSelectDataset(res.dataset_id);
    } catch (err: any) {
      setUploadError(err.message || 'Upload failed');
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-white tracking-tight">Dataset Ingestion & Validation</h1>
        <p className="text-xs text-slate-400">
          Upload network traffic flow CSVs or select verified CIC-IDS2017 benchmark sample sets
        </p>
      </div>

      {/* Upload Dropzone */}
      <div className="p-8 rounded-2xl bg-slate-900/40 border-2 border-dashed border-slate-800 hover:border-cyan-500/50 transition-all flex flex-col items-center justify-center text-center space-y-4">
        <div className="p-4 rounded-full bg-cyan-500/10 border border-cyan-500/20 text-cyan-400">
          <UploadCloud className="w-8 h-8" />
        </div>

        <div>
          <h3 className="text-base font-semibold text-slate-100">Upload Network Flow CSV</h3>
          <p className="text-xs text-slate-400 max-w-md mt-1">
            Accepts CSV format with CIC-IDS2017 schema (up to 70 numeric flow attributes). Maximum upload size 200MB.
          </p>
        </div>

        <input
          type="file"
          accept=".csv"
          ref={fileInputRef}
          onChange={handleFileChange}
          className="hidden"
          id="csv-file-input"
        />

        <label
          htmlFor="csv-file-input"
          className={`cursor-pointer inline-flex items-center space-x-2 px-5 py-2.5 rounded-lg bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-semibold text-xs shadow-md transition-all ${
            isUploading ? 'opacity-50 pointer-events-none' : ''
          }`}
        >
          <span>{isUploading ? 'Uploading & Parsing...' : 'Browse Local CSV'}</span>
        </label>

        {uploadError && (
          <div className="text-xs text-rose-400 bg-rose-500/10 border border-rose-500/20 px-3 py-1.5 rounded-md">
            {uploadError}
          </div>
        )}
      </div>

      {/* Available Datasets Selector */}
      <div className="p-6 rounded-xl bg-slate-900/40 border border-slate-800 space-y-4">
        <h2 className="text-sm font-semibold text-slate-200 flex items-center space-x-2">
          <Database className="w-4 h-4 text-cyan-400" />
          <span>Available Datasets in Environment</span>
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {datasets.map((ds) => (
            <div
              key={ds.id}
              onClick={() => handleSelectDataset(ds.id)}
              className={`cursor-pointer p-4 rounded-lg border transition-all ${
                selectedDatasetId === ds.id
                  ? 'bg-cyan-950/20 border-cyan-500 text-slate-100 shadow-md shadow-cyan-500/5'
                  : 'bg-slate-900/60 border-slate-800/80 hover:border-slate-700 text-slate-300'
              }`}
            >
              <div className="flex items-center justify-between text-xs mb-2">
                <span className="font-mono text-cyan-400 truncate max-w-[180px] font-medium">{ds.filename}</span>
                {ds.is_sample && (
                  <span className="px-1.5 py-0.5 rounded bg-blue-500/10 border border-blue-500/20 text-blue-400 text-[10px]">
                    Verified Sample
                  </span>
                )}
              </div>

              <div className="text-[11px] text-slate-400 space-y-1">
                <div>Flows: <span className="font-mono text-slate-200">{ds.row_count.toLocaleString()}</span></div>
                <div>Columns: <span className="font-mono text-slate-200">{ds.column_count}</span></div>
                <div>Ground Truth: <span className="font-mono text-slate-200">{ds.has_labels ? 'Included (Label)' : 'None'}</span></div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Validation Report & Schema Inspection */}
      {isValidating && (
        <div className="p-8 text-center text-xs text-slate-400 bg-slate-900/30 rounded-xl border border-slate-800">
          Validating dataset schema against frozen models...
        </div>
      )}

      {validationReport && (
        <div className="space-y-6">
          <div className="p-6 rounded-xl bg-slate-900/40 border border-slate-800 space-y-6">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-800 pb-4">
              <div>
                <h3 className="font-semibold text-slate-100 text-sm flex items-center space-x-2">
                  <FileCheck className="w-4 h-4 text-emerald-400" />
                  <span>Schema Compatibility Report</span>
                </h3>
                <span className="text-xs text-slate-400">
                  Inspected for feature integrity, infinite values, NaN values, and model readiness
                </span>
              </div>

              <div>
                {validationReport.valid ? (
                  <span className="inline-flex items-center space-x-1.5 px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-medium">
                    <CheckCircle2 className="w-3.5 h-3.5" />
                    <span>Ready for Analysis</span>
                  </span>
                ) : (
                  <span className="inline-flex items-center space-x-1.5 px-3 py-1 rounded-full bg-rose-500/10 border border-rose-500/20 text-rose-400 text-xs font-medium">
                    <AlertCircle className="w-3.5 h-3.5" />
                    <span>Schema Incompatible</span>
                  </span>
                )}
              </div>
            </div>

            {/* Validation Stats Grid */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs">
              <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800/80">
                <div className="text-slate-400 text-[11px]">Matching Required Features</div>
                <div className="text-lg font-mono font-bold text-cyan-400">
                  {validationReport.matching_features.length} / 70
                </div>
              </div>

              <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800/80">
                <div className="text-slate-400 text-[11px]">Non-Finite / Inf Values</div>
                <div className="text-lg font-mono font-bold text-slate-200">
                  {validationReport.infinite_values_count}
                </div>
              </div>

              <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800/80">
                <div className="text-slate-400 text-[11px]">Duplicate Rows</div>
                <div className="text-lg font-mono font-bold text-slate-200">
                  {validationReport.duplicate_rows_count}
                </div>
              </div>

              <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800/80">
                <div className="text-slate-400 text-[11px]">Compatible Models</div>
                <div className="text-lg font-mono font-bold text-emerald-400">
                  {validationReport.compatible_models.length} Available
                </div>
              </div>
            </div>

            {/* Warnings or Errors */}
            {validationReport.warnings.length > 0 && (
              <div className="p-3.5 rounded-lg bg-amber-500/10 border border-amber-500/20 text-xs text-amber-300 space-y-1">
                <div className="font-semibold flex items-center gap-1.5">
                  <AlertCircle className="w-3.5 h-3.5" /> Pipeline Notices:
                </div>
                {validationReport.warnings.map((w, idx) => (
                  <div key={idx} className="text-[11px] text-amber-200/90 pl-5">
                    • {w}
                  </div>
                ))}
              </div>
            )}

            {validationReport.errors.length > 0 && (
              <div className="p-3.5 rounded-lg bg-rose-500/10 border border-rose-500/20 text-xs text-rose-300 space-y-1">
                <div className="font-semibold flex items-center gap-1.5">
                  <AlertCircle className="w-3.5 h-3.5" /> Blocking Errors:
                </div>
                {validationReport.errors.map((e, idx) => (
                  <div key={idx} className="text-[11px] text-rose-200/90 pl-5">
                    • {e}
                  </div>
                ))}
              </div>
            )}

            {/* Quick Action Button */}
            {validationReport.valid && (
              <div className="pt-2 flex justify-end">
                <Link
                  to={`/detect?dataset_id=${selectedDatasetId}`}
                  className="inline-flex items-center space-x-2 px-5 py-2.5 rounded-lg bg-cyan-500 hover:bg-cyan-400 text-slate-950 text-xs font-semibold shadow transition-all"
                >
                  <span>Proceed to Anomaly Detection</span>
                  <ArrowRight className="w-4 h-4" />
                </Link>
              </div>
            )}
          </div>

          {/* Dataset Preview Table */}
          {previewData && (
            <div className="p-6 rounded-xl bg-slate-900/40 border border-slate-800 space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="font-semibold text-slate-100 text-sm flex items-center space-x-2">
                  <Table className="w-4 h-4 text-blue-400" />
                  <span>Dataset Feature Preview (First {previewData.preview_count} Rows)</span>
                </h3>
                <span className="text-xs text-slate-500 font-mono">
                  {previewData.total_rows.toLocaleString()} total rows
                </span>
              </div>

              <div className="overflow-x-auto max-h-96 border border-slate-800 rounded-lg">
                <table className="w-full text-left text-[11px] border-collapse font-mono">
                  <thead className="sticky top-0 bg-slate-900 border-b border-slate-800 text-slate-400">
                    <tr>
                      <th className="py-2 px-3">#</th>
                      {previewData.columns.slice(0, 10).map((col: string) => (
                        <th key={col} className="py-2 px-3 whitespace-nowrap">
                          {col}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/40 bg-slate-950/40">
                    {previewData.rows.map((row: any, i: number) => (
                      <tr key={i} className="hover:bg-slate-800/30">
                        <td className="py-1.5 px-3 text-slate-500">{i + 1}</td>
                        {previewData.columns.slice(0, 10).map((col: string) => (
                          <td key={col} className="py-1.5 px-3 text-slate-300 whitespace-nowrap">
                            {row[col] !== null ? String(row[col]) : 'NaN'}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
