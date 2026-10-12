import React from 'react';
import {
  Award,
  BookOpen,
  ShieldAlert,
  Cpu,
  Layers,
  CheckCircle2,
  AlertTriangle,
  Globe
} from 'lucide-react';

export const AboutPage: React.FC = () => {
  return (
    <div className="space-y-8 max-w-4xl mx-auto">
      <div>
        <h1 className="text-2xl font-bold text-white tracking-tight">Academic Project & System Specifications</h1>
        <p className="text-xs text-slate-400">
          Formal institutional overview, research objectives, methodology, and limitations
        </p>
      </div>

      {/* Institutional Details */}
      <div className="p-6 rounded-2xl bg-slate-900/40 border border-slate-800 space-y-4">
        <h2 className="text-base font-bold text-slate-100 flex items-center space-x-2">
          <Award className="w-5 h-5 text-cyan-400" />
          <span>Institutional Information</span>
        </h2>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs">
          <div className="p-4 rounded-lg bg-slate-950/60 border border-slate-800/80 space-y-1">
            <span className="text-slate-500 font-mono text-[10px]">INSTITUTION</span>
            <div className="font-semibold text-slate-200 text-sm">KLH University</div>
            <div className="text-slate-400">KLH Bachupally Campus, Hyderabad</div>
          </div>

          <div className="p-4 rounded-lg bg-slate-950/60 border border-slate-800/80 space-y-1">
            <span className="text-slate-500 font-mono text-[10px]">DEPARTMENT & TEAM</span>
            <div className="font-semibold text-slate-200 text-sm">Computer Science and Engineering</div>
            <div className="text-cyan-400 font-medium">Team Number: 21</div>
          </div>

          <div className="p-4 rounded-lg bg-slate-950/60 border border-slate-800/80 space-y-1">
            <span className="text-slate-500 font-mono text-[10px]">PROJECT GUIDE</span>
            <div className="font-semibold text-slate-200 text-sm">Mr. Shaik Asif</div>
            <div className="text-slate-400">Faculty Guide, Department of CSE</div>
          </div>

          <div className="p-4 rounded-lg bg-slate-950/60 border border-slate-800/80 space-y-1">
            <span className="text-slate-500 font-mono text-[10px]">PROJECT TITLE</span>
            <div className="font-semibold text-slate-200 text-xs">
              Anomaly-Based Network Intrusion Detection System Using Isolation Forest and DBSCAN
            </div>
          </div>
        </div>
      </div>

      {/* Problem Statement & Objectives */}
      <div className="p-6 rounded-2xl bg-slate-900/40 border border-slate-800 space-y-4 text-xs text-slate-300">
        <h2 className="text-base font-bold text-slate-100 flex items-center space-x-2">
          <BookOpen className="w-5 h-5 text-blue-400" />
          <span>Problem Statement & Objectives</span>
        </h2>

        <p className="leading-relaxed">
          Modern computer networks generate massive volumes of continuous flow telemetry. Traditional signature-based intrusion detection systems (IDS) fail against zero-day exploits and polymorphic variations. Unsupervised anomaly detection methods offer the ability to identify anomalous patterns without labeled attack signatures.
        </p>

        <p className="leading-relaxed">
          This project conducts a rigorous empirical investigation comparing two fundamentally distinct unsupervised paradigms on the CIC-IDS2017 benchmark dataset (2.52M deduplicated flows):
        </p>

        <ul className="list-disc pl-5 space-y-1.5 text-slate-400">
          <li>
            <strong>Isolation Forest</strong>: An out-of-sample decision boundary approach trained solely on BENIGN flows to test whether abnormal attack profiles inherently require fewer random axis-aligned splits.
          </li>
          <li>
            <strong>DBSCAN</strong>: A transductive density clustering approach that groups continuous high-density areas and classifies unclustered points as noise.
          </li>
        </ul>
      </div>

      {/* Limitations & Future Enhancements */}
      <div className="p-6 rounded-2xl bg-slate-900/40 border border-slate-800 space-y-4 text-xs text-slate-300">
        <h2 className="text-base font-bold text-slate-100 flex items-center space-x-2">
          <AlertTriangle className="w-5 h-5 text-amber-400" />
          <span>Scientific Scope & Known Limitations</span>
        </h2>

        <div className="space-y-2 text-slate-400 leading-relaxed">
          <div>
            1. <strong>Testbed Evaluation</strong>: The research is conducted on the synthetic CIC-IDS2017 evaluation testbed; live packet ingestion or live enterprise networks were not part of the initial paper scope.
          </div>
          <div>
            2. <strong>DBSCAN Memory Complexity</strong>: Full transductive clustering across 100k+ flows demands quadratic distance storage; a representative 50k-flow seeded sample was used for primary evaluation.
          </div>
          <div>
            3. <strong>Dense Attack Patterns</strong>: High-volume flood attacks (e.g. DoS Hulk, DDoS) naturally form dense local clusters rather than sparse noise, which fundamentally restricts DBSCAN's recall compared to Isolation Forest.
          </div>
          <div>
            4. <strong>Binary Anomaly Classification</strong>: Both algorithms output binary anomaly classifications rather than granular multi-class threat categories.
          </div>
        </div>
      </div>
    </div>
  );
};
