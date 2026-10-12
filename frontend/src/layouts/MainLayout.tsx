import React, { useState } from 'react';
import { NavLink, Outlet } from 'react-router-dom';
import {
  ShieldAlert,
  LayoutDashboard,
  UploadCloud,
  Cpu,
  Activity,
  TableProperties,
  BarChart3,
  LineChart,
  BookOpen,
  Boxes,
  FileSpreadsheet,
  Info,
  Menu,
  X,
  Radio,
  ExternalLink
} from 'lucide-react';

const navItems = [
  { path: '/', label: 'Home Overview', icon: ShieldAlert },
  { path: '/dashboard', label: 'Main Dashboard', icon: LayoutDashboard },
  { path: '/upload', label: 'Dataset Upload', icon: UploadCloud },
  { path: '/preprocessing', label: 'Preprocessing Flow', icon: Cpu },
  { path: '/detect', label: 'Anomaly Detection', icon: Activity },
  { path: '/results', label: 'Analysis Results', icon: TableProperties },
  { path: '/performance', label: 'Model Benchmarks', icon: BarChart3 },
  { path: '/analytics', label: 'Visual Analytics', icon: LineChart },
  { path: '/research', label: 'Research Experiments', icon: BookOpen },
  { path: '/models', label: 'Model Inventory', icon: Boxes },
  { path: '/reports', label: 'Reports & Export', icon: FileSpreadsheet },
  { path: '/about', label: 'Academic & System Info', icon: Info },
];

export const MainLayout: React.FC = () => {
  const [sidebarOpen, setSidebarOpen] = useState(false);

  return (
    <div className="flex h-screen bg-slate-950 text-slate-100 overflow-hidden font-sans">
      {/* Sidebar */}
      <aside
        className={`fixed inset-y-0 left-0 z-50 w-72 bg-slate-900/95 backdrop-blur-md border-r border-slate-800/80 flex flex-col transition-transform duration-300 ease-in-out lg:translate-x-0 lg:static ${
          sidebarOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        {/* Brand / Logo */}
        <div className="p-5 border-b border-slate-800/80 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-cyan-500/10 border border-cyan-500/30 rounded-lg text-cyan-400">
              <ShieldAlert className="w-6 h-6 animate-pulse" />
            </div>
            <div>
              <div className="font-bold text-base tracking-wide bg-gradient-to-r from-cyan-400 to-blue-400 bg-clip-text text-transparent">
                CIC-IDS2017 NIDS
              </div>
              <div className="text-xs text-slate-400 font-mono">KLH • Team 21</div>
            </div>
          </div>
          <button
            onClick={() => setSidebarOpen(false)}
            className="p-1 rounded-lg hover:bg-slate-800 lg:hidden text-slate-400"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Academic Tag */}
        <div className="px-5 py-2.5 bg-slate-950/60 border-b border-slate-800/50 flex items-center justify-between text-xs">
          <span className="text-slate-400">Guide: Mr. Shaik Asif</span>
          <span className="px-2 py-0.5 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 font-medium">
            Active
          </span>
        </div>

        {/* Navigation Links */}
        <nav className="flex-1 overflow-y-auto px-3 py-4 space-y-1">
          {navItems.map((item) => {
            const Icon = item.icon;
            return (
              <NavLink
                key={item.path}
                to={item.path}
                onClick={() => setSidebarOpen(false)}
                className={({ isActive }) =>
                  `flex items-center space-x-3 px-3.5 py-2.5 rounded-lg text-sm font-medium transition-all ${
                    isActive
                      ? 'bg-gradient-to-r from-cyan-500/15 to-blue-500/15 border border-cyan-500/30 text-cyan-300 shadow-sm shadow-cyan-500/5'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
                  }`
                }
              >
                <Icon className="w-4 h-4 shrink-0" />
                <span>{item.label}</span>
              </NavLink>
            );
          })}
        </nav>

        {/* System Status Footer */}
        <div className="p-4 border-t border-slate-800/80 bg-slate-950/40 text-xs">
          <div className="flex items-center space-x-2 text-slate-400 mb-2">
            <Radio className="w-3.5 h-3.5 text-cyan-400 animate-ping" />
            <span className="font-mono text-[11px] text-cyan-300">SYSTEM ENGINE READY</span>
          </div>
          <div className="text-[11px] text-slate-500">
            Isolation Forest (Frozen) + DBSCAN (Transductive)
          </div>
        </div>
      </aside>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        {/* Top Header */}
        <header className="h-16 bg-slate-900/80 backdrop-blur-md border-b border-slate-800/80 flex items-center justify-between px-6 z-10 shrink-0">
          <div className="flex items-center space-x-4">
            <button
              onClick={() => setSidebarOpen(true)}
              className="p-2 rounded-lg hover:bg-slate-800 lg:hidden text-slate-300"
            >
              <Menu className="w-6 h-6" />
            </button>
            <div className="hidden sm:block">
              <span className="text-xs uppercase tracking-widest text-slate-500 font-mono">
                Anomaly-Based Network Intrusion Detection Platform
              </span>
            </div>
          </div>

          <div className="flex items-center space-x-4 text-xs font-mono">
            <span className="px-2.5 py-1 rounded bg-slate-800 border border-slate-700 text-slate-300">
              CIC-IDS2017 Benchmark
            </span>
            <span className="px-2.5 py-1 rounded bg-blue-500/10 border border-blue-500/30 text-blue-400">
              Dept. of CSE • KLH
            </span>
          </div>
        </header>

        {/* Scrollable Page Outlet */}
        <main className="flex-1 overflow-y-auto p-6 md:p-8 bg-slate-950">
          <div className="max-w-7xl mx-auto">
            <Outlet />
          </div>
        </main>
      </div>
    </div>
  );
};
