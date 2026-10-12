import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { MainLayout } from './layouts/MainLayout';
import { HomePage } from './pages/HomePage';
import { DashboardPage } from './pages/DashboardPage';
import { UploadPage } from './pages/UploadPage';
import { PreprocessingPage } from './pages/PreprocessingPage';
import { DetectionPage } from './pages/DetectionPage';
import { ResultsPage } from './pages/ResultsPage';
import { PerformancePage } from './pages/PerformancePage';
import { AnalyticsPage } from './pages/AnalyticsPage';
import { ResearchPage } from './pages/ResearchPage';
import { ModelsPage } from './pages/ModelsPage';
import { ReportsPage } from './pages/ReportsPage';
import { AboutPage } from './pages/AboutPage';

export function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<MainLayout />}>
          <Route index element={<HomePage />} />
          <Route path="dashboard" element={<DashboardPage />} />
          <Route path="upload" element={<UploadPage />} />
          <Route path="preprocessing" element={<PreprocessingPage />} />
          <Route path="detect" element={<DetectionPage />} />
          <Route path="results" element={<ResultsPage />} />
          <Route path="performance" element={<PerformancePage />} />
          <Route path="analytics" element={<AnalyticsPage />} />
          <Route path="research" element={<ResearchPage />} />
          <Route path="models" element={<ModelsPage />} />
          <Route path="reports" element={<ReportsPage />} />
          <Route path="about" element={<AboutPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
