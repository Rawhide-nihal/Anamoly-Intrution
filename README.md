# Anomaly-Based Intrusion Detection System Using DBSCAN and Isolation Forest
## Complete Console-to-Web Transformation Platform

**Academic Project**: Anomaly-Based Network Intrusion Detection System Using Isolation Forest and DBSCAN  
**Institution**: KLH University, KLH Bachupally Campus, Hyderabad  
**Department**: Computer Science and Engineering  
**Team Number**: 21  
**Project Guide**: Mr. Shaik Asif  
**Repository**: [https://github.com/Samrat-Reddy/ML-Project](https://github.com/Samrat-Reddy/ML-Project)

---

## 1. Overview & Research Scope

Unsupervised network intrusion detection on the **CIC-IDS2017** benchmark dataset (2,522,362 deduplicated flows, 70 features, 14 attack categories), comparing two fundamental paradigms for defining an anomaly:

- **Isolation Forest**: Trained exclusively on BENIGN traffic (1,467,538 flows), scoring 1,054,824 unseen test flows out-of-sample via continuous path-length anomaly scoring (`-decision_function(X)`).
- **DBSCAN**: Transductively clusters a representative 50,000-flow sample using unsupervised k-distance elbow parameters (`eps=2.552, min_samples=10`). Points labeled noise (`-1`) are treated as anomalies.

### Empirical Research Benchmarks

| Metric | Isolation Forest (Unseen Test Partition) | DBSCAN (50,000 Mixed Sample) |
|---|---|---|
| Evaluation Population | 1,054,824 unseen flows (628,946 BENIGN + 425,878 ATTACK) | 50,000-flow seeded sample (41,665 BENIGN + 8,335 ATTACK) |
| Precision | **0.8387** (83.87%) | **0.1061** (10.61%) |
| Recall | **0.5789** (57.89%) | **0.0205** (2.05%) |
| F1 Score | **0.6850** (68.50%) | **0.0344** (3.44%) |
| False Positive Rate (FPR) | **0.0754** (7.54%) | **0.0346** (3.46%) |
| False Negative Rate (FNR) | **0.4211** (42.11%) | **0.9795** (97.95%) |
| ROC-AUC | **0.8082** | Not Applicable (binary noise output) |

> **Mandatory Methodological Qualification**:  
> Isolation Forest was evaluated on an unseen test population after being trained exclusively on BENIGN traffic, whereas DBSCAN was evaluated transductively on a representative mixed sample. Therefore, their raw metric values should be interpreted within their respective experimental designs rather than as a direct apples-to-apples benchmark.  
>  
> **Key Research Finding**: Many cyber attacks in CIC-IDS2017 (DoS Hulk, DDoS, PortScan) generate dense packet floods rather than sparse outliers. In DBSCAN, 41.0% of attack flows grouped into attack-dominated clusters and 56.1% into mixed clusters; only 2.05% formed isolated noise. Conversely, Isolation Forest trained solely on normal traffic isolates high-volume DoS traffic easily (89.3% of DoS Hulk flagged).

---

## 2. Web Application Architecture

The project has been transformed into a full-stack, responsive web application:

```text
ML-Project/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI application entrypoint with CORS
│   │   ├── config.py            # Path resolution and pipeline constants
│   │   ├── api/
│   │   │   └── endpoints.py     # REST API routes (Health, Datasets, Jobs, Models, Results, Research)
│   │   ├── ml/
│   │   │   └── service.py       # Reusable ML service (Isolation Forest & DBSCAN clustering)
│   │   ├── jobs/
│   │   │   └── manager.py       # Threaded background job manager with stage tracking
│   │   ├── services/
│   │   │   ├── dataset_service.py # CSV parsing, schema validation, and row preview
│   │   │   └── research_service.py # Frozen benchmarks, report viewer, and plots
│   │   ├── storage/
│   │   │   └── db.py            # SQLite database manager for metadata & job history
│   │   └── schemas/
│   │       └── base.py          # Strongly typed Pydantic models
│   └── tests/
│       ├── test_ml.py           # Unit tests for models, preprocessors, and conventions
│       └── test_e2e_pipeline.py # End-to-end integration test of API pipeline
├── frontend/
│   ├── src/
│   │   ├── layouts/
│   │   │   └── MainLayout.tsx   # Premium cybersecurity midnight-navy interface
│   │   ├── pages/
│   │   │   ├── HomePage.tsx     # Landing overview with Team 21 & KLH University credits
│   │   │   ├── DashboardPage.tsx # Traffic metrics, classification charts, and recent jobs
│   │   │   ├── UploadPage.tsx   # CSV upload, schema validation, and feature inspection
│   │   │   ├── PreprocessingPage.tsx # 6-stage scientific data cleaning pipeline
│   │   │   ├── DetectionPage.tsx # Model/algorithm selector and live job monitor
│   │   │   ├── ResultsPage.tsx  # Per-row prediction inspection, scores, and CSV export
│   │   │   ├── PerformancePage.tsx # Frozen benchmark metrics and mandatory disclaimer
│   │   │   ├── AnalyticsPage.tsx # Interactive high-resolution empirical plots gallery
│   │   │   ├── ResearchPage.tsx # Phases 1–3 experimental logs and documentation
│   │   │   ├── ModelsPage.tsx   # Saved models inventory and preprocessor inspection
│   │   │   ├── ReportsPage.tsx  # Printable formal evaluation report and export
│   │   │   └── AboutPage.tsx    # Academic specifications, guide, and limitations
│   │   ├── services/
│   │   │   └── api.ts           # Typed client calling backend REST endpoints
│   │   └── types/
│   │       └── index.ts         # TypeScript interface contracts
├── models/                      # Frozen Isolation Forest models, imputers, scalers
├── results/                     # Research manifests, JSON metrics, and final report
├── plots/                       # High-resolution empirical figures (Phases 2A, 2B, 2C, 3)
├── sample_data/                 # Verified sample traffic flows for local demonstrations
├── demo.py                      # Original CLI demo (preserved and fully functional)
└── start_app.py                 # Convenient Windows one-command launcher
```

---

## 3. Quick Start (Web Application)

### Option A: One-Command Windows Launcher
Run the single launcher script from the repository root:
```powershell
python start_app.py
```
This script launches the FastAPI backend on `http://127.0.0.1:8000`, the Vite frontend on `http://localhost:5173`, and opens the platform in your browser.

---

### Option B: Manual Service Startup

#### 1. Backend Service
```powershell
cd backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
Interactive API documentation will be accessible at: `http://127.0.0.1:8000/docs`

#### 2. Frontend Application
```powershell
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173` in your browser.

---

## 4. Running the Original CLI Demo

The original CLI demo remains intact and functional:
```powershell
python demo.py           # Full demo with hash verification and live evaluation
python demo.py --quick   # Fast replay using frozen cluster assignments
python demo.py --open    # Open final research figures on Windows
```

---

## 5. Dataset Setup Instructions

To perform full retraining or reproduce the raw Phase 1 deduplication of all 8 CSV files (840 MB), download the **MachineLearningCVE** CSV files of CIC-IDS2017 from the Canadian Institute for Cybersecurity ([UNB Dataset Portal](https://www.unb.ca/cic/datasets/ids-2017.html)) and place them in the repository root:
```text
Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
Friday-WorkingHours-Morning.pcap_ISCX.csv
Monday-WorkingHours.pcap_ISCX.csv
Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
Tuesday-WorkingHours.pcap_ISCX.csv
Wednesday-workingHours.pcap_ISCX.csv
```
*Note: If the 8 raw files are not present in the root, the web platform remains fully operational using the frozen models, serialized preprocessors, verified benchmark artifacts, and uploaded flow CSVs.*

---

## 6. Automated Testing & Verification

Run the test suite:
```powershell
# ML Service & Convention Tests
python backend/tests/test_ml.py

# End-to-End API Pipeline Test
python backend/tests/test_e2e_pipeline.py

# Frontend TypeScript & Vite Production Build
cd frontend
npm run build
```

---

## 7. Institutional Credits

- **Institution**: KLH University (KL Deemed to be University, Off-Campus), Bachupally Campus, Hyderabad
- **Department**: Department of Computer Science and Engineering
- **Team Number**: 21
- **Faculty Guide**: Mr. Shaik Asif
