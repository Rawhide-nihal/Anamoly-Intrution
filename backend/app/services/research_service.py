import json
import glob
from pathlib import Path
from typing import Dict, Any, List
from ..config import RESULTS_DIR, PLOTS_DIR, ROOT_DIR

class ResearchService:
    def __init__(self):
        pass

    def check_cicids_files_status(self) -> Dict[str, Any]:
        expected_files = [
            "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
            "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
            "Friday-WorkingHours-Morning.pcap_ISCX.csv",
            "Monday-WorkingHours.pcap_ISCX.csv",
            "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
            "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
            "Tuesday-WorkingHours.pcap_ISCX.csv",
            "Wednesday-workingHours.pcap_ISCX.csv"
        ]
        found = []
        missing = []
        for f in expected_files:
            if (ROOT_DIR / f).exists():
                found.append(f)
            else:
                missing.append(f)

        return {
            "all_present": len(missing) == 0,
            "found_count": len(found),
            "total_expected": len(expected_files),
            "found_files": found,
            "missing_files": missing,
            "message": "Full CIC-IDS2017 dataset (840 MB) present in root." if len(missing) == 0 else "Original raw CIC-IDS2017 CSV files are not in root. Full retraining / live Phase 1 deduplication reproduction is disabled, but saved model inference and frozen research benchmarks are fully operational."
        }

    def get_phase3_comparison(self) -> Dict[str, Any]:
        p = RESULTS_DIR / "phase3_comparison.json"
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def get_phase2a_metrics(self) -> Dict[str, Any]:
        p = RESULTS_DIR / "phase2a_metrics.json"
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def get_phase2b_metrics(self) -> Dict[str, Any]:
        p = RESULTS_DIR / "phase2b_metrics.json"
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def get_phase2c_metrics(self) -> Dict[str, Any]:
        p = RESULTS_DIR / "phase2c_metrics.json"
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def get_reports(self) -> Dict[str, str]:
        reports = {}
        for fname in ["phase2a_report.txt", "phase2b_report.txt", "phase2c_report.txt", "phase3_final_report.txt", "zero_variance_reconciliation.txt"]:
            p = RESULTS_DIR / fname
            if p.exists():
                with open(p, "r", encoding="utf-8", errors="replace") as f:
                    reports[fname] = f.read()
        return reports

    def get_available_plots(self) -> List[Dict[str, str]]:
        plots = []
        for phase in ["phase2a", "phase2b", "phase2c", "phase3"]:
            phase_dir = PLOTS_DIR / phase
            if phase_dir.exists():
                for p in sorted(phase_dir.glob("*.png")):
                    plots.append({
                        "phase": phase,
                        "filename": p.name,
                        "path": f"/api/research/plots/{phase}/{p.name}",
                        "title": p.stem.replace("_", " ").title()
                    })
        return plots

research_service = ResearchService()
