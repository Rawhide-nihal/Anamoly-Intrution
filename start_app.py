import os
import sys
import time
import subprocess
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND_DIR = ROOT / "backend"
FRONTEND_DIR = ROOT / "frontend"

def main():
    print("=" * 75)
    print("   CIC-IDS2017 NETWORK INTRUSION DETECTION SYSTEM — WEB PLATFORM")
    print("   Anomaly Detection Using Isolation Forest & DBSCAN")
    print("   KLH University • Department of CSE • Team 21 • Guide: Mr. Shaik Asif")
    print("=" * 75)

    # 1. Start Backend FastAPI Server
    print("\n[1/3] Starting FastAPI Backend on http://localhost:8000 ...")
    backend_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd=str(BACKEND_DIR)
    )

    # 2. Start Frontend Vite Dev Server
    print("[2/3] Starting React Vite Frontend on http://localhost:5173 ...")
    frontend_cmd = "npm.cmd" if sys.platform == "win32" else "npm"
    frontend_proc = subprocess.Popen(
        [frontend_cmd, "run", "dev"],
        cwd=str(FRONTEND_DIR)
    )

    time.sleep(3)
    target_url = "http://localhost:5173"
    print(f"[3/3] Application online at: {target_url}")
    print("\nPress Ctrl+C to terminate both servers.")

    try:
        webbrowser.open(target_url)
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down servers...")
        backend_proc.terminate()
        frontend_proc.terminate()
        print("Application stopped.")

if __name__ == "__main__":
    main()
