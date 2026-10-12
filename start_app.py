import os
import sys
import time
import socket
import urllib.request
import subprocess
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND_DIR = ROOT / "backend"
FRONTEND_DIR = ROOT / "frontend"

def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0

def wait_for_service(url: str, timeout: int = 20) -> bool:
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with urllib.request.urlopen(url, timeout=1) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False

def main():
    print("=" * 78)
    print("   CIC-IDS2017 NETWORK INTRUSION DETECTION SYSTEM — WEB PLATFORM")
    print("   Anomaly-Based Traffic Analysis using Isolation Forest and DBSCAN")
    print("   KLH University • Department of CSE • Team 21 • Guide: Mr. Shaik Asif")
    print("=" * 78)

    # Pre-flight check: Port availability
    backend_port = 8000
    frontend_port = 5173

    if is_port_in_use(backend_port):
        print(f"[!] Warning: Port {backend_port} is already in use. Terminating previous backend instances or use different port.")

    # 1. Start Backend FastAPI Server
    print(f"\n[1/3] Starting FastAPI Backend on http://127.0.0.1:{backend_port} ...")
    backend_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(backend_port)],
        cwd=str(BACKEND_DIR)
    )

    backend_ok = wait_for_service(f"http://127.0.0.1:{backend_port}/api/health", timeout=15)
    if not backend_ok:
        print("[X] Error: Backend failed to start or respond to health check.")
        backend_proc.terminate()
        sys.exit(1)
    print("    -> Backend healthy & connected to SQLite session engine.")

    # 2. Start Frontend Vite Dev Server
    print(f"\n[2/3] Starting React Vite Frontend on http://localhost:{frontend_port} ...")
    frontend_cmd = "npm.cmd" if sys.platform == "win32" else "npm"
    frontend_proc = subprocess.Popen(
        [frontend_cmd, "run", "dev"],
        cwd=str(FRONTEND_DIR)
    )

    frontend_ok = wait_for_service(f"http://localhost:{frontend_port}", timeout=15)
    if not frontend_ok:
        print("[!] Frontend dev server still initializing or building modules...")
    else:
        print("    -> Frontend online & serving React TypeScript application.")

    target_url = f"http://localhost:{frontend_port}"
    print(f"\n[3/3] Application online at: {target_url}")
    print("      Swagger API Documentation: http://127.0.0.1:8000/docs")
    print("\nPress Ctrl+C to terminate both servers.")

    try:
        webbrowser.open(target_url)
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down servers cleanly...")
        backend_proc.terminate()
        frontend_proc.terminate()
        print("Application stopped.")

if __name__ == "__main__":
    main()
