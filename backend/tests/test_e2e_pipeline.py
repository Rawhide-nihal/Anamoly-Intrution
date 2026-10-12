import sys
import time
import json
import urllib.request
import subprocess

def test_pipeline():
    print("Launching FastAPI server...")
    p_back = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd="backend"
    )
    time.sleep(3)

    try:
        # Datasets
        dss = json.loads(urllib.request.urlopen("http://127.0.0.1:8000/api/datasets").read().decode())
        ds_id = dss[0]["id"]
        print(f"Selected dataset: {ds_id} ({dss[0]['filename']})")

        # Create Isolation Forest job
        req = urllib.request.Request(
            "http://127.0.0.1:8000/api/jobs",
            data=json.dumps({"dataset_id": ds_id, "model_id": "if_with_port", "algorithm": "isolation_forest"}).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        job_res = json.loads(urllib.request.urlopen(req).read().decode())
        job_id = job_res["job_id"]
        print(f"Job queued: {job_id}")

        # Poll status
        for _ in range(15):
            time.sleep(1)
            st = json.loads(urllib.request.urlopen(f"http://127.0.0.1:8000/api/jobs/{job_id}").read().decode())
            print(f"Polling Status: {st['status']} (progress: {st['progress']}%, stage: {st['current_stage']})")
            if st["status"] in ("completed", "failed"):
                break

        assert st["status"] == "completed", f"Job failed: {st.get('error_message')}"

        # Results summary
        results = json.loads(urllib.request.urlopen(f"http://127.0.0.1:8000/api/jobs/{job_id}/results").read().decode())
        print("Results Summary:", json.dumps({
            "total_rows": results["total_rows"],
            "normal_count": results["normal_count"],
            "anomaly_count": results["anomaly_count"],
            "metrics": results["metrics"]
        }, indent=2))

        # Predictions
        preds = json.loads(urllib.request.urlopen(f"http://127.0.0.1:8000/api/jobs/{job_id}/predictions?page=1&page_size=3").read().decode())
        print(f"Retrieved {len(preds['rows'])} prediction rows.")
        for r in preds['rows']:
            print(f"  Row {r['row_index']}: label={r['prediction_label']}, score={r['score']:.4f}")

        # CSV Export
        csv_data = urllib.request.urlopen(f"http://127.0.0.1:8000/api/jobs/{job_id}/export/csv").read().decode()
        print(f"CSV Export verified: {len(csv_data.splitlines())} lines downloaded.")

        print("\nALL END-TO-END PIPELINE CHECKS PASSED SUCCESSFULLY!")

    finally:
        p_back.terminate()

if __name__ == "__main__":
    test_pipeline()
