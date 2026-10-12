import sys
import time
import json
import urllib.request
import subprocess

def test_pipeline():
    print("Launching FastAPI server on port 8009...")
    p_back = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8009"],
        cwd="backend"
    )
    time.sleep(3)

    try:
        # Datasets
        dss = json.loads(urllib.request.urlopen("http://127.0.0.1:8009/api/datasets").read().decode())
        ds_id = dss[0]["id"]
        print(f"Selected dataset: {ds_id} ({dss[0]['filename']})")

        # 1. Test Isolation Forest Job
        print("\n--- Testing Isolation Forest Pipeline ---")
        req_if = urllib.request.Request(
            "http://127.0.0.1:8009/api/jobs",
            data=json.dumps({"dataset_id": ds_id, "model_id": "if_with_port", "algorithm": "isolation_forest"}).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        job_res_if = json.loads(urllib.request.urlopen(req_if).read().decode())
        job_id_if = job_res_if["job_id"]
        print(f"IF Job queued: {job_id_if}")

        for _ in range(15):
            time.sleep(1)
            st = json.loads(urllib.request.urlopen(f"http://127.0.0.1:8009/api/jobs/{job_id_if}").read().decode())
            print(f"  Status: {st['status']} ({st['progress']}%, {st['current_stage']})")
            if st["status"] in ("completed", "failed"):
                break

        assert st["status"] == "completed", f"IF Job failed: {st.get('error_message')}"

        results_if = json.loads(urllib.request.urlopen(f"http://127.0.0.1:8009/api/jobs/{job_id_if}/results").read().decode())
        print(f"IF Complete: {results_if['total_rows']} rows, {results_if['anomaly_count']} anomalies")

        # 2. Test DBSCAN Job with k-distance elbow curve
        print("\n--- Testing DBSCAN Pipeline & k-distance ---")
        k_curve = json.loads(urllib.request.urlopen(f"http://127.0.0.1:8009/api/datasets/{ds_id}/k-distance?k=10").read().decode())
        print(f"k-distance suggested eps: {k_curve['suggested_eps']}")

        req_db = urllib.request.Request(
            "http://127.0.0.1:8009/api/jobs",
            data=json.dumps({
                "dataset_id": ds_id,
                "model_id": "dbscan_clustering",
                "algorithm": "dbscan",
                "dbscan_eps": k_curve['suggested_eps'],
                "dbscan_min_samples": 5,
                "sample_limit": 50000
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        job_res_db = json.loads(urllib.request.urlopen(req_db).read().decode())
        job_id_db = job_res_db["job_id"]
        print(f"DBSCAN Job queued: {job_id_db}")

        for _ in range(15):
            time.sleep(1)
            st_db = json.loads(urllib.request.urlopen(f"http://127.0.0.1:8009/api/jobs/{job_id_db}").read().decode())
            print(f"  Status: {st_db['status']} ({st_db['progress']}%, {st_db['current_stage']})")
            if st_db["status"] in ("completed", "failed"):
                break

        assert st_db["status"] == "completed", f"DBSCAN Job failed: {st_db.get('error_message')}"

        results_db = json.loads(urllib.request.urlopen(f"http://127.0.0.1:8009/api/jobs/{job_id_db}/results").read().decode())
        print(f"DBSCAN Complete: {results_db['total_rows']} rows, clusters={results_db['cluster_stats']['n_clusters']}, noise={results_db['cluster_stats']['noise_count']}")

        # Predictions inspection
        preds = json.loads(urllib.request.urlopen(f"http://127.0.0.1:8009/api/jobs/{job_id_db}/predictions?page=1&page_size=5").read().decode())
        print(f"Retrieved {len(preds['rows'])} prediction rows.")
        for r in preds['rows']:
            print(f"  Row {r['row_index']}: label={r['prediction_label']}, cluster={r['cluster_id']}, gt={r['ground_truth']}")

        # CSV Export
        csv_data = urllib.request.urlopen(f"http://127.0.0.1:8009/api/jobs/{job_id_db}/export/csv").read().decode()
        header_line = csv_data.splitlines()[0]
        print(f"DBSCAN CSV Export header: {header_line[:60]}... (Total {len(csv_data.splitlines())} lines)")
        assert "original_dataset_row_index" in header_line

        # 3. Test Model Training / Reproduction Pipeline
        print("\n--- Testing Model Training / Reproduction Pipeline ---")
        req_tr = urllib.request.Request(
            "http://127.0.0.1:8009/api/research/train",
            data=json.dumps({
                "dataset_id": ds_id,
                "variant": "with_port",
                "n_estimators": 50,
                "contamination": 0.05,
                "random_state": 42
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        job_res_tr = json.loads(urllib.request.urlopen(req_tr).read().decode())
        job_id_tr = job_res_tr["job_id"]
        print(f"Training Job queued: {job_id_tr}")

        for _ in range(15):
            time.sleep(1)
            st_tr = json.loads(urllib.request.urlopen(f"http://127.0.0.1:8009/api/research/train/jobs/{job_id_tr}").read().decode())
            print(f"  Status: {st_tr['status']} ({st_tr['progress']}%, {st_tr['current_stage']})")
            if st_tr["status"] in ("completed", "failed"):
                break

        assert st_tr["status"] == "completed", f"Training Job failed: {st_tr.get('error_message')}"
        print(f"Training Complete! Trained rows: {st_tr['training_rows']}, Features: {st_tr['feature_count']}, Saved to: {st_tr['model_path']}")

        # 4. Test Job Cancellation (Resource Control)
        print("\n--- Testing Job Cancellation & Resource Control ---")
        req_cancel_test = urllib.request.Request(
            "http://127.0.0.1:8009/api/jobs",
            data=json.dumps({"dataset_id": ds_id, "model_id": "if_with_port", "algorithm": "isolation_forest"}).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        cancel_job_res = json.loads(urllib.request.urlopen(req_cancel_test).read().decode())
        c_job_id = cancel_job_res["job_id"]
        # Immediately cancel
        req_cancel_post = urllib.request.Request(
            f"http://127.0.0.1:8009/api/jobs/{c_job_id}/cancel",
            data=b"{}",
            headers={"Content-Type": "application/json"}
        )
        cancelled_ack = json.loads(urllib.request.urlopen(req_cancel_post).read().decode())
        print(f"Cancellation dispatched for {c_job_id}: status={cancelled_ack['status']}")
        time.sleep(1)
        st_cancelled = json.loads(urllib.request.urlopen(f"http://127.0.0.1:8009/api/jobs/{c_job_id}").read().decode())
        print(f"Final status of cancelled job: {st_cancelled['status']} ({st_cancelled['current_stage']})")
        assert st_cancelled["status"] == "cancelled", f"Job status was {st_cancelled['status']}, expected 'cancelled'"

        print("\nALL END-TO-END PIPELINE CHECKS (IF + DBSCAN + EXPORT + TRAINING REPRODUCTION + CANCELLATION) PASSED SUCCESSFULLY!")

    finally:
        p_back.terminate()

if __name__ == "__main__":
    test_pipeline()

