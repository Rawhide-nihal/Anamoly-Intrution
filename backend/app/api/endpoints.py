import io
import json
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from typing import List, Optional, Dict, Any

from ..schemas.base import (
    DatasetInfo, ValidationReport, ModelInfo,
    AnalysisJobCreate, AnalysisJobStatus, AnalysisResultSummary,
    PaginatedPredictions, PredictionRow
)
from ..services.dataset_service import dataset_service
from ..services.research_service import research_service
from ..ml.service import ml_service
from ..jobs.manager import job_manager
from ..config import JOBS_DATA_DIR, PLOTS_DIR

router = APIRouter()

# ----------------- SYSTEM & HEALTH -----------------
@router.get("/health")
def get_health():
    return {
        "status": "healthy",
        "service": "CIC-IDS2017 Intrusion Detection Web Engine",
        "institution": "KLH University, KLH Bachupally Campus, Hyderabad",
        "department": "Computer Science and Engineering",
        "team": "Team 21",
        "guide": "Mr. Shaik Asif"
    }

# ----------------- DATASETS -----------------
@router.get("/datasets", response_model=List[DatasetInfo])
def list_datasets():
    return dataset_service.list_datasets()

@router.post("/datasets/upload")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
    contents = await file.read()
    if len(contents) > 200 * 1024 * 1024: # 200MB limit for local web demo
        raise HTTPException(status_code=400, detail="File exceeds 200MB limit.")
    dataset_id = dataset_service.save_upload(contents, file.filename)
    return {"dataset_id": dataset_id, "message": "Dataset uploaded and registered successfully."}

@router.get("/datasets/{dataset_id}/validate", response_model=ValidationReport)
def validate_dataset(dataset_id: str):
    try:
        return dataset_service.validate_dataset(dataset_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.get("/datasets/{dataset_id}/preview")
def preview_dataset(dataset_id: str, limit: int = Query(50, le=200)):
    try:
        return dataset_service.get_preview(dataset_id, limit=limit)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

# ----------------- MODELS -----------------
@router.get("/models", response_model=List[ModelInfo])
def list_models():
    return ml_service.get_models_inventory()

# ----------------- JOBS & ANALYSIS -----------------
@router.post("/jobs", response_model=Dict[str, str] if False else Any)
def create_job(job_req: AnalysisJobCreate):
    params = {
        "dbscan_eps": job_req.dbscan_eps,
        "dbscan_min_samples": job_req.dbscan_min_samples,
        "sample_limit": job_req.sample_limit
    }
    job_id = job_manager.create_job(
        dataset_id=job_req.dataset_id,
        model_id=job_req.model_id,
        algorithm=job_req.algorithm,
        params=params
    )
    return {"job_id": job_id, "status": "queued"}

@router.get("/jobs", response_model=List[AnalysisJobStatus])
def list_jobs():
    jobs = job_manager.list_jobs()
    return jobs

@router.get("/jobs/{job_id}", response_model=AnalysisJobStatus)
def get_job_status(job_id: str):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@router.get("/jobs/{job_id}/results", response_model=AnalysisResultSummary)
def get_job_results(job_id: str):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job status is {job['status']}, results not ready")
    summary = json.loads(job["results_summary_json"])
    return summary

@router.get("/jobs/{job_id}/predictions", response_model=PaginatedPredictions)
def get_job_predictions(
    job_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    filter_type: str = Query("all", pattern="^(all|normal|anomaly)$")
):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed")

    res_file = JOBS_DATA_DIR / f"{job_id}_results.npz"
    if not res_file.exists():
        raise HTTPException(status_code=404, detail="Result files not found")

    data = np.load(res_file)
    preds = data["preds"]
    scores = data["scores"] if "scores" in data else None
    clusters = data["clusters"] if "clusters" in data else None

    # Load dataset for row features preview
    ds = dataset_service.get_dataset(job["dataset_id"])
    df = pd.read_csv(ds["filepath"])
    df.columns = [str(c).strip() for c in df.columns]

    # Filter
    if filter_type == "normal":
        mask = (preds == 0)
    elif filter_type == "anomaly":
        mask = (preds == 1)
    else:
        mask = np.ones(len(preds), dtype=bool)

    indices = np.where(mask)[0]
    total_matching = len(indices)
    total_pages = max(1, int(np.ceil(total_matching / page_size)))
    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, total_matching)

    slice_indices = indices[start_idx:end_idx]

    rows = []
    # Pick a few key columns for row preview
    preview_cols = [c for c in ["Destination Port", "Flow Duration", "Total Fwd Packets", "Flow Bytes/s", "Packet Length Mean"] if c in df.columns]
    if len(preview_cols) == 0:
        preview_cols = list(df.columns[:5])

    for idx in slice_indices:
        pred_val = int(preds[idx])
        pred_label = "Anomaly" if pred_val == 1 else "Normal"
        sc = float(scores[idx]) if scores is not None else None
        cl = int(clusters[idx]) if clusters is not None else None
        gt = str(df["Label"].iloc[idx]) if "Label" in df.columns else None

        row_feat = {c: (None if pd.isna(df[c].iloc[idx]) else df[c].iloc[idx]) for c in preview_cols}

        rows.append(PredictionRow(
            row_index=int(idx),
            prediction=pred_val,
            prediction_label=pred_label,
            score=sc,
            cluster_id=cl,
            ground_truth=gt,
            features_preview=row_feat
        ))

    return PaginatedPredictions(
        job_id=job_id,
        total_rows=total_matching,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
        filter_type=filter_type,
        rows=rows
    )

@router.get("/jobs/{job_id}/export/csv")
def export_job_csv(job_id: str):
    job = job_manager.get_job(job_id)
    if not job or job["status"] != "completed":
        raise HTTPException(status_code=404, detail="Job not completed or not found")

    res_file = JOBS_DATA_DIR / f"{job_id}_results.npz"
    data = np.load(res_file)
    preds = data["preds"]
    scores = data["scores"] if "scores" in data else None
    clusters = data["clusters"] if "clusters" in data else None

    ds = dataset_service.get_dataset(job["dataset_id"])
    df = pd.read_csv(ds["filepath"])
    # If DBSCAN sampled, slice to sample size
    df = df.iloc[:len(preds)].copy()

    df["prediction"] = preds
    df["prediction_label"] = np.where(preds == 1, "Anomaly", "Normal")
    if scores is not None:
        df["anomaly_score"] = scores
    if clusters is not None:
        df["cluster_id"] = clusters

    stream = io.StringIO()
    df.to_csv(stream, index=False)
    stream.seek(0)

    filename = f"detection_results_{job['algorithm']}_{job_id[:8]}.csv"
    return StreamingResponse(
        iter([stream.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

# ----------------- RESEARCH BENCHMARKS & ARTIFACTS -----------------
@router.get("/research/status")
def get_research_status():
    return research_service.check_cicids_files_status()

@router.get("/research/phase3-comparison")
def get_phase3_comparison():
    return research_service.get_phase3_comparison()

@router.get("/research/metrics/{phase}")
def get_phase_metrics(phase: str):
    if phase == "phase2a":
        return research_service.get_phase2a_metrics()
    elif phase == "phase2b":
        return research_service.get_phase2b_metrics()
    elif phase == "phase2c":
        return research_service.get_phase2c_metrics()
    else:
        raise HTTPException(status_code=404, detail="Phase not found")

@router.get("/research/reports")
def get_research_reports():
    return research_service.get_reports()

@router.get("/research/plots")
def get_research_plots():
    return research_service.get_available_plots()

@router.get("/research/plots/{phase}/{filename}")
def get_plot_image(phase: str, filename: str):
    plot_path = PLOTS_DIR / phase / filename
    if not plot_path.exists():
        raise HTTPException(status_code=404, detail="Plot file not found")
    return FileResponse(plot_path, media_type="image/png")
