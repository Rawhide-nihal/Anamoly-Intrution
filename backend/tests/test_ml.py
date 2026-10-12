import unittest
import numpy as np
import pandas as pd
from app.ml.service import ml_service
from app.services.dataset_service import dataset_service
from app.jobs.manager import job_manager

def test_models_inventory():
    models = ml_service.get_models_inventory()
    assert len(models) >= 3
    ids = [m["id"] for m in models]
    assert "if_with_port" in ids
    assert "if_without_port" in ids
    assert "dbscan_clustering" in ids

def test_isolation_forest_sample_inference():
    # Load sample dataset
    sample = dataset_service.get_dataset("sample_cicids2017")
    assert sample is not None
    df = pd.read_csv(sample["filepath"])
    
    # Run IF inference
    preds, scores = ml_service.run_isolation_forest_inference(df, model_id="if_with_port")
    assert len(preds) == len(df)
    assert len(scores) == len(df)
    # Check predictions are binary 0 or 1
    assert set(np.unique(preds)).issubset({0, 1})
    # Check score convention: score > 0 yields pred == 1
    assert np.all((scores > 0).astype(np.int8) == preds)

def test_dbscan_clustering_sample():
    sample = dataset_service.get_dataset("sample_cicids2017")
    df = pd.read_csv(sample["filepath"])
    preds, clusters, stats = ml_service.run_dbscan_clustering(df, eps=2.5522, min_samples=5, sample_limit=100)
    assert len(preds) == len(df)
    assert len(clusters) == len(df)
    assert stats["n_clusters"] >= 0
    assert "noise_count" in stats
