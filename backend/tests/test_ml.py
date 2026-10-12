import sys
import unittest
import numpy as np
import pandas as pd
from pathlib import Path

# Ensure backend root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ml.service import ml_service
from app.services.dataset_service import dataset_service
from app.jobs.manager import job_manager

class TestMLService(unittest.TestCase):
    def test_models_inventory(self):
        models = ml_service.get_models_inventory()
        self.assertGreaterEqual(len(models), 3)
        ids = [m["id"] for m in models]
        self.assertIn("if_with_port", ids)
        self.assertIn("if_without_port", ids)
        self.assertIn("dbscan_clustering", ids)
        # Verify status 'ready' for valid model artifacts
        for m in models:
            self.assertEqual(m["status"], "ready")

    def test_isolation_forest_sample_inference(self):
        sample = dataset_service.get_dataset("sample_cicids2017")
        self.assertIsNotNone(sample)
        df = pd.read_csv(sample["filepath"])
        
        preds, scores, source_indices = ml_service.run_isolation_forest_inference(df, model_id="if_with_port")
        self.assertEqual(len(preds), len(df))
        self.assertEqual(len(scores), len(df))
        self.assertEqual(len(source_indices), len(df))
        self.assertTrue(set(np.unique(preds)).issubset({0, 1}))
        # Check score convention: score > 0 yields pred == 1
        np.testing.assert_array_equal((scores > 0).astype(np.int8), preds)

    def test_isolation_forest_no_port_inference(self):
        sample = dataset_service.get_dataset("sample_cicids2017")
        self.assertIsNotNone(sample)
        df = pd.read_csv(sample["filepath"])
        
        preds, scores, source_indices = ml_service.run_isolation_forest_inference(df, model_id="if_without_port")
        self.assertEqual(len(preds), len(df))
        self.assertEqual(len(scores), len(df))
        self.assertEqual(len(source_indices), len(df))
        self.assertTrue(set(np.unique(preds)).issubset({0, 1}))

    def test_dbscan_clustering_sample(self):
        sample = dataset_service.get_dataset("sample_cicids2017")
        df = pd.read_csv(sample["filepath"])
        preds, clusters, source_indices, stats = ml_service.run_dbscan_clustering(
            df, eps=2.5522, min_samples=5, sample_limit=100
        )
        self.assertEqual(len(preds), len(df))
        self.assertEqual(len(clusters), len(df))
        self.assertEqual(len(source_indices), len(df))
        self.assertGreaterEqual(stats["n_clusters"], 0)
        self.assertIn("noise_count", stats)
        self.assertIn("sampling_method", stats)

if __name__ == "__main__":
    unittest.main()
