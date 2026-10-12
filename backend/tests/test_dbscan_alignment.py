import sys
import unittest
import numpy as np
import pandas as pd
from pathlib import Path

# Ensure backend is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ml.service import ml_service
from app.services.dataset_service import dataset_service
from app.jobs.manager import job_manager

class TestDBSCANAlignment(unittest.TestCase):
    def test_dbscan_sampled_indices_and_alignment(self):
        """
        Construct a test dataset with 300 rows where each row has a unique identifier
        and known labels. Force a sample_limit of 50 rows.
        Assert that every prediction, ground-truth label, source-row identifier,
        and extracted row feature correspond to the exact same original row.
        """
        # Load sample dataset as base schema
        sample = dataset_service.get_dataset("sample_cicids2017")
        self.assertIsNotNone(sample, "Sample dataset must exist")
        df_base = pd.read_csv(sample["filepath"])

        # Repeat base data to create 300 unique rows
        df_300 = pd.concat([df_base, df_base, df_base], ignore_index=True)
        # Add distinct flow durations so each row is uniquely identifiable
        df_300["Flow Duration"] = [float(1000 + i) for i in range(len(df_300))]
        # Alternate labels deliberately: first 150 BENIGN, next 150 DoS Hulk
        labels = ["BENIGN"] * 150 + ["DoS Hulk"] * 150
        df_300["Label"] = labels

        sample_limit = 50
        preds, cluster_labels, source_indices, stats = ml_service.run_dbscan_clustering(
            df_300,
            eps=2.5522,
            min_samples=5,
            sample_limit=sample_limit,
            seed=42
        )

        # Verification 1: Array lengths match sample_limit
        self.assertEqual(len(preds), sample_limit)
        self.assertEqual(len(cluster_labels), sample_limit)
        self.assertEqual(len(source_indices), sample_limit)
        self.assertTrue(stats["is_sampled"])
        self.assertEqual(stats["original_size"], 300)
        self.assertEqual(stats["sample_size"], sample_limit)

        # Verification 2: source_indices are strictly within [0, 299]
        self.assertTrue(np.all(source_indices >= 0))
        self.assertTrue(np.all(source_indices < 300))
        self.assertEqual(len(np.unique(source_indices)), sample_limit, "Indices must be distinct")

        # Verification 3: Flow Duration on sliced dataframe matches exact source_indices
        extracted_durations = df_300["Flow Duration"].iloc[source_indices].to_numpy()
        for i, idx in enumerate(source_indices):
            expected_dur = float(1000 + idx)
            self.assertEqual(extracted_durations[i], expected_dur)
            # Label alignment check
            expected_label = "BENIGN" if idx < 150 else "DoS Hulk"
            actual_label = df_300["Label"].iloc[idx]
            self.assertEqual(actual_label, expected_label)

    def test_dbscan_un_sampled_dataset(self):
        """
        When dataset size <= sample_limit, source_indices must be np.arange(len(df))
        and is_sampled must be False.
        """
        sample = dataset_service.get_dataset("sample_cicids2017")
        df = pd.read_csv(sample["filepath"])

        preds, cluster_labels, source_indices, stats = ml_service.run_dbscan_clustering(
            df,
            eps=2.5522,
            min_samples=5,
            sample_limit=50000
        )
        self.assertEqual(len(preds), len(df))
        self.assertFalse(stats["is_sampled"])
        np.testing.assert_array_equal(source_indices, np.arange(len(df)))

    def test_k_distance_elbow_curve(self):
        """
        Verify k-distance elbow curve returns a sorted distance array and a positive suggested eps.
        """
        sample = dataset_service.get_dataset("sample_cicids2017")
        df = pd.read_csv(sample["filepath"])

        res = ml_service.compute_k_distance_elbow(df, k=5, sample_size=50)
        self.assertIn("suggested_eps", res)
        self.assertGreater(res["suggested_eps"], 0.0)
        self.assertIn("curve", res)
        self.assertGreater(len(res["curve"]), 0)

if __name__ == "__main__":
    unittest.main()
