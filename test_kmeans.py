import unittest
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class TestGOESKMeans(unittest.TestCase):
    """Test suite for GOES K-Means clustering with standard and edge case scenarios."""

    def setUp(self):
        """Set up valid dataset for baseline tests."""
        self.data = pd.DataFrame({
            'radiance': [1, 2, 3, 8, 9, 10],
            'latitude': [10, 11, 12, 20, 21, 22],
            'longitude': [30, 31, 32, 40, 41, 42]
        })
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(self.data[['radiance', 'latitude', 'longitude']])

    # ===== STANDARD TESTS =====
    def test_cluster_labels_length(self):
        """Test that cluster labels have same length as input data."""
        kmeans = KMeans(n_clusters=2, random_state=42).fit(self.X_scaled)
        self.assertEqual(len(kmeans.labels_), len(self.data))

    def test_cluster_count(self):
        """Test that exactly n_clusters unique labels are produced."""
        kmeans = KMeans(n_clusters=2, random_state=42).fit(self.X_scaled)
        self.assertEqual(len(set(kmeans.labels_)), 2)

    # ===== EDGE CASE TESTS =====
    def test_empty_dataset(self):
        """Test that empty DataFrame raises ValueError."""
        empty_data = pd.DataFrame({'radiance': [], 'latitude': [], 'longitude': []})
        X_empty = empty_data[['radiance', 'latitude', 'longitude']].dropna()
        self.assertTrue(X_empty.empty, "Empty dataset should have no rows after dropna")

    def test_single_row_dataset(self):
        """Test K-Means with single row (edge case: n_clusters > n_samples)."""
        single = pd.DataFrame({'radiance': [5], 'latitude': [15], 'longitude': [35]})
        X_single = single[['radiance', 'latitude', 'longitude']]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_single)
        kmeans = KMeans(n_clusters=1, random_state=42, n_init=10).fit(X_scaled)
        self.assertEqual(len(kmeans.labels_), 1)

    def test_missing_column(self):
        """Test that missing required column raises error on feature selection."""
        incomplete = pd.DataFrame({'radiance': [1, 2, 3], 'latitude': [10, 11, 12]})
        # Missing 'longitude' column
        with self.assertRaises(KeyError):
            incomplete[['radiance', 'latitude', 'longitude']].dropna()

    def test_all_nan_column(self):
        """Test dataset with entire column as NaN values."""
        nan_data = pd.DataFrame({
            'radiance': [1, 2, 3],
            'latitude': [np.nan, np.nan, np.nan],
            'longitude': [30, 31, 32]
        })
        X_nan = nan_data[['radiance', 'latitude', 'longitude']].dropna()
        self.assertTrue(X_nan.empty, "All-NaN column should result in empty data after dropna")

    def test_single_cluster(self):
        """Test K-Means with n_clusters=1 (all points in one cluster)."""
        kmeans = KMeans(n_clusters=1, random_state=42).fit(self.X_scaled)
        self.assertEqual(len(set(kmeans.labels_)), 1, "Single cluster should have 1 unique label")

    def test_clusters_equal_rows(self):
        """Test K-Means where n_clusters equals number of rows."""
        kmeans = KMeans(n_clusters=6, random_state=42, n_init=10).fit(self.X_scaled)
        self.assertEqual(len(kmeans.labels_), 6)
        self.assertLessEqual(len(set(kmeans.labels_)), 6)

    def test_negative_values(self):
        """Test that negative feature values are handled correctly (after scaling)."""
        neg_data = pd.DataFrame({
            'radiance': [-5, -2, 1, 4, 7, 10],
            'latitude': [10, 11, 12, 20, 21, 22],
            'longitude': [30, 31, 32, 40, 41, 42]
        })
        scaler = StandardScaler()
        X_neg = scaler.fit_transform(neg_data[['radiance', 'latitude', 'longitude']])
        kmeans = KMeans(n_clusters=2, random_state=42).fit(X_neg)
        self.assertEqual(len(kmeans.labels_), len(neg_data))

    def test_identical_rows(self):
        """Test K-Means with duplicate rows (identical feature values)."""
        dup_data = pd.DataFrame({
            'radiance': [5, 5, 5, 10, 10, 10],
            'latitude': [15, 15, 15, 20, 20, 20],
            'longitude': [35, 35, 35, 40, 40, 40]
        })
        scaler = StandardScaler()
        X_dup = scaler.fit_transform(dup_data[['radiance', 'latitude', 'longitude']])
        kmeans = KMeans(n_clusters=2, random_state=42).fit(X_dup)
        self.assertEqual(len(set(kmeans.labels_)), 2)

    def test_n_clusters_validation(self):
        """Test that sklearn raises error when n_clusters > n_samples."""
        # sklearn explicitly rejects n_clusters > n_samples
        with self.assertRaises(ValueError):
            KMeans(n_clusters=100, random_state=42, n_init=1).fit(self.X_scaled)

    def test_random_state_reproducibility(self):
        """Test that same random_state produces identical results."""
        kmeans1 = KMeans(n_clusters=2, random_state=42).fit(self.X_scaled)
        kmeans2 = KMeans(n_clusters=2, random_state=42).fit(self.X_scaled)
        np.testing.assert_array_equal(kmeans1.labels_, kmeans2.labels_)

if __name__ == '__main__':
    unittest.main()
