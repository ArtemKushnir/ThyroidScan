import numpy as np
import pandas as pd
import pytest

from src.training_module.feature_engineering_layer.plugins.k_means import KMeansClusteringFeatures


class TestKMeansClusteringFeatures:

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        # Create data with clear clusters
        cluster1 = np.random.normal([0, 0], 1, (50, 2))
        cluster2 = np.random.normal([5, 5], 1, (50, 2))
        features = np.vstack([cluster1, cluster2])

        data = {"feature1": features[:, 0], "feature2": features[:, 1], "target": np.random.choice([0, 1], 100)}
        return pd.DataFrame(data)

    @pytest.fixture
    def transformer(self):
        return KMeansClusteringFeatures(n_clusters=3, random_state=42)

    def test_initialization(self, transformer):
        from sklearn.cluster import KMeans

        assert isinstance(transformer.kmeans, KMeans)
        assert transformer.n_clusters == 3
        assert transformer.random_state == 42
        assert transformer.is_fitted == False
        assert transformer.numeric_columns == []

    def test_fit_creates_clusters(self, transformer, sample_data):
        transformer.target_column = "target"
        transformer._fit(sample_data)

        assert transformer.is_fitted == True
        assert transformer.cluster_centers is not None
        assert transformer.cluster_centers.shape == (3, 2)  # 3 clusters, 2 features
        assert len(transformer.numeric_columns) == 2

    def test_transform_adds_cluster_features(self, transformer, sample_data):
        transformer.target_column = "target"
        transformer._fit(sample_data)

        transformed = transformer._transform(sample_data)

        # Should add cluster label + distance features
        expected_new_cols = 1 + transformer.n_clusters  # cluster + distances
        assert transformed.shape[1] == sample_data.shape[1] + expected_new_cols
        assert "kmeans_cluster" in transformed.columns

        # Check distance columns
        for i in range(transformer.n_clusters):
            assert f"dist_to_cluster_{i}" in transformed.columns
            assert transformed[f"dist_to_cluster_{i}"].min() >= 0  # Distances are non-negative

    def test_manual_predict_method(self, transformer, sample_data):
        transformer.target_column = "target"
        transformer._fit(sample_data)

        test_points = np.array([[0, 0], [5, 5]])
        predictions = transformer._manual_predict(test_points)

        assert predictions.shape == (2,)
        assert all(0 <= pred < transformer.n_clusters for pred in predictions)

    def test_state_persistence(self, transformer, sample_data):
        transformer.target_column = "target"
        transformer._fit(sample_data)

        state = transformer._get_state()
        new_transformer = KMeansClusteringFeatures(n_clusters=3, random_state=42)
        new_transformer._set_state(state)

        assert new_transformer.is_fitted == transformer.is_fitted
        assert np.array_equal(new_transformer.cluster_centers, transformer.cluster_centers)
        assert new_transformer.numeric_columns == transformer.numeric_columns
