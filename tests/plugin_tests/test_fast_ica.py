import numpy as np
import pandas as pd
import pytest

from src.training_module.feature_engineering_layer.plugins.fast_ica import FastICATransformation


class TestFastICATransformation:

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        # Create mixed signals for ICA
        n_samples = 200
        time = np.linspace(0, 8, n_samples)

        s1 = np.sin(2 * time)  # Signal 1: sine wave
        s2 = np.sign(np.sin(3 * time))  # Signal 2: square wave
        s3 = time % 1  # Signal 3: sawtooth wave

        S = np.c_[s1, s2, s3]
        S += 0.2 * np.random.normal(size=S.shape)  # Add noise

        # Mix signals
        A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])  # Mixing matrix
        X = np.dot(S, A.T)  # Generate mixed signals

        data = {
            "feature1": X[:, 0],
            "feature2": X[:, 1],
            "feature3": X[:, 2],
            "target": np.random.choice([0, 1], n_samples),
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def transformer(self):
        return FastICATransformation(n_components=3, random_state=42)

    def test_initialization(self, transformer):
        from sklearn.decomposition import FastICA

        assert isinstance(transformer.ica, FastICA)
        assert transformer.ica.n_components == 3
        assert transformer.ica.random_state == 42
        assert transformer.numeric_columns == []

    def test_fit_processes_numeric_columns(self, transformer, sample_data):
        transformer.target_column = "target"
        transformer._fit(sample_data)

        assert len(transformer.numeric_columns) == 3
        assert hasattr(transformer.ica, "components_")
        assert hasattr(transformer.ica, "mixing_")

    def test_transform_creates_ica_features(self, transformer, sample_data):
        transformer.target_column = "target"
        transformer._fit(sample_data)

        transformed = transformer._transform(sample_data)

        # Should replace original features with ICA features
        ica_cols = [col for col in transformed.columns if col.startswith("ica_")]
        assert len(ica_cols) == 3  # n_components=3

        # Original numeric features should be removed
        for col in ["feature1", "feature2", "feature3"]:
            assert col not in transformed.columns

        # Target should remain
        assert "target" in transformed.columns
        assert transformed.shape[0] == sample_data.shape[0]

    def test_transform_with_no_numeric_columns(self, transformer):
        transformer.target_column = "target"
        # Fit with data that has numeric columns
        fit_data = pd.DataFrame({"feature1": [1, 2, 3], "target": [0, 1, 0]})
        transformer._fit(fit_data)

        # Transform data with no numeric columns
        transform_data = pd.DataFrame({"text_feature": ["a", "b", "c"], "target": [0, 1, 0]})

        result = transformer._transform(transform_data)
        assert result.equals(transform_data)  # Should return unchanged

    def test_state_persistence(self, transformer, sample_data):
        transformer.target_column = "target"
        transformer._fit(sample_data)

        state = transformer._get_state()
        new_transformer = FastICATransformation(n_components=3, random_state=42)
        new_transformer._set_state(state)

        assert new_transformer.numeric_columns == transformer.numeric_columns
        assert new_transformer.ica.n_features_in_ == transformer.ica.n_features_in_

        if hasattr(transformer.ica, "components_"):
            assert np.array_equal(new_transformer.ica.components_, transformer.ica.components_)
