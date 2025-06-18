import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA

from src.training_module.feature_engineering_layer.plugins.pca import PCATransformation


class TestPCATransformation:

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        data = {
            "feature1": np.random.normal(0, 1, 100),
            "feature2": np.random.normal(0, 1, 100),
            "feature3": np.random.normal(0, 1, 100),
            "target": np.random.choice([0, 1], 100),
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def transformer(self):
        return PCATransformation(n_components=2)

    def test_initialization(self, transformer):
        assert isinstance(transformer.pca, PCA)
        assert transformer.pca.n_components == 2
        assert transformer.columns is None

    def test_fit_transform(self, transformer, sample_data):
        transformer.target_column = "target"
        transformer._fit(sample_data)

        assert transformer.columns is not None
        assert len(transformer.columns) == 3  # All features except target

        transformed = transformer._transform(sample_data)
        assert transformed.shape[1] == 2  # n_components=2
        assert transformed.shape[0] == sample_data.shape[0]

    def test_transform_without_fit_raises_error(self, transformer, sample_data):
        transformer.target_column = "target"
        with pytest.raises(ValueError, match="Call 'fit' with appropriate arguments"):
            transformer._transform(sample_data)

    def test_state_persistence(self, transformer, sample_data):
        transformer.target_column = "target"
        transformer._fit(sample_data)

        state = transformer._get_state()
        new_transformer = PCATransformation(n_components=2)
        new_transformer._set_state(state)

        assert np.array_equal(new_transformer.pca.components_, transformer.pca.components_)
        assert np.array_equal(new_transformer.pca.mean_, transformer.pca.mean_)
