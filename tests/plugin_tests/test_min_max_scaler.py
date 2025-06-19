import numpy as np
import pandas as pd
import pytest

from src.training_module.feature_engineering_layer.plugins.min_max_scaler import MinMaxScalerTransformation


class TestMinMaxScalerTransformation:

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        data = {
            "feature1": np.random.uniform(0, 10, 100),
            "feature2": np.random.uniform(-5, 15, 100),
            "feature3": np.random.uniform(100, 200, 100),
            "target": np.random.choice([0, 1], 100),
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def transformer(self):
        return MinMaxScalerTransformation(feature_range=(0, 1))

    def test_initialization(self, transformer):
        from sklearn.preprocessing import MinMaxScaler

        assert isinstance(transformer.scaler, MinMaxScaler)
        assert transformer.scaler.feature_range == (0, 1)
        assert transformer.columns is None

    def test_fit_transform(self, transformer, sample_data):
        transformer.target_column = "target"
        transformer._fit(sample_data)

        assert transformer.columns is not None
        assert len(transformer.columns) == 3  # All features except target

        transformed = transformer._transform(sample_data)
        assert transformed.shape == sample_data.shape

        # Check if values are scaled to [0, 1]
        for col in ["feature1", "feature2", "feature3"]:
            assert transformed[col].min() >= 0
            assert transformed[col].max() <= 1 + 1e-6

    def test_custom_feature_range(self):
        transformer = MinMaxScalerTransformation(feature_range=(-1, 1))
        assert transformer.scaler.feature_range == (-1, 1)

    def test_transform_without_fit_raises_error(self, transformer, sample_data):
        transformer.target_column = "target"
        with pytest.raises(ValueError, match="Call 'fit' with appropriate arguments"):
            transformer._transform(sample_data)

    def test_state_persistence(self, transformer, sample_data):
        transformer.target_column = "target"
        transformer._fit(sample_data)

        state = transformer._get_state()
        new_transformer = MinMaxScalerTransformation()
        new_transformer._set_state(state)

        assert np.array_equal(new_transformer.scaler.data_min_, transformer.scaler.data_min_)
        assert np.array_equal(new_transformer.scaler.data_max_, transformer.scaler.data_max_)
        assert np.array_equal(new_transformer.scaler.scale_, transformer.scaler.scale_)
