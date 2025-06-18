import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import RobustScaler

from src.training_module.feature_engineering_layer.plugins.robust_scaler import RobustScalerTransformation


class TestRobustScalerTransformation:

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        data = {
            "feature1": np.random.normal(0, 1, 100),
            "feature2": np.random.normal(5, 2, 100),
            "feature3": np.random.normal(-2, 0.5, 100),
            "target": np.random.choice([0, 1], 100),
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def transformer(self):
        return RobustScalerTransformation()

    def test_initialization(self, transformer):
        assert isinstance(transformer.scaler, RobustScaler)
        assert transformer.scaler.quantile_range == (25.0, 75.0)
        assert transformer.columns is None

    def test_fit_transform(self, transformer, sample_data):
        transformer.target_column = "target"
        transformer._fit(sample_data)

        assert transformer.columns is not None
        assert len(transformer.columns) == 3  # All features except target

        transformed = transformer._transform(sample_data)
        assert transformed.shape == sample_data.shape
        assert "target" in transformed.columns

    def test_transform_without_fit_raises_error(self, transformer, sample_data):
        transformer.target_column = "target"
        with pytest.raises(ValueError, match="Call 'fit' with appropriate arguments"):
            transformer._transform(sample_data)

    def test_state_persistence(self, transformer, sample_data):
        transformer.target_column = "target"
        transformer._fit(sample_data)

        state = transformer._get_state()
        new_transformer = RobustScalerTransformation()
        new_transformer._set_state(state)

        assert np.array_equal(new_transformer.scaler.center_, transformer.scaler.center_)
        assert np.array_equal(new_transformer.scaler.scale_, transformer.scaler.scale_)
