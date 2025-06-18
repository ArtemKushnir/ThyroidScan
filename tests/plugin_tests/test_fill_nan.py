import numpy as np
import pandas as pd
import pytest

from src.training_module.feature_engineering_layer.plugins.fill_nan import FillNan


class TestFillNan:

    @pytest.fixture
    def sample_data_with_nan(self):
        np.random.seed(42)
        data = {
            "feature1": [1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, 10],
            "feature2": [np.nan, 2, 3, np.nan, 5, 6, 7, np.nan, 9, 10],
            "feature3": [1, 1, 2, 2, 2, 3, 3, 3, 3, 4],  # For mode testing
            "target": np.random.choice([0, 1], 10),
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def transformer_mean(self):
        return FillNan(strategy="mean")

    @pytest.fixture
    def transformer_median(self):
        return FillNan(strategy="median")

    @pytest.fixture
    def transformer_constant(self):
        return FillNan(strategy="constant", fill_value=999)

    def test_initialization(self, transformer_mean):
        assert transformer_mean.strategy == "mean"
        assert transformer_mean.fill_value is None
        assert transformer_mean.stats == {}
        assert transformer_mean.numeric_columns == []

    def test_invalid_strategy_raises_error(self):
        with pytest.raises(ValueError, match="Strategy 'invalid' is not supported"):
            FillNan(strategy="invalid")

    def test_fit_calculates_stats_mean(self, transformer_mean, sample_data_with_nan):
        transformer_mean.target_column = "target"
        transformer_mean._fit(sample_data_with_nan)

        assert len(transformer_mean.numeric_columns) == 3
        assert "feature1" in transformer_mean.stats
        # Check that mean is calculated correctly (ignoring NaN)
        expected_mean = sample_data_with_nan["feature1"].mean()
        assert abs(transformer_mean.stats["feature1"] - expected_mean) < 1e-10

    def test_fit_calculates_stats_median(self, transformer_median, sample_data_with_nan):
        transformer_median.target_column = "target"
        transformer_median._fit(sample_data_with_nan)

        expected_median = sample_data_with_nan["feature1"].median()
        assert transformer_median.stats["feature1"] == expected_median

    def test_constant_strategy_without_fill_value_raises_error(self):
        transformer = FillNan(strategy="constant")
        transformer.target_column = "target"

        sample_data = pd.DataFrame({"feature1": [1, np.nan, 3], "target": [0, 1, 0]})

        with pytest.raises(ValueError, match="fill_value must be specified"):
            transformer._fit(sample_data)

    def test_transform_fills_nan_values(self, transformer_mean, sample_data_with_nan):
        transformer_mean.target_column = "target"
        transformer_mean._fit(sample_data_with_nan)

        transformed = transformer_mean._transform(sample_data_with_nan)

        # Check that NaN values are filled
        assert not transformed["feature1"].isna().any()
        assert not transformed["feature2"].isna().any()
        assert transformed.shape == sample_data_with_nan.shape

    def test_transform_constant_strategy(self, transformer_constant, sample_data_with_nan):
        transformer_constant.target_column = "target"
        transformer_constant._fit(sample_data_with_nan)

        transformed = transformer_constant._transform(sample_data_with_nan)

        # Check that NaN values are filled with constant
        original_nan_mask = sample_data_with_nan["feature1"].isna()
        filled_values = transformed.loc[original_nan_mask, "feature1"]
        assert all(filled_values == 999)

    def test_state_persistence(self, transformer_mean, sample_data_with_nan):
        transformer_mean.target_column = "target"
        transformer_mean._fit(sample_data_with_nan)

        state = transformer_mean._get_state()
        new_transformer = FillNan(strategy="mean")
        new_transformer._set_state(state)

        assert new_transformer.strategy == transformer_mean.strategy
        assert new_transformer.stats == transformer_mean.stats
        assert new_transformer.numeric_columns == transformer_mean.numeric_columns
