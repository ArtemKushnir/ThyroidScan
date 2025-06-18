import numpy as np
import pandas as pd
import pytest
from sklearn.feature_selection import SelectKBest

from src.training_module.feature_engineering_layer.plugins.select_k_best import SelectKBestTransformation


class TestSelectKBestTransformation:

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        # Create features with different correlations to target
        data = {
            "good_feature1": np.random.normal(0, 1, 100),
            "good_feature2": np.random.normal(0, 1, 100),
            "noise_feature1": np.random.normal(0, 1, 100),
            "noise_feature2": np.random.normal(0, 1, 100),
            "target": np.random.choice([0, 1], 100),
        }
        df = pd.DataFrame(data)
        # Make some features more correlated with target
        df.loc[df["target"] == 1, "good_feature1"] += 2
        df.loc[df["target"] == 1, "good_feature2"] += 1.5
        return df

    @pytest.fixture
    def transformer(self):
        return SelectKBestTransformation(k=2)

    def test_initialization(self, transformer):
        assert isinstance(transformer.selector, SelectKBest)
        assert transformer.selector.k == 2
        assert transformer.numeric_columns == []
        assert transformer.selected_features == []

    def test_fit_selects_features(self, transformer, sample_data):
        transformer.target_column = "target"
        transformer._fit(sample_data)

        assert len(transformer.numeric_columns) == 4  # All numeric features
        assert len(transformer.selected_features) == 2  # k=2
        assert transformer.selected_indices is not None

    def test_transform_filters_features(self, transformer, sample_data):
        transformer.target_column = "target"
        transformer._fit(sample_data)

        transformed = transformer._transform(sample_data)
        expected_cols = len(transformer.selected_features) + 1  # +1 for target
        assert transformed.shape[1] == expected_cols
        assert "target" in transformed.columns

    def test_transform_without_target(self, transformer, sample_data):
        transformer.target_column = "target"
        transformer._fit(sample_data)

        test_data = sample_data.drop(columns=["target"])
        transformed = transformer._transform(test_data)
        assert transformed.shape[1] == len(transformer.selected_features)

    def test_state_persistence(self, transformer, sample_data):
        transformer.target_column = "target"
        transformer._fit(sample_data)

        state = transformer._get_state()
        new_transformer = SelectKBestTransformation(k=2)
        new_transformer._set_state(state)

        assert new_transformer.numeric_columns == transformer.numeric_columns
        assert new_transformer.selected_features == transformer.selected_features
