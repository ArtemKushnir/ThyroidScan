import numpy as np
import pandas as pd
import pytest

from src.training_module.feature_engineering_layer.plugins.outlier_removal import OutlierRemovalTransformation


class TestOutlierRemovalTransformation:

    @pytest.fixture
    def sample_data_with_outliers(self):
        np.random.seed(42)
        data = {
            "feature1": np.concatenate([np.random.normal(0, 1, 95), [10, -10, 15, -15, 20]]),  # Add outliers
            "feature2": np.concatenate([np.random.normal(5, 2, 95), [50, -20, 30, -25, 40]]),  # Add outliers
            "target": np.random.choice([0, 1], 100),
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def transformer_clip(self):
        return OutlierRemovalTransformation(threshold=3.0, method="clip")

    @pytest.fixture
    def transformer_remove(self):
        return OutlierRemovalTransformation(threshold=3.0, method="remove")

    def test_initialization(self, transformer_clip):
        assert transformer_clip.threshold == 3.0
        assert transformer_clip.method == "clip"
        assert transformer_clip.numeric_columns == []

    def test_fit_calculates_stats(self, transformer_clip, sample_data_with_outliers):
        transformer_clip.target_column = "target"
        transformer_clip._fit(sample_data_with_outliers)

        assert len(transformer_clip.numeric_columns) == 2
        assert "feature1" in transformer_clip.means
        assert "feature2" in transformer_clip.means
        assert "feature1" in transformer_clip.stds
        assert "feature2" in transformer_clip.stds

    def test_transform_clip_method(self, transformer_clip, sample_data_with_outliers):
        transformer_clip.target_column = "target"
        transformer_clip._fit(sample_data_with_outliers)

        original_max = sample_data_with_outliers["feature1"].max()
        transformed = transformer_clip._transform(sample_data_with_outliers)
        new_max = transformed["feature1"].max()

        assert new_max < original_max  # Outliers should be clipped
        assert transformed.shape[0] == sample_data_with_outliers.shape[0]  # No rows removed

    def test_transform_remove_method(self, transformer_remove, sample_data_with_outliers):
        transformer_remove.target_column = "target"
        transformer_remove._fit(sample_data_with_outliers)

        transformed = transformer_remove._transform(sample_data_with_outliers)
        assert transformed.shape[0] < sample_data_with_outliers.shape[0]  # Some rows removed

    def test_state_persistence(self, transformer_clip, sample_data_with_outliers):
        transformer_clip.target_column = "target"
        transformer_clip._fit(sample_data_with_outliers)

        state = transformer_clip._get_state()
        new_transformer = OutlierRemovalTransformation()
        new_transformer._set_state(state)

        assert new_transformer.threshold == transformer_clip.threshold
        assert new_transformer.method == transformer_clip.method
        assert new_transformer.means == transformer_clip.means
        assert new_transformer.stds == transformer_clip.stds
