import numpy as np
import pandas as pd
import pytest

from src.training_module.feature_engineering_layer.plugins.mutual_info import KHighestMutualInfo


class TestKHighestMutualInfo:

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        # Create features with different mutual information with target
        data = {
            "informative1": np.random.normal(0, 1, 100),
            "informative2": np.random.normal(0, 1, 100),
            "noise1": np.random.normal(0, 1, 100),
            "noise2": np.random.normal(0, 1, 100),
            "target": np.random.choice([0, 1], 100),
        }
        df = pd.DataFrame(data)
        # Make some features more informative
        df.loc[df["target"] == 1, "informative1"] += 3
        df.loc[df["target"] == 1, "informative2"] += 2
        return df

    @pytest.fixture
    def transformer(self):
        return KHighestMutualInfo(k=2)

    def test_initialization(self, transformer):
        assert transformer.k == 2
        assert transformer.mutual_info is None

    def test_fit_calculates_mutual_info(self, transformer, sample_data):
        transformer.target_column = "target"
        transformer._fit(sample_data)

        assert transformer.mutual_info is not None
        assert len(transformer.mutual_info) == 2  # k=2
        assert all(isinstance(v, float) for v in transformer.mutual_info.values())

    def test_transform_selects_top_features(self, transformer, sample_data):
        transformer.target_column = "target"
        transformer._fit(sample_data)

        transformed = transformer._transform(sample_data)
        assert transformed.shape[1] == 2  # k=2
        assert all(col in transformer.mutual_info for col in transformed.columns)

    def test_transform_without_fit_raises_error(self, transformer, sample_data):
        transformer.target_column = "target"
        with pytest.raises(ValueError, match="Plugin not fitted"):
            transformer._transform(sample_data)

    def test_state_persistence(self, transformer, sample_data):
        transformer.target_column = "target"
        transformer._fit(sample_data)

        state = transformer._get_state()
        new_transformer = KHighestMutualInfo(k=2)
        new_transformer._set_state(state)

        assert new_transformer.mutual_info == transformer.mutual_info
        assert new_transformer.k == transformer.k
