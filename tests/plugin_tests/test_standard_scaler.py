import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from sklearn.preprocessing import StandardScaler

from src.training_module.feature_engineering_layer.plugins.standard_scaler import StandardScalerTransformation


class TestStandardScalerTransformation:

    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample DataFrame for testing"""
        return pd.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "feature2": [10.0, 20.0, 30.0, 40.0, 50.0],
                "feature3": [0.1, 0.2, 0.3, 0.4, 0.5],
                "target": [0, 1, 0, 1, 0],
            }
        )

    @pytest.fixture
    def sample_dataframe_no_target(self):
        """Create a sample DataFrame without target column"""
        return pd.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "feature2": [10.0, 20.0, 30.0, 40.0, 50.0],
                "feature3": [0.1, 0.2, 0.3, 0.4, 0.5],
            }
        )

    def test_init(self):
        """Test StandardScalerTransformation initialization"""
        scaler_transform = StandardScalerTransformation()

        assert isinstance(scaler_transform.scaler, StandardScaler)
        assert scaler_transform.columns is None
        assert scaler_transform.target_column == "target"  # Default from parent class

    def test_fit_with_target_column(self, sample_dataframe):
        """Test _fit method with target column present"""
        scaler_transform = StandardScalerTransformation()

        scaler_transform._fit(sample_dataframe)

        # Check that scaler was fitted
        assert hasattr(scaler_transform.scaler, "mean_")
        assert hasattr(scaler_transform.scaler, "scale_")
        assert scaler_transform.columns is not None

        # Check that target column was excluded
        expected_columns = ["feature1", "feature2", "feature3"]
        assert list(scaler_transform.columns) == expected_columns

    def test_fit_without_target_column(self, sample_dataframe_no_target):
        """Test _fit method without target column"""
        scaler_transform = StandardScalerTransformation()

        scaler_transform._fit(sample_dataframe_no_target)

        # Check that all columns were used for fitting
        expected_columns = ["feature1", "feature2", "feature3"]
        assert list(scaler_transform.columns) == expected_columns

    def test_transform_after_fit(self, sample_dataframe):
        """Test _transform method after fitting"""
        scaler_transform = StandardScalerTransformation()
        scaler_transform._fit(sample_dataframe)

        transformed_df = scaler_transform._transform(sample_dataframe)

        # Check that DataFrame structure is preserved
        assert list(transformed_df.columns) == list(sample_dataframe.columns)
        assert len(transformed_df) == len(sample_dataframe)

        # Check that target column was not transformed
        pd.testing.assert_series_equal(transformed_df["target"], sample_dataframe["target"])

        # Check that features were standardized (mean ≈ 0, std ≈ 1)
        for col in ["feature1", "feature2", "feature3"]:
            feature_values = transformed_df[col].values
            assert abs(np.mean(feature_values)) < 1e-10  # Should be very close to 0
            assert abs(np.std(feature_values, ddof=0) - 1.0) < 1e-10  # Should be close to 1

    def test_transform_without_fit_raises_error(self, sample_dataframe):
        """Test that _transform raises error when not fitted"""
        scaler_transform = StandardScalerTransformation()

        with pytest.raises(ValueError, match="Call 'fit' with appropriate arguments"):
            scaler_transform._transform(sample_dataframe)

    def test_get_state(self, sample_dataframe):
        """Test _get_state method"""
        scaler_transform = StandardScalerTransformation()
        scaler_transform._fit(sample_dataframe)

        state = scaler_transform._get_state()

        # Check that all necessary state is saved
        required_keys = ["mean_", "var_", "scale_", "n_features_in_", "n_samples_seen_", "columns"]
        for key in required_keys:
            assert key in state

        # Check types and values
        assert isinstance(state["mean_"], np.ndarray)
        assert isinstance(state["var_"], np.ndarray)
        assert isinstance(state["scale_"], np.ndarray)
        assert isinstance(state["n_features_in_"], int)
        assert isinstance(state["columns"], list)
        assert len(state["columns"]) == 3  # feature1, feature2, feature3

    def test_set_state(self, sample_dataframe):
        """Test _set_state method"""
        # First, create a fitted scaler to get state
        scaler_transform1 = StandardScalerTransformation()
        scaler_transform1._fit(sample_dataframe)
        state = scaler_transform1._get_state()

        # Create a new scaler and set its state
        scaler_transform2 = StandardScalerTransformation()
        scaler_transform2._set_state(state)

        # Check that state was set correctly
        assert np.array_equal(scaler_transform2.scaler.mean_, scaler_transform1.scaler.mean_)
        assert np.array_equal(scaler_transform2.scaler.var_, scaler_transform1.scaler.var_)
        assert np.array_equal(scaler_transform2.scaler.scale_, scaler_transform1.scaler.scale_)
        assert scaler_transform2.scaler.n_features_in_ == scaler_transform1.scaler.n_features_in_
        assert list(scaler_transform2.columns) == list(scaler_transform1.columns)

    def test_state_persistence_with_feature_names(self, sample_dataframe):
        """Test state persistence when sklearn scaler has feature_names_in_"""
        scaler_transform = StandardScalerTransformation()
        scaler_transform._fit(sample_dataframe)

        # Manually set feature_names_in_ to test this path
        scaler_transform.scaler.feature_names_in_ = np.array(["feature1", "feature2", "feature3"])

        state = scaler_transform._get_state()
        assert "feature_names_in_" in state
        assert state["feature_names_in_"] is not None

        # Test setting state with feature names
        new_scaler = StandardScalerTransformation()
        new_scaler._set_state(state)
        assert hasattr(new_scaler.scaler, "feature_names_in_")
        assert np.array_equal(new_scaler.scaler.feature_names_in_, ["feature1", "feature2", "feature3"])

    def test_full_pipeline_integration(self, sample_dataframe):
        """Test full fit -> transform -> save -> load -> transform pipeline"""
        scaler_transform = StandardScalerTransformation()

        # Fit and transform
        scaler_transform.fit(sample_dataframe)
        transformed_df1 = scaler_transform.transform(sample_dataframe)

        # Save and load state
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = os.path.join(temp_dir, "scaler_state.pkl")
            scaler_transform.save_state(state_path)

            # Create new scaler and load state
            new_scaler = StandardScalerTransformation()
            new_scaler.load_state(state_path)

            # Transform with loaded scaler
            transformed_df2 = new_scaler.transform(sample_dataframe)

            # Results should be identical
            pd.testing.assert_frame_equal(transformed_df1, transformed_df2)

    @given(
        st.lists(
            st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False), min_size=5, max_size=100
        )
    )
    def test_standardization_properties_hypothesis(self, values):
        """Test standardization properties with hypothesis-generated data"""
        assume(len(set(values)) > 1)  # Need variance > 0
        assume(np.std(values) > 1e-10)  # Avoid numerical issues

        # Create DataFrame
        df = pd.DataFrame({"feature1": values, "target": [i % 2 for i in range(len(values))]})

        scaler_transform = StandardScalerTransformation()
        scaler_transform._fit(df)
        transformed_df = scaler_transform._transform(df)

        # Check standardization properties
        transformed_values = transformed_df["feature1"].values
        assert abs(np.mean(transformed_values)) < 1e-10
        assert abs(np.std(transformed_values, ddof=0) - 1.0) < 1e-10

    @given(st.integers(min_value=2, max_value=20), st.integers(min_value=5, max_value=100))
    def test_multiple_features_hypothesis(self, num_features, num_samples):
        """Test with hypothesis-generated number of features and samples"""
        # Generate random data
        data = {}
        for i in range(num_features):
            # Generate data with different scales
            scale = 10 ** (i % 4)  # Different orders of magnitude
            data[f"feature_{i}"] = np.random.normal(0, scale, num_samples)
        data["target"] = np.random.randint(0, 2, num_samples)

        df = pd.DataFrame(data)

        scaler_transform = StandardScalerTransformation()
        scaler_transform._fit(df)
        transformed_df = scaler_transform._transform(df)

        # Check that all features are standardized
        for i in range(num_features):
            col = f"feature_{i}"
            values = transformed_df[col].values
            assert abs(np.mean(values)) < 1e-10
            if np.std(values) > 1e-10:  # Avoid zero variance case
                assert abs(np.std(values, ddof=0) - 1.0) < 1e-10

    def test_single_sample(self):
        """Test behavior with single sample (zero variance)"""
        df = pd.DataFrame({"feature1": [5.0], "target": [1]})

        scaler_transform = StandardScalerTransformation()
        scaler_transform._fit(df)

        # Transform should handle zero variance case
        transformed_df = scaler_transform._transform(df)

        # With zero variance, sklearn typically returns zeros
        assert len(transformed_df) == 1
        assert "feature1" in transformed_df.columns
