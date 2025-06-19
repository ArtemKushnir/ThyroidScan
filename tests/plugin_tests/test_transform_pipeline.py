import json
import os
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.image_data.image_data import Image
from src.training_module.feature_engineering_layer.transform_pipeline import TransformPipeline


class MockTransformPlugin:
    """Mock plugin for testing"""

    def __init__(self, name="mock_plugin"):
        self.name = name
        self.fitted = False

    def fit(self, df):
        self.fitted = True

    def transform(self, df):
        return df.copy()

    def save_state(self, path):
        pass

    def load_state(self, path):
        pass


class TestTransformPipeline:

    def create_sample_image(self, name="test_image.jpg", target=1, features=None):
        """Create a sample Image object for testing"""
        image_data = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        if features is None:
            features = {"feature1": 1.0, "feature2": 2.0}
        image = Image(name, image_data, metadata={"target": target})
        image.features = features
        return image

    @pytest.fixture
    def sample_images(self):
        """Create a list of sample images"""
        images = []
        for i in range(5):
            features = {"feature1": float(i), "feature2": float(i * 2)}
            image = self.create_sample_image(f"test_image_{i}.jpg", i % 2, features)
            images.append(image)
        return images

    @pytest.fixture
    def mock_plugins_config(self):
        return [("mock_plugin_1", {}), ("mock_plugin_2", {"param": "value"})]

    @patch("src.training_module.feature_engineering_layer.transform_pipeline.TransformRegistry.get_plugin")
    def test_init(self, mock_get_plugin, mock_plugins_config):
        """Test TransformPipeline initialization"""
        mock_get_plugin.return_value = MockTransformPlugin()

        pipeline = TransformPipeline(mock_plugins_config)

        assert pipeline.transform_plugins_config == mock_plugins_config
        assert len(pipeline.transform_plugins) == 2
        assert not pipeline.is_fitted
        assert mock_get_plugin.call_count == 2

    @patch("src.training_module.feature_engineering_layer.transform_pipeline.TransformRegistry.get_plugin")
    def test_load_plugins(self, mock_get_plugin):
        """Test _load_plugins static method"""
        mock_get_plugin.return_value = MockTransformPlugin()
        plugins_config = [("plugin1", {}), ("plugin2", {"param": "value"})]

        plugins = TransformPipeline._load_plugins(plugins_config)

        assert len(plugins) == 2
        assert mock_get_plugin.call_count == 2
        mock_get_plugin.assert_any_call("plugin1")
        mock_get_plugin.assert_any_call("plugin2", param="value")

    @patch("src.training_module.feature_engineering_layer.transform_pipeline.TransformRegistry.get_plugin")
    def test_fit(self, mock_get_plugin, sample_images):
        """Test fit method"""
        mock_plugin = MockTransformPlugin()
        mock_get_plugin.return_value = mock_plugin

        pipeline = TransformPipeline([("mock_plugin", {})])
        pipeline.fit(sample_images)

        assert pipeline.is_fitted
        assert mock_plugin.fitted

    @patch("src.training_module.feature_engineering_layer.transform_pipeline.TransformRegistry.get_plugin")
    def test_transform_without_fit_raises_error(self, mock_get_plugin, sample_images):
        """Test transform method raises error when not fitted"""
        mock_get_plugin.return_value = MockTransformPlugin()

        pipeline = TransformPipeline([("mock_plugin", {})])

        with pytest.raises(ValueError, match="Pipeline is not fitted yet"):
            pipeline.transform(sample_images)

    @patch("src.training_module.feature_engineering_layer.transform_pipeline.TransformRegistry.get_plugin")
    def test_transform_after_fit(self, mock_get_plugin, sample_images):
        """Test transform method after fitting"""
        mock_plugin = MockTransformPlugin()
        mock_get_plugin.return_value = mock_plugin

        pipeline = TransformPipeline([("mock_plugin", {})])
        pipeline.fit(sample_images)

        # Should not raise error
        pipeline.transform(sample_images)

        # Check that features were updated
        for image in sample_images:
            assert image.features is not None

    @patch("src.training_module.feature_engineering_layer.transform_pipeline.TransformRegistry.get_plugin")
    def test_fit_transform(self, mock_get_plugin, sample_images):
        """Test fit_transform method"""
        mock_plugin = MockTransformPlugin()
        mock_get_plugin.return_value = mock_plugin

        pipeline = TransformPipeline([("mock_plugin", {})])
        pipeline.fit_transform(sample_images)

        assert pipeline.is_fitted
        assert mock_plugin.fitted

    def test_get_df_with_target(self, sample_images):
        """Test _get_df method when pipeline is not fitted (includes target)"""
        pipeline = TransformPipeline([])

        df = pipeline._get_df(sample_images)

        assert "target" in df.columns
        assert len(df) == len(sample_images)
        assert "feature1" in df.columns
        assert "feature2" in df.columns

    def test_get_df_without_target(self, sample_images):
        """Test _get_df method when pipeline is fitted (no target)"""
        pipeline = TransformPipeline([])
        pipeline.is_fitted = True

        df = pipeline._get_df(sample_images)

        assert "target" not in df.columns
        assert len(df) == len(sample_images)
        assert "feature1" in df.columns
        assert "feature2" in df.columns

    def test_get_features_dict(self):
        """Test _get_features_dict static method"""
        df = pd.DataFrame({"feature1": [1.0, 2.0, 3.0], "feature2": [4.0, 5.0, 6.0]})

        features_dict = TransformPipeline._get_features_dict(df)

        assert len(features_dict) == 3
        assert features_dict[0] == {"feature1": 1.0, "feature2": 4.0}
        assert features_dict[1] == {"feature1": 2.0, "feature2": 5.0}
        assert features_dict[2] == {"feature1": 3.0, "feature2": 6.0}

    @patch("src.training_module.feature_engineering_layer.transform_pipeline.TransformRegistry.get_plugin")
    def test_save_state(self, mock_get_plugin):
        """Test save_state method"""
        mock_plugin = MockTransformPlugin()
        mock_get_plugin.return_value = mock_plugin

        pipeline = TransformPipeline([("mock_plugin", {})])

        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline.save_state(temp_dir)
            # Test passes if no exception is raised

    @patch("src.training_module.feature_engineering_layer.transform_pipeline.TransformRegistry.get_plugin")
    def test_save_config(self, mock_get_plugin):
        """Test save_config method"""
        mock_get_plugin.return_value = MockTransformPlugin()
        config = [("mock_plugin", {"param": "value"})]

        pipeline = TransformPipeline(config)

        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline.save_config(temp_dir)

            config_path = os.path.join(temp_dir, "pipeline_plugin_config.json")
            assert os.path.exists(config_path)

            with open(config_path, "r") as f:
                saved_config = json.load(f)

            for i in range(len(config)):
                config[i] = list(config[i])

            assert saved_config["transform_plugins"] == config

    @patch("src.training_module.feature_engineering_layer.transform_pipeline.TransformRegistry.get_plugin")
    def test_load_config(self, mock_get_plugin):
        """Test load_config class method"""
        mock_get_plugin.return_value = MockTransformPlugin()
        config = {"transform_plugins": [("mock_plugin", {})]}

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "pipeline_plugin_config.json")
            with open(config_path, "w") as f:
                json.dump(config, f)

            pipeline = TransformPipeline.load_config(temp_dir)

            plugins = config["transform_plugins"]
            for i in range(len(plugins)):
                plugins[i] = list(plugins[i])

            assert pipeline.transform_plugins_config == config["transform_plugins"]

    @patch("src.training_module.feature_engineering_layer.transform_pipeline.TransformRegistry.get_plugin")
    def test_load_state(self, mock_get_plugin):
        """Test load_state method"""
        mock_plugin = MockTransformPlugin()
        mock_get_plugin.return_value = mock_plugin

        pipeline = TransformPipeline([("mock_plugin", {})])

        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline.load_state(temp_dir)
            assert pipeline.is_fitted

    @given(st.lists(st.integers(min_value=1, max_value=10), min_size=1, max_size=5))
    def test_pipeline_with_multiple_plugins(self, plugin_count_list):
        """Test pipeline with multiple plugins using hypothesis"""
        with patch(
            "src.training_module.feature_engineering_layer.transform_pipeline.TransformRegistry.get_plugin"
        ) as mock_get_plugin:
            mock_get_plugin.return_value = MockTransformPlugin()

            plugins_config = [(f"plugin_{i}", {}) for i in range(len(plugin_count_list))]
            pipeline = TransformPipeline(plugins_config)

            assert len(pipeline.transform_plugins) == len(plugin_count_list)
            assert mock_get_plugin.call_count == len(plugin_count_list)
