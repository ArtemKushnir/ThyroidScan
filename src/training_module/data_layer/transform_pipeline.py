import json
import os
from typing import Any

import cv2
import numpy as np
import pandas as pd

from src.image_data.image_data import Image
from src.training_module.feature_engineering_layer.plugin_registry import TransformRegistry
from src.training_module.feature_engineering_layer.plugins.transform_plugin import TransformPlugin

DEFAULT_CONFIG_DIR = "training_module/config/plugin_config"

DEFAULT_STATE_DIR = "training_module/pipeline_artifacts/plugin_artifacts"


class TransformPipeline:
    """Pipeline for feature transform with state saving/loading capabilities"""

    def __init__(self, transform_plugins: list[tuple[str, dict]]) -> None:
        self.transform_plugins_config = transform_plugins
        self.transform_plugins = self._load_plugins(transform_plugins)
        self.is_fitted = False

    @staticmethod
    def _load_plugins(plugins: list[tuple[str, dict]]) -> list[TransformPlugin]:

        return [TransformRegistry.get_plugin(plugin[0], **plugin[1]) for plugin in plugins]

    def fit(self, images: list[Image]) -> None:
        df = self._get_df(images)
        curr_df = df.copy()
        for plugin in self.transform_plugins:
            plugin.fit(curr_df)
            curr_df = plugin.transform(curr_df)
            curr_df["tirads"] = df["tirads"]
        self.is_fitted = True

    def transform(self, images: list[Image]) -> None:
        if not self.is_fitted:
            raise ValueError("Pipeline is not fitted yet. Call 'fit' first.")

        curr_image_df = self._get_df(images)
        for plugin in self.transform_plugins:
            curr_image_df = plugin.transform(curr_image_df)

        features_dict = self._get_features_dict(curr_image_df)
        for i, image in enumerate(images):
            image.features = features_dict[i]

    def fit_transform(self, images: list[Image]) -> None:
        self.fit(images)
        self.transform(images)

    def _get_df(self, images: list[Image]) -> pd.DataFrame:
        df = pd.DataFrame([image.features for image in images])
        if not self.is_fitted:
            df["tirads"] = np.array([image.metadata["tirads"] for image in images])  # type: ignore
        return df

    @staticmethod
    def _get_features_dict(df: pd.DataFrame) -> list[dict[str, Any]]:
        index_dict = df.to_dict(orient="index")
        return [value for _, value in index_dict.items()]

    def save_state(self, directory: str = DEFAULT_STATE_DIR) -> None:
        for i, plugin in enumerate(self.transform_plugins):
            plugin_name = self.transform_plugins_config[i][0]
            plugin_path = os.path.join(directory, f"{i}_{plugin_name}.pkl")
            plugin.save_state(plugin_path)

    def save_config(self, directory: str = DEFAULT_CONFIG_DIR) -> None:
        os.makedirs(directory, exist_ok=True)

        config_path = os.path.join(directory, "pipeline_plugin_config.json")
        config = {
            "transform_plugins": self.transform_plugins_config,
        }
        with open(config_path, "w") as f:
            json.dump(config, f)

    @classmethod
    def load_config(cls, directory: str = DEFAULT_STATE_DIR) -> "TransformPipeline":
        config_path = os.path.join(directory, "pipeline_plugin_config.json")
        with open(config_path, "r") as f:
            config = json.load(f)

        pipeline = cls(config["transform_plugins"])
        return pipeline

    def load_state(self, directory: str = DEFAULT_STATE_DIR) -> None:
        self.is_fitted = True
        for i, plugin in enumerate(self.transform_plugins):
            plugin_name = self.transform_plugins_config[i][0]
            plugin_path = os.path.join(directory, f"{i}_{plugin_name}.pkl")
            plugin.load_state(plugin_path)

    @staticmethod
    def convert_to_grayscale(images: list[Image]) -> None:
        for image in images:
            if image.org_image is None:
                raise ValueError("Original image is not available in the Image object")

            if len(image.org_image.shape) == 3 and image.org_image.shape[2] == 3:
                gray_image = cv2.cvtColor(image.org_image, cv2.COLOR_BGR2GRAY)
            elif len(image.org_image.shape) == 2:
                gray_image = image.org_image
            else:
                raise ValueError(f"Unsupported image format with shape {image.org_image.shape}")
            image.org_image = gray_image
