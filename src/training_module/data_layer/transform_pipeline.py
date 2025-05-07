import json
import os
from typing import Any

import numpy as np
import pandas as pd

from src.image_data.image_data import Image
from src.training_module.feature_engineering_layer.plugin_registry import TransformRegistry
from src.training_module.feature_engineering_layer.plugins.transform_plugin import TransformPlugin


class TransformPipeline:
    """Pipeline for feature transform with state saving/loading capabilities"""

    def __init__(self, transform_plugins: list[tuple[str, dict]], is_label: bool = True) -> None:
        self.transform_plugins_config = transform_plugins
        self.transform_plugins = self._load_plugins(transform_plugins)
        self.is_label = is_label
        self.is_fitted = False

    @staticmethod
    def _load_plugins(plugins: list[tuple[str, dict]]) -> list[TransformPlugin]:

        return [TransformRegistry.get_plugin(plugin[0], **plugin[1]) for plugin in plugins]

    def fit(self, images: list[Image]) -> None:
        df = self._get_df(images)
        for plugin in self.transform_plugins:
            plugin.fit(df)
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
        if self.is_label:
            df["tirads"] = np.array([image.metadata["tirads"] for image in images])  # type: ignore
        return df

    @staticmethod
    def _get_features_dict(df: pd.DataFrame) -> list[dict[str, Any]]:
        index_dict = df.to_dict(orient="index")
        return [value for _, value in index_dict.items()]

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)

        config_path = os.path.join(directory, "pipeline_config.json")
        config = {
            "transform_plugins": self.transform_plugins_config,
            "is_label": self.is_label,
            "is_fitted": self.is_fitted,
        }
        with open(config_path, "w") as f:
            json.dump(config, f)

        for i, plugin in enumerate(self.transform_plugins):
            plugin_name = self.transform_plugins_config[i][0]
            plugin_path = os.path.join(directory, f"{i}_{plugin_name}.pkl")
            plugin.save_state(plugin_path)

    @classmethod
    def load(cls, directory: str, is_label: bool = True) -> "TransformPipeline":
        config_path = os.path.join(directory, "pipeline_config.json")
        with open(config_path, "r") as f:
            config = json.load(f)

        pipeline = cls(config["transform_plugins"], is_label)
        pipeline.is_fitted = config["is_fitted"]

        for i, plugin in enumerate(pipeline.transform_plugins):
            plugin_name = pipeline.transform_plugins_config[i][0]
            plugin_path = os.path.join(directory, f"{i}_{plugin_name}.pkl")
            plugin.load_state(plugin_path)

        return pipeline
