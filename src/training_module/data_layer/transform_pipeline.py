from typing import Any

import pandas as pd

from src.image_data.image_data import Image
from src.training_module.feature_engineering_layer.plugin_registry import TransformRegistry
from src.training_module.feature_engineering_layer.plugins import TransformPlugin


class TransformPipeline:
    """Pipeline for feature transform"""

    def __init__(self, transform_plugins: list[tuple[str, dict]]) -> None:
        self.transform_plugins = self._load_plugins(transform_plugins)

    @staticmethod
    def _load_plugins(plugins: list[tuple[str, dict]]) -> list[TransformPlugin]:
        return [TransformRegistry.get_plugin(plugin[0], **plugin[1]) for plugin in plugins]

    def fit(self, images: list[Image]) -> None:
        for plugin in self.transform_plugins:
            plugin.fit(self._get_df(images))

    def transform(self, images: list[Image]) -> None:
        curr_image_df = self._get_df(images)
        for plugin in self.transform_plugins:
            curr_image_df = plugin.transform(curr_image_df)
        features_dict = self._get_features_dict(curr_image_df)
        for i, image in enumerate(images):
            image.features = features_dict[i]

    @staticmethod
    def _get_df(images: list[Image]) -> pd.DataFrame:
        return pd.DataFrame([image.features for image in images])

    @staticmethod
    def _get_features_dict(df: pd.DataFrame) -> list[dict[str, Any]]:
        index_dict = df.to_dict(orient="index")
        return [value for _, value in index_dict.items()]
