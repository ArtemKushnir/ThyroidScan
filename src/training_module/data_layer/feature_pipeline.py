from typing import Type

from src.image_data.image_data import Image
from src.training_module.feature_engineering_layer.plugin_registry import (
    FeatureRegistry,
    PluginRegistry,
    TransformRegistry,
)
from src.training_module.feature_engineering_layer.plugins import Plugin, TransformPlugin


class DataPipeline:
    """Pipeline for feature processing and feature extraction"""

    def __init__(self, feature_plugins: list[str], transform_plugins: list[str]) -> None:
        self.feature_plugins = self._load_plugins(feature_plugins, FeatureRegistry)
        self.transform_plugins = self._load_plugins(transform_plugins, TransformRegistry)

    @staticmethod
    def _load_plugins(plugin_names: list[str], registry: Type[PluginRegistry]) -> list[Plugin]:
        return [registry.get_plugin(name) for name in plugin_names]

    def process(self, images: list[Image]) -> None:
        for image in images:
            self._process_single_feature_plugin(image)
        for tr_plugin in self.transform_plugins:
            if not isinstance(tr_plugin, TransformPlugin):
                raise TypeError("Plugin must be a TransformPlugin")
            tr_plugin.fit(images)
        for image in images:
            self._process_single_transform_plugin(image)

    def _process_single_feature_plugin(self, image: Image) -> None:
        if image.features is None:
            image.features = {}

        for plugin in self.feature_plugins:
            new_features = plugin.transform(image)
            image.features.update(new_features)

    def _process_single_transform_plugin(self, image: Image) -> None:
        if image.features is None:
            image.features = {}

        for plugin in self.transform_plugins:
            new_features = plugin.transform(image)
            image.features = new_features
