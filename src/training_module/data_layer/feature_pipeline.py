from typing import Optional, Union

from src.image_data.image_data import Image
from src.training_module.feature_engineering_layer.feature_plugins import FeaturePlugin
from src.training_module.feature_engineering_layer.plugin_registry import FeatureRegistry


class FeaturePipeline:
    """Pipeline for feature extraction"""

    def __init__(self, feature_plugins: Optional[list[str]] = None) -> None:
        if feature_plugins is None:
            feature_plugins = []
        self.feature_plugins: list[FeaturePlugin] = self._load_plugins(feature_plugins)

    @staticmethod
    def _load_plugins(plugin_names: list[str]) -> list[FeaturePlugin]:
        return [FeatureRegistry.get_plugin(name) for name in plugin_names]

    def process(self, images: list[Image]) -> None:
        for image in images:
            self._process_single(image)

    def _process_single(self, image: Image) -> None:
        if image.features is None:
            image.features = {}

        for plugin in self.feature_plugins:
            new_features = plugin.extract(image)
            image.features.update(new_features)
