from src.image_data.image_data import Image
from src.training_module.feature_engineering_layer.feature_plugin import FeaturePlugin
from src.training_module.feature_engineering_layer.feature_registry import FeatureRegistry


class FeaturePipeline:
    """Pipeline for image processing and feature extraction"""

    def __init__(self, plugin_names: list[str]) -> None:
        self.plugins = self._load_plugins(plugin_names)

    @staticmethod
    def _load_plugins(plugin_names: list[str]) -> list[FeaturePlugin]:
        return [FeatureRegistry.get_plugin(name) for name in plugin_names]

    def process(self, images: list[Image]) -> None:
        for image in images:
            self._process_single(image)

    def _process_single(self, image: Image) -> None:
        if image.features is None:
            image.features = {}

        for plugin in self.plugins:
            new_features = plugin.extract(image)
            image.features.update(new_features)

    @staticmethod
    def _validate_features(features: dict[str, float]) -> None:
        if not isinstance(features, dict):
            raise TypeError("Features must be a dictionary")
        for key, value in features.items():
            if not isinstance(key, str):
                raise TypeError("Feature name must be a string")
            if not isinstance(value, (int, float)):
                raise TypeError(f"Feature {key} value must be numeric")
