from src.image_data.image_data import Image
from src.training_module.data_layer.feature_pipeline import FeaturePipeline


class ImageProcessor:
    """Main image processor"""

    def __init__(self, plugin_names: list[str]) -> None:
        self.executor = FeaturePipeline(plugin_names)

    def run(self, images: list[Image]) -> None:
        self.executor.process(images)
