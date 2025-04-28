from src.image_data.image_data import Image
from src.training_module.data_layer.feature_pipeline import DataPipeline


class ImageHandler:
    """Main image processor"""

    def __init__(self, feature_plugins: list[str], transform_plugins: list[str]) -> None:
        self.feature_pipeline = DataPipeline(feature_plugins, transform_plugins)

    def run(self, images: list[Image]) -> None:
        self.feature_pipeline.process(images)
