import abc

from src.image_data.image_data import Image


class FeaturePlugin(abc.ABC):
    """Class for creating custom plugins for extracting features from an image"""

    @abc.abstractmethod
    def extract(self, image: Image) -> dict[str, float]:
        pass
