import abc

from src.image_data.image_data import Image


class FeaturePlugin(abc.ABC):
    @abc.abstractmethod
    def extract(self, image: Image) -> dict[str, float]:
        pass
