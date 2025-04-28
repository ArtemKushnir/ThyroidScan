import abc

from src.image_data.image_data import Image


class Plugin(abc.ABC):
    @abc.abstractmethod
    def transform(self, image: Image) -> dict[str, float]:
        pass


class TransformPlugin(Plugin):
    """Class for applying various data conversion plugins"""

    @abc.abstractmethod
    def fit(self, images: list[Image]) -> None:
        pass

    @abc.abstractmethod
    def save(self, path: str) -> None:
        pass


class FeaturePlugin(Plugin):
    pass
