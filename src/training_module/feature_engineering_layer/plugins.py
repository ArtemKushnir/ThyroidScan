import abc

import pandas as pd

from src.image_data.image_data import Image


class FeaturePlugin(abc.ABC):
    @abc.abstractmethod
    def extract(self, image: Image) -> dict[str, float]:
        pass


class TransformPlugin(abc.ABC):
    def __init__(self) -> None:
        self._is_fit = False

    @abc.abstractmethod
    def fit(self, df: pd.DataFrame) -> None:
        pass

    @abc.abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass
