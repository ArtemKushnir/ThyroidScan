import abc
from typing import Optional

import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from src.image_data.image_data import Image
from src.training_module.feature_engineering_layer.plugin_registry import TransformRegistry
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

class FeaturePlugin(abc.ABC):
    @abc.abstractmethod
    def extract(self, image: Image) -> dict[str, float]:
        pass


class TransformPlugin(abc.ABC):
    def __init__(self) -> None:
        self._is_fit = False

    def fit(self, df: pd.DataFrame) -> None:
        self._fit(df)
        self._is_fit = True

    @abc.abstractmethod
    def _fit(self, df: pd.DataFrame) -> None:
        pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self._check_fit()
        return self._transform(df)

    @abc.abstractmethod
    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def _check_fit(self):
        if not self._is_fit:
            raise ValueError(" Call 'fit' with appropriate arguments before using this transform plugin")


@TransformRegistry.register_plugin("standard_scaler")
class StandardScalerTransformation(TransformPlugin):
    def __init__(self):
        super().__init__()
        self.scaler = StandardScaler()

    def _fit(self, df: pd.DataFrame) -> None:
        X = df.drop(columns=["tirads"])
        self.scaler.fit(X)

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.scaler.transform(df)

@TransformRegistry.register_plugin("pca")
class PCATransformation(TransformPlugin):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.pca = PCA(*args, **kwargs)

    def _fit(self, df: pd.DataFrame) -> None:
        X = df.drop(columns=["tirads"])
        self.pca.fit(X)

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.pca.transform(df)


@TransformRegistry.register_plugin("k_highest_mutual_info")
class KHighestMutualInfo(TransformPlugin):
    def __init__(self, k: int = 100):
        super().__init__()
        self.k: int = k
        self.columns: Optional[dict[str, float]] = None

    def _fit(self, df: pd.DataFrame) -> None:
        X = df.drop(columns=["tirads"])
        y = df["tirads"]
        if not pd.api.types.is_numeric_dtype(y):
            y = LabelEncoder().fit_transform(y)
        mi_scores = mutual_info_classif(X, y, discrete_features=False)
        mi_series = pd.Series(mi_scores, index=X.columns)

        self.columns = mi_series.sort_values(ascending=False).head(self.k).to_dict()

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return