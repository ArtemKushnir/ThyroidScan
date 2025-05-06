import abc
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, FastICA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler

from src.training_module.feature_engineering_layer.plugin_registry import TransformRegistry


class TransformPlugin(abc.ABC):
    def __init__(self) -> None:
        self._is_fit = False
        self.target_column = "tirads"

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

    def _check_fit(self) -> None:
        if not self._is_fit:
            raise ValueError(" Call 'fit' with appropriate arguments before using this transform plugin")


@TransformRegistry.register_plugin("standard_scaler")
class StandardScalerTransformation(TransformPlugin):
    def __init__(self) -> None:
        super().__init__()
        self.scaler = StandardScaler()

    def _fit(self, df: pd.DataFrame) -> None:
        X = df.drop(columns=[self.target_column])
        self.scaler.fit(X)

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.scaler.transform(df)


@TransformRegistry.register_plugin("pca")
class PCATransformation(TransformPlugin):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.pca = PCA(*args, **kwargs)

    def _fit(self, df: pd.DataFrame) -> None:
        X = df.drop(columns=[self.target_column])
        self.pca.fit(X)

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.pca.transform(df)


@TransformRegistry.register_plugin("mutual_info")
class KHighestMutualInfo(TransformPlugin):
    def __init__(self, k: int = 100) -> None:
        super().__init__()
        self.k: int = k
        self.mutual_info: Optional[dict[str, float]] = None

    def _fit(self, df: pd.DataFrame) -> None:
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        if not pd.api.types.is_numeric_dtype(y):
            y = LabelEncoder().fit_transform(y)
        mi_scores = mutual_info_classif(X, y, discrete_features=False)
        mi_series = pd.Series(mi_scores, index=X.columns)

        self.mutual_info = mi_series.sort_values(ascending=False).head(self.k).to_dict()

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[list(self.mutual_info.keys())] #type: ignore
