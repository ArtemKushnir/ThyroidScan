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
        return df[list(self.mutual_info.keys())]  # type: ignore


@TransformRegistry.register_plugin("min_max_scaler")
class MinMaxScalerTransformation(TransformPlugin):
    def __init__(self, feature_range: Any = (0, 1)) -> None:
        super().__init__()
        self.scaler = MinMaxScaler(feature_range=feature_range)

    def _fit(self, df: pd.DataFrame) -> None:
        X = df.drop(columns=[self.target_column], errors="ignore")
        self.columns = X.columns
        self.scaler.fit(X)

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result_df = df.copy()
        features = [col for col in self.columns if col in df.columns]
        result_df[features] = self.scaler.transform(df[features])
        return result_df


@TransformRegistry.register_plugin("robust_scaler")
class RobustScalerTransformation(TransformPlugin):
    def __init__(self, quantile_range: Any = (25.0, 75.0)) -> None:
        super().__init__()
        self.scaler = RobustScaler(quantile_range=quantile_range)
        self.target_column = "tirads"

    def _fit(self, df: pd.DataFrame) -> None:
        X = df.drop(columns=[self.target_column], errors="ignore")
        self.columns = X.columns
        self.scaler.fit(X)

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result_df = df.copy()
        features = [col for col in self.columns if col in df.columns]
        result_df[features] = self.scaler.transform(df[features])
        return result_df


@TransformRegistry.register_plugin("outlier_removal")
class OutlierRemovalTransformation(TransformPlugin):
    """Removal or limitation of emissions based on the Z-score"""

    def __init__(self, threshold: float = 3.0, method: str = "clip") -> None:
        super().__init__()
        self.threshold = threshold
        self.method = method  # 'clip' or 'remove'
        self.numeric_columns: list[str] = []
        self.means: dict[str, float] = {}
        self.stds: dict[str, float] = {}

    def _fit(self, df: pd.DataFrame) -> None:
        X = df.drop(columns=[self.target_column], errors="ignore")
        self.numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()

        for col in self.numeric_columns:
            self.means[col] = X[col].mean()
            self.stds[col] = X[col].std()

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result_df = df.copy()

        for col in self.numeric_columns:
            if col not in df.columns:
                continue

            if self.method == "clip":
                lower_bound = self.means[col] - self.threshold * self.stds[col]
                upper_bound = self.means[col] + self.threshold * self.stds[col]
                result_df[col] = result_df[col].clip(lower_bound, upper_bound)
            elif self.method == "remove":
                z_scores = np.abs(stats.zscore(result_df[col]))
                result_df = result_df[z_scores <= self.threshold]

        return result_df


@TransformRegistry.register_plugin("fast_ica")
class FastICATransformation(TransformPlugin):
    """Using the FastICA algorithm to isolate independent components"""

    def __init__(self, n_components: Any = None, random_state: int = 42) -> None:
        super().__init__()
        self.ica = FastICA(n_components=n_components, random_state=random_state)

    def _fit(self, df: pd.DataFrame) -> None:
        X = df.drop(columns=[self.target_column], errors="ignore")
        self.numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()
        self.ica.fit(X[self.numeric_columns])

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result_df = df.copy()
        num_cols = [col for col in self.numeric_columns if col in df.columns]

        if not num_cols:
            return result_df

        ica_features = self.ica.transform(df[num_cols])
        n_components = ica_features.shape[1]
        ica_df = pd.DataFrame(ica_features, columns=[f"ica_{i}" for i in range(n_components)], index=df.index)

        result_df = result_df.drop(columns=num_cols)
        result_df = pd.concat([result_df, ica_df], axis=1)

        return result_df


@TransformRegistry.register_plugin("select_k_best")
class SelectKBestTransformation(TransformPlugin):
    """Selecting the K best features based on statistical tests"""

    def __init__(self, k: int = 10, score_func: Any = f_classif) -> None:
        super().__init__()
        self.selector = SelectKBest(score_func=score_func, k=k)
        self.target_column = "tirads"

    def _fit(self, df: pd.DataFrame) -> None:
        X = df.drop(columns=[self.target_column], errors="ignore")
        y = df[self.target_column]
        self.numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()
        self.selector.fit(X[self.numeric_columns], y)
        self.selected_indices = self.selector.get_support()
        self.selected_features = [col for selected, col in zip(self.selected_indices, self.numeric_columns) if selected]

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result_df = df.copy()
        available_features = [col for col in self.selected_features if col in df.columns]

        if self.target_column in df.columns:
            return result_df[[*available_features, self.target_column]]
        else:
            return result_df[available_features]


@TransformRegistry.register_plugin("kmeans_clustering_features")
class KMeansClusteringFeatures(TransformPlugin):
    """Creating features based on KMeans clusterization"""

    def __init__(self, n_clusters: int = 5, random_state: int = 42) -> None:
        super().__init__()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.target_column = "tirads"

    def _fit(self, df: pd.DataFrame) -> None:
        X = df.drop(columns=[self.target_column], errors="ignore")
        self.numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()
        self.kmeans.fit(X[self.numeric_columns])

        self.cluster_centers = self.kmeans.cluster_centers_

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result_df = df.copy()
        num_cols = [col for col in self.numeric_columns if col in df.columns]

        if not num_cols:
            return result_df

        cluster_labels = self.kmeans.predict(df[num_cols])
        result_df["kmeans_cluster"] = cluster_labels

        X_array = df[num_cols].values
        n_clusters = self.cluster_centers.shape[0]

        for i in range(n_clusters):
            distances = np.sqrt(((X_array - self.cluster_centers[i]) ** 2).sum(axis=1))
            result_df[f"dist_to_cluster_{i}"] = distances

        return result_df
