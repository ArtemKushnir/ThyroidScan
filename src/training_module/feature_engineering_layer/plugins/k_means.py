from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from src.training_module.feature_engineering_layer.plugin_registry import TransformRegistry
from src.training_module.feature_engineering_layer.plugins.transform_plugin import TransformPlugin


@TransformRegistry.register_plugin("k_means")
class KMeansClusteringFeatures(TransformPlugin):
    """Creating features based on KMeans clusterization"""

    def __init__(self, n_clusters: int = 5, random_state: int = 42) -> None:
        super().__init__()
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.numeric_columns: list[str] = []
        self.cluster_centers = None
        self.is_fitted = False

    def _fit(self, df: pd.DataFrame) -> None:
        X = df.drop(columns=[self.target_column], errors="ignore")
        self.numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()
        self.kmeans.fit(X[self.numeric_columns])
        self.cluster_centers = self.kmeans.cluster_centers_
        self.is_fitted = True

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result_df = df.copy()
        num_cols = [col for col in self.numeric_columns if col in df.columns]

        if not num_cols:
            return result_df

        if not self.is_fitted or self.cluster_centers is None:
            raise ValueError("Model not fitted properly. Cluster centers are missing.")

        cluster_labels = self._manual_predict(df[num_cols].values)
        result_df["kmeans_cluster"] = cluster_labels

        X_array = df[num_cols].values
        n_clusters = self.cluster_centers.shape[0]

        for i in range(n_clusters):
            distances = np.sqrt(((X_array - self.cluster_centers[i]) ** 2).sum(axis=1))
            result_df[f"dist_to_cluster_{i}"] = distances

        return result_df

    def _manual_predict(self, X: np.ndarray) -> np.ndarray:
        if self.cluster_centers is None:
            raise ValueError("Model not fitted properly. Cluster centers are missing.")
        distances = np.sqrt(((X - self.cluster_centers[:, np.newaxis]) ** 2).sum(axis=2))
        return np.argmin(distances, axis=0)

    def _get_state(self) -> dict[str, Any]:
        state = {
            "numeric_columns": self.numeric_columns,
            "cluster_centers": self.cluster_centers,
            "n_clusters": self.n_clusters,
            "random_state": self.random_state,
            "is_fitted": self.is_fitted,
        }
        return state

    def _set_state(self, state: dict[str, Any]) -> None:
        self.numeric_columns = state["numeric_columns"]
        self.cluster_centers = state["cluster_centers"]
        self.n_clusters = state.get("n_clusters", 5)
        self.random_state = state.get("random_state", 42)
        self.is_fitted = state.get("is_fitted", False)

        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)

        if self.is_fitted and self.cluster_centers is not None:
            self.kmeans.cluster_centers_ = self.cluster_centers
            self.kmeans.n_features_in_ = self.cluster_centers.shape[1]
