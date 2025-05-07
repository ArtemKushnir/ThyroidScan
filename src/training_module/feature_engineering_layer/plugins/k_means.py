from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from src.training_module.feature_engineering_layer.plugin_registry import TransformRegistry
from src.training_module.feature_engineering_layer.plugins.transform_plugin import TransformPlugin


@TransformRegistry.register_plugin("kmeans_clustering_features")
class KMeansClusteringFeatures(TransformPlugin):
    """Creating features based on KMeans clusterization"""

    def __init__(self, n_clusters: int = 5, random_state: int = 42) -> None:
        super().__init__()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.numeric_columns: list[str] = []
        self.cluster_centers = None

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

        if self.cluster_centers is None:
            raise ValueError("Model not fitted properly. Cluster centers are missing.")

        cluster_labels = self.kmeans.predict(df[num_cols])
        result_df["kmeans_cluster"] = cluster_labels

        X_array = df[num_cols].values
        n_clusters = self.cluster_centers.shape[0]

        for i in range(n_clusters):
            distances = np.sqrt(((X_array - self.cluster_centers[i]) ** 2).sum(axis=1))
            result_df[f"dist_to_cluster_{i}"] = distances

        return result_df

    def _get_state(self) -> dict[str, Any]:
        state = {
            "numeric_columns": self.numeric_columns,
            "cluster_centers": self.cluster_centers,
            "n_features_in_": self.kmeans.n_features_in_,
        }

        if hasattr(self.kmeans, "cluster_centers_"):
            state["cluster_centers_"] = self.kmeans.cluster_centers_
        else:
            state["cluster_centers_"] = None

        if hasattr(self.kmeans, "labels_"):
            state["labels_"] = self.kmeans.labels_
        else:
            state["labels_"] = None

        if hasattr(self.kmeans, "inertia_"):
            state["inertia_"] = self.kmeans.inertia_
        else:
            state["inertia_"] = None

        if hasattr(self.kmeans, "n_iter_"):
            state["n_iter_"] = self.kmeans.n_iter_
        else:
            state["n_iter_"] = None

        if hasattr(self.kmeans, "feature_names_in_"):
            state["feature_names_in_"] = self.kmeans.feature_names_in_
        else:
            state["feature_names_in_"] = None

        return state

    def _set_state(self, state: dict[str, Any]) -> None:
        self.numeric_columns = state["numeric_columns"]
        self.cluster_centers = state["cluster_centers"]
        self.kmeans.n_features_in_ = state["n_features_in_"]

        if state["cluster_centers_"] is not None:
            self.kmeans.cluster_centers_ = state["cluster_centers_"]

        if state["labels_"] is not None:
            self.kmeans.labels_ = state["labels_"]

        if state["inertia_"] is not None:
            self.kmeans.inertia_ = state["inertia_"]

        if state["n_iter_"] is not None:
            self.kmeans.n_iter_ = state["n_iter_"]

        if state["feature_names_in_"] is not None:
            self.kmeans.feature_names_in_ = state["feature_names_in_"]
