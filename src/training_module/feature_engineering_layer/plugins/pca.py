from typing import Any

import pandas as pd
from sklearn.decomposition import PCA

from src.training_module.feature_engineering_layer.plugin_registry import TransformRegistry
from src.training_module.feature_engineering_layer.plugins.transform_plugin import TransformPlugin


@TransformRegistry.register_plugin("pca")
class PCATransformation(TransformPlugin):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.pca = PCA(*args, **kwargs)
        self.columns = None

    def _fit(self, df: pd.DataFrame) -> None:
        X = df.drop(columns=[self.target_column], errors="ignore")
        self.columns = X.columns
        self.pca.fit(X)

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.columns is None:
            raise ValueError("Call 'fit' with appropriate arguments before using this transform plugin")
        features = [col for col in self.columns if col in df.columns]
        return pd.DataFrame(self.pca.transform(df[features]))

    def _get_state(self) -> dict[str, Any]:
        state = {
            "components_": self.pca.components_,
            "explained_variance_": self.pca.explained_variance_,
            "explained_variance_ratio_": self.pca.explained_variance_ratio_,
            "singular_values_": self.pca.singular_values_,
            "mean_": self.pca.mean_,
            "n_components_": self.pca.n_components_,
            "noise_variance_": self.pca.noise_variance_,
            "n_features_in_": self.pca.n_features_in_,
        }

        if hasattr(self.pca, "feature_names_in_"):
            state["feature_names_in_"] = self.pca.feature_names_in_
        else:
            state["feature_names_in_"] = None

        if self.columns is not None:
            state["columns"] = self.columns.tolist() if hasattr(self.columns, "tolist") else list(self.columns)
        else:
            state["columns"] = None

        return state

    def _set_state(self, state: dict[str, Any]) -> None:
        self.pca.components_ = state["components_"]
        self.pca.explained_variance_ = state["explained_variance_"]
        self.pca.explained_variance_ratio_ = state["explained_variance_ratio_"]
        self.pca.singular_values_ = state["singular_values_"]
        self.pca.mean_ = state["mean_"]
        self.pca.n_components_ = state["n_components_"]
        self.pca.noise_variance_ = state["noise_variance_"]
        self.pca.n_features_in_ = state["n_features_in_"]

        if state["feature_names_in_"] is not None:
            self.pca.feature_names_in_ = state["feature_names_in_"]

        if state["columns"] is not None:
            self.columns = pd.Index(state["columns"])
