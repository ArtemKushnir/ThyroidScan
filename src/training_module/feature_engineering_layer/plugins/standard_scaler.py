from typing import Any

import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.training_module.feature_engineering_layer.plugin_registry import TransformRegistry
from src.training_module.feature_engineering_layer.plugins.transform_plugin import TransformPlugin


@TransformRegistry.register_plugin("standard_scaler")
class StandardScalerTransformation(TransformPlugin):
    def __init__(self) -> None:
        super().__init__()
        self.scaler = StandardScaler()
        self.columns = None

    def _fit(self, df: pd.DataFrame) -> None:
        X = df.drop(columns=[self.target_column], errors="ignore")
        self.columns = X.columns
        self.scaler.fit(X)

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result_df = df.copy()
        if self.columns is None:
            raise ValueError("Call 'fit' with appropriate arguments before using this transform plugin")
        features = [col for col in self.columns if col in df.columns]
        result_df[features] = self.scaler.transform(df[features])

        return result_df

    def _get_state(self) -> dict[str, Any]:
        state = {
            "mean_": self.scaler.mean_,
            "var_": self.scaler.var_,
            "scale_": self.scaler.scale_,
            "n_features_in_": self.scaler.n_features_in_,
            "n_samples_seen_": self.scaler.n_samples_seen_,
        }

        if hasattr(self.scaler, "feature_names_in_"):
            state["feature_names_in_"] = self.scaler.feature_names_in_
        else:
            state["feature_names_in_"] = None

        if self.columns is not None:
            state["columns"] = self.columns.tolist() if hasattr(self.columns, "tolist") else list(self.columns)
        else:
            state["columns"] = None

        return state

    def _set_state(self, state: dict[str, Any]) -> None:
        self.scaler.mean_ = state["mean_"]
        self.scaler.var_ = state["var_"]
        self.scaler.scale_ = state["scale_"]
        self.scaler.n_features_in_ = state["n_features_in_"]
        self.scaler.n_samples_seen_ = state["n_samples_seen_"]

        if state["feature_names_in_"] is not None:
            self.scaler.feature_names_in_ = state["feature_names_in_"]

        if state["columns"] is not None:
            self.columns = pd.Index(state["columns"])
