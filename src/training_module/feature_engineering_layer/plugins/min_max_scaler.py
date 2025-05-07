from typing import Any

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.training_module.feature_engineering_layer.plugin_registry import TransformRegistry
from src.training_module.feature_engineering_layer.plugins.transform_plugin import TransformPlugin


@TransformRegistry.register_plugin("min_max_scaler")
class MinMaxScalerTransformation(TransformPlugin):
    def __init__(self, feature_range: Any = (0, 1)) -> None:
        super().__init__()
        self.scaler = MinMaxScaler(feature_range=feature_range)
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
            "data_min_": self.scaler.data_min_,
            "data_max_": self.scaler.data_max_,
            "data_range_": self.scaler.data_range_,
            "scale_": self.scaler.scale_,
            "min_": self.scaler.min_,
            "feature_range": self.scaler.feature_range,
            "n_features_in_": self.scaler.n_features_in_,
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
        self.scaler.data_min_ = state["data_min_"]
        self.scaler.data_max_ = state["data_max_"]
        self.scaler.data_range_ = state["data_range_"]
        self.scaler.scale_ = state["scale_"]
        self.scaler.min_ = state["min_"]
        self.scaler.feature_range = state["feature_range"]
        self.scaler.n_features_in_ = state["n_features_in_"]

        if state["feature_names_in_"] is not None:
            self.scaler.feature_names_in_ = state["feature_names_in_"]

        if state["columns"] is not None:
            self.columns = pd.Index(state["columns"])
