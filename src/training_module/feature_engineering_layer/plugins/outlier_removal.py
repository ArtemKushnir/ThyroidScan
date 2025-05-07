from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.training_module.feature_engineering_layer.plugin_registry import TransformRegistry
from src.training_module.feature_engineering_layer.plugins.transform_plugin import TransformPlugin


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

    def _get_state(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "method": self.method,
            "numeric_columns": self.numeric_columns,
            "means": self.means,
            "stds": self.stds,
        }

    def _set_state(self, state: dict[str, Any]) -> None:
        self.threshold = state["threshold"]
        self.method = state["method"]
        self.numeric_columns = state["numeric_columns"]
        self.means = state["means"]
        self.stds = state["stds"]
