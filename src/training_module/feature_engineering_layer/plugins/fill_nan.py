from typing import Any, Optional

import numpy as np
import pandas as pd

from src.training_module.feature_engineering_layer.plugin_registry import TransformRegistry
from src.training_module.feature_engineering_layer.plugins.transform_plugin import TransformPlugin


@TransformRegistry.register_plugin("fill_nan")
class FillNan(TransformPlugin):
    """Filling in the nan"""

    _strategies = ["mean", "median", "mode", "zero", "constant"]

    def __init__(self, strategy:str ="mean", fill_value: Optional[float]=None) -> None:
        super().__init__()
        if strategy not in self._strategies:
            raise ValueError(f"Strategy '{strategy}' is not supported. Choose from {self._strategies}")
        self.strategy = strategy
        self.fill_value = fill_value
        self.stats: dict[str, float] = {}
        self.numeric_columns: list[str] = []

    def _fit(self, df: pd.DataFrame) -> None:
        X = df.drop(columns=[self.target_column], errors="ignore")
        self.numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()

        for col in self.numeric_columns:
            if self.strategy == "mean":
                self.stats[col] = X[col].mean()
            elif self.strategy == "median":
                self.stats[col] = X[col].median()
            elif self.strategy == "mode":
                self.stats[col] = X[col].mode().iloc[0] if not X[col].mode().empty else np.nan
            elif self.strategy == "zero":
                self.stats[col] = 0
            elif self.strategy == "constant":
                if self.fill_value is None:
                    raise ValueError("When choosing a 'constant' strategy, fill_value must be specified")
                self.stats[col] = self.fill_value

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result_df = df.copy()

        cols_to_transform = [col for col in self.numeric_columns if col in df.columns]

        if not cols_to_transform:
            return result_df

        for col in cols_to_transform:
            result_df[col] = result_df[col].fillna(self.stats.get(col, np.nan))

        return result_df

    def _get_state(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "fill_value": self.fill_value,
            "stats": self.stats,
            "numeric_columns": self.numeric_columns,
        }

    def _set_state(self, state: dict[str, Any]) -> None:
        self.strategy = state["strategy"]
        self.fill_value = state["fill_value"]
        self.stats = state["stats"]
        self.numeric_columns = state["numeric_columns"]
