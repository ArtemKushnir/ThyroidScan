from typing import Any

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif

from src.training_module.feature_engineering_layer.plugin_registry import TransformRegistry
from src.training_module.feature_engineering_layer.plugins.transform_plugin import TransformPlugin


@TransformRegistry.register_plugin("select_k_best")
class SelectKBestTransformation(TransformPlugin):
    """Selecting the K best features based on statistical tests"""

    def __init__(self, k: int = 10, score_func: Any = f_classif) -> None:
        super().__init__()
        self.selector = SelectKBest(score_func=score_func, k=k)
        self.numeric_columns: list[str] = []
        self.selected_indices = None
        self.selected_features: list[str] = []

    def _fit(self, df: pd.DataFrame) -> None:
        X = df.drop(columns=[self.target_column], errors="ignore")
        y = df[self.target_column]
        self.numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()
        self.selector.fit(X[self.numeric_columns], y)
        self.selected_indices = self.selector.get_support()
        self.selected_features = [col for selected, col in zip(self.selected_indices, self.numeric_columns) if selected]  # type: ignore

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result_df = df.copy()
        available_features = [col for col in self.selected_features if col in df.columns]

        if self.target_column in df.columns:
            return result_df[[*available_features, self.target_column]]
        else:
            return result_df[available_features]

    def _get_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "numeric_columns": self.numeric_columns,
            "selected_features": self.selected_features,
        }

        if hasattr(self.selector, "scores_"):
            state["scores_"] = self.selector.scores_
        else:
            state["scores_"] = None

        if hasattr(self.selector, "pvalues_"):
            state["pvalues_"] = self.selector.pvalues_
        else:
            state["pvalues_"] = None

        if self.selected_indices is not None:
            state["selected_indices"] = (
                self.selected_indices.tolist()
                if hasattr(self.selected_indices, "tolist")
                else list(self.selected_indices)
            )
        else:
            state["selected_indices"] = None

        return state

    def _set_state(self, state: dict[str, Any]) -> None:
        self.numeric_columns = state["numeric_columns"]
        self.selected_features = state["selected_features"]

        if state["scores_"] is not None:
            self.selector.scores_ = state["scores_"]

        if state["pvalues_"] is not None:
            self.selector.pvalues_ = state["pvalues_"]

        if state["selected_indices"] is not None:
            self.selected_indices = np.array(state["selected_indices"])  # type: ignore
