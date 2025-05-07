from typing import Any, Optional

import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

from src.training_module.feature_engineering_layer.plugin_registry import TransformRegistry
from src.training_module.feature_engineering_layer.plugins.transform_plugin import TransformPlugin


@TransformRegistry.register_plugin("mutual_info")
class KHighestMutualInfo(TransformPlugin):
    def __init__(self, k: int = 100) -> None:
        super().__init__()
        self.k: int = k
        self.mutual_info: Optional[dict[str, float]] = None

    def _fit(self, df: pd.DataFrame) -> None:
        X = df.drop(columns=[self.target_column], errors="ignore")
        y = df[self.target_column]
        if not pd.api.types.is_numeric_dtype(y):
            y = LabelEncoder().fit_transform(y)
        mi_scores = mutual_info_classif(X, y, discrete_features=False)
        mi_series = pd.Series(mi_scores, index=X.columns)

        self.mutual_info = mi_series.sort_values(ascending=False).head(self.k).to_dict()

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.mutual_info is None:
            raise ValueError("Plugin not fitted. Call 'fit' before transform.")
        selected_columns = [col for col in self.mutual_info.keys() if col in df.columns]
        return df[selected_columns]

    def _get_state(self) -> dict[str, Any]:
        return {"mutual_info": self.mutual_info, "k": self.k}

    def _set_state(self, state: dict[str, Any]) -> None:
        self.mutual_info = state["mutual_info"]
        self.k = state["k"]
