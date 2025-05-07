from typing import Any

import pandas as pd
from sklearn.decomposition import FastICA

from src.training_module.feature_engineering_layer.plugin_registry import TransformRegistry
from src.training_module.feature_engineering_layer.plugins.transform_plugin import TransformPlugin


@TransformRegistry.register_plugin("fast_ica")
class FastICATransformation(TransformPlugin):
    """Using the FastICA algorithm to isolate independent components"""

    def __init__(self, n_components: Any = None, random_state: int = 42) -> None:
        super().__init__()
        self.ica = FastICA(n_components=n_components, random_state=random_state)
        self.numeric_columns: list[str] = []

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

    def _get_state(self) -> dict[str, Any]:
        state = {
            "n_features_in_": self.ica.n_features_in_,
            "numeric_columns": self.numeric_columns,
        }

        if hasattr(self.ica, "n_components_"):
            state["n_components_"] = self.ica.n_components_
        else:
            state["n_components_"] = None

        if hasattr(self.ica, "components_"):
            state["components_"] = self.ica.components_
        else:
            state["components_"] = None

        if hasattr(self.ica, "mixing_"):
            state["mixing_"] = self.ica.mixing_
        else:
            state["mixing_"] = None

        if hasattr(self.ica, "mean_"):
            state["mean_"] = self.ica.mean_
        else:
            state["mean_"] = None

        if hasattr(self.ica, "whitening_"):
            state["whitening_"] = self.ica.whitening_
        else:
            state["whitening_"] = None

        if hasattr(self.ica, "feature_names_in_"):
            state["feature_names_in_"] = self.ica.feature_names_in_
        else:
            state["feature_names_in_"] = None

        return state

    def _set_state(self, state: dict[str, Any]) -> None:
        self.numeric_columns = state["numeric_columns"]
        self.ica.n_features_in_ = state["n_features_in_"]

        if state["n_components_"] is not None:
            self.ica.n_components_ = state["n_components_"]

        if state["components_"] is not None:
            self.ica.components_ = state["components_"]

        if state["mixing_"] is not None:
            self.ica.mixing_ = state["mixing_"]

        if state["mean_"] is not None:
            self.ica.mean_ = state["mean_"]

        if state["whitening_"] is not None:
            self.ica.whitening_ = state["whitening_"]

        if state["feature_names_in_"] is not None:
            self.ica.feature_names_in_ = state["feature_names_in_"]
