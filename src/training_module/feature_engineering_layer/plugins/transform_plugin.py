import abc
import os
import pickle
from typing import Any

import pandas as pd


class TransformPlugin(abc.ABC):
    def __init__(self) -> None:
        self._is_fit = False
        self.target_column = "tirads"

    def fit(self, df: pd.DataFrame) -> None:
        self._fit(df)
        self._is_fit = True

    @abc.abstractmethod
    def _fit(self, df: pd.DataFrame) -> None:
        pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self._check_fit()
        df = self._transform(df)
        if self.target_column in df.columns:
            return df.drop(columns=[self.target_column])
        return df

    @abc.abstractmethod
    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def _check_fit(self) -> None:
        if not self._is_fit:
            raise ValueError("Call 'fit' with appropriate arguments before using this transform plugin")

    def save_state(self, path: str) -> None:
        """Save the plugin state to a file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.get_state(), f)

    def load_state(self, path: str) -> None:
        """Load the plugin state from a file."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.set_state(state)
        self._is_fit = True

    def get_state(self) -> dict[str, Any]:
        """Get the current state of the plugin as a serializable dictionary."""
        state = self._get_state()
        state["target_column"] = self.target_column
        return state

    def set_state(self, state: dict[str, Any]) -> None:
        """Set the plugin state from a dictionary."""
        self._set_state(state)
        if "target_column" in state:
            self.target_column = state["target_column"]

    @abc.abstractmethod
    def _get_state(self) -> dict[str, Any]:
        """Get the plugin-specific state."""
        pass

    @abc.abstractmethod
    def _set_state(self, state: dict[str, Any]) -> None:
        """Set the plugin-specific state."""
        pass
