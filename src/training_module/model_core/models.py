import abc
import os
import pickle
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn import clone
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.svm import SVC
from torch.utils.data import DataLoader
from xgboost import XGBClassifier

from src.image_data.image_data import Image
from src.training_module import root_loger
from src.training_module.data_model_bridge.data_adapters import BaseDataAdapter, DataAdapterFactory
from src.training_module.model_core.tuners import BaseTuner


class BaseModel(abc.ABC):

    def __init__(self, model_params: Optional[dict[str, Any]] = None):
        self.model_params: dict[str, Any] = model_params or {}
        self.model: Any = None
        self.is_fitted: bool = False

    @abc.abstractmethod
    def fit(self, data_adapter: BaseDataAdapter) -> "BaseModel":
        pass

    def tune_hyperparameters(
        self, data_adapter: BaseDataAdapter, tuner: BaseTuner, param_space: dict[str, Any], is_update: bool = True
    ) -> dict[str, Any]:
        best_value, best_params = tuner.tune(self, data_adapter, param_space)
        if is_update:
            self._update_params(best_params)
        return {"best_value": best_value, "best_params": best_params}

    @abc.abstractmethod
    def predict(self, data_adapter: BaseDataAdapter) -> np.ndarray:
        pass

    @abc.abstractmethod
    def _update_params(self, params: dict[str, Any]) -> None:
        pass

    @abc.abstractmethod
    def predict_proba(self, data_adapter: BaseDataAdapter) -> np.ndarray:
        pass

    @abc.abstractmethod
    def get_clone_model(self, params: dict[str, Any]) -> Any:
        pass

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "BaseModel":
        with open(path, "rb") as f:
            return pickle.load(f)

    def evaluate(self, data_adapter: BaseDataAdapter) -> dict[str, float]:
        test_data = data_adapter.data
        if isinstance(self, SklearnModel):
            X_test, y_test = test_data
            y_pred = self.predict(X_test)
            y_proba = self.predict_proba(X_test)

            if len(np.unique(y_test)) == 2:
                metrics = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred),
                    "recall": recall_score(y_test, y_pred),
                    "f1": f1_score(y_test, y_pred),
                    "roc_auc": roc_auc_score(y_test, y_proba[:, 1] if y_proba.ndim > 1 else y_proba),
                }
            else:
                metrics = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision_macro": precision_score(y_test, y_pred, average="macro"),
                    "recall_macro": recall_score(y_test, y_pred, average="macro"),
                    "f1_macro": f1_score(y_test, y_pred, average="macro"),
                }

            return metrics

        else:
            return self._evaluate_pytorch(test_data)

    def _evaluate_pytorch(self, test_loader: DataLoader) -> dict[str, float]:
        if not isinstance(self, PyTorchModel):
            raise TypeError("Method only applicable for PyTorch models")

        raise NotImplementedError("Should be implemented in PyTorch model classes")


class SklearnModel(BaseModel, BaseEstimator, ClassifierMixin):

    def fit(self, data_adapter: BaseDataAdapter) -> "SklearnModel":
        X, y = data_adapter.data
        if self.model is None:
            raise ValueError("Model has not been initialized")

        self.model.fit(X, y)
        train_metrics = self.evaluate(data_adapter)
        log = "".join([f"{key}: {value}\n" for key, value in train_metrics.items()])
        root_loger.info(f"Training metrics\n{log}")
        self.evaluate(data_adapter)
        return self

    def predict(self, data_adapter: BaseDataAdapter) -> np.ndarray:
        X, y = data_adapter.data
        if self.model is None:
            raise ValueError("Model has not been initialized")
        y_pred = self.model.predict(X)
        if y is not None:
            pass
        return y_pred

    def predict_proba(self, data_adapter: BaseDataAdapter) -> np.ndarray:
        X, y = data_adapter.data
        if self.model is None:
            raise ValueError("Model has not been initialized")
        return self.model.predict(X)

    def get_clone_model(self, params: dict[str, Any]) -> Any:
        pass

    def _update_params(self, params: dict[str, Any]) -> None:
        self.model.set_params(**params)


class SVM(SklearnModel):

    def __init__(self, model_params: Optional[dict[str, Any]] = None) -> None:
        super().__init__(model_params)
        self.model = SVC(**self.model_params)


class RandomForest(SklearnModel):

    def __init__(self, model_params: Optional[dict[str, Any]] = None) -> None:
        super().__init__(model_params)
        self.model = RandomForestClassifier(**self.model_params)


class CatBoostModel(SklearnModel):

    def __init__(self, model_params: Optional[dict[str, Any]] = None) -> None:
        super().__init__(model_params)
        if "verbose" not in self.model_params:
            self.model_params["verbose"] = 0
        self.model = CatBoostClassifier(**self.model_params)


class XGBoostModel(SklearnModel):

    def __init__(self, model_params: Optional[dict[str, Any]] = None) -> None:
        super().__init__(model_params)
        self.model = XGBClassifier(**self.model_params)


class LightGBMModel(SklearnModel):

    def __init__(self, model_params: Optional[dict[str, Any]] = None) -> None:
        super().__init__(model_params)
        self.model = LGBMClassifier(**self.model_params)


class PyTorchModel(BaseModel):
    def fit(self, data_adapter: BaseDataAdapter) -> "PyTorchModel":
        return self

    def predict_proba(self, data_adapter: BaseDataAdapter) -> np.ndarray:
        return np.arange(5)

    def predict(self, data_adapter: BaseDataAdapter) -> np.ndarray:
        return np.arange(5)

    def _evaluate_pytorch(self, test_loader: DataLoader) -> dict[str, float]:
        return {"pass": -1}

    def get_clone_model(self, params: dict[str, Any]) -> Any:
        pass

    def _update_params(self, params: dict[str, Any]) -> None:
        pass
