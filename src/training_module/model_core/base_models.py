import abc
import copy
import os
import pickle
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from src.training_module.data_model_bridge.data_adapters import BaseDataAdapter
from src.training_module.model_core.models.torch_models.criterions import BINARY_CRITERION, MULTICLASS_CRITERION
from src.training_module.model_core.models.torch_models.optimizers import OPTIMIZER


class BaseTuner(abc.ABC):
    """Base abstract class for all hyperparameter tuners."""

    def __init__(self, scoring: str = "f1") -> None:
        self.scoring = scoring

    @abc.abstractmethod
    def tune(self, model: Any, data: Any, param_space: dict[str, Any]) -> tuple[float, dict[str, Any]]:
        pass


class BaseModel(abc.ABC):
    DEFAULT_BINARY_MODEL_DIR = "../pipeline_artifacts/model_artifacts/binary_classification"
    DEFAULT_MULTICLASS_MODEL_DIR = "../pipeline_artifacts/model_artifacts/multiclass_classification"
    name = ""

    def __init__(self, model_params: Optional[dict[str, Any]] = None, is_binary: bool = True):
        self.model_params: dict[str, Any] = model_params or {}
        self.model: Any = None
        self.is_fitted: bool = False
        self.is_binary: bool = is_binary

    @abc.abstractmethod
    def fit(self, train_adapter: BaseDataAdapter, test_adapter: Optional[BaseDataAdapter] = None) -> "BaseModel":
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

    @abc.abstractmethod
    def save(self, directory: str = None) -> None:
        pass

    @staticmethod
    @abc.abstractmethod
    def load(is_binary: bool, model_name: str, directory: str) -> "BaseModel":
        pass

    @staticmethod
    def get_model(is_binary: bool, model_name: str, directory: str = None) -> "BaseModel":
        from src.training_module.model_core.model_registry import ModelRegistry

        model_instance = ModelRegistry.get_model(model_name)
        if directory is None:
            directory = BaseModel.DEFAULT_BINARY_MODEL_DIR if is_binary else BaseModel.DEFAULT_MULTICLASS_MODEL_DIR
        if isinstance(model_instance, SklearnModel):
            if not os.path.isfile(f"{directory}/{model_name}.pkl"):
                return model_instance
            return SklearnModel.load(is_binary=is_binary, model_name=model_name, directory=directory)
        elif isinstance(model_instance, PyTorchModel):
            if not os.path.isfile(f"{directory}/{model_name}.pth"):
                return model_instance
            return PyTorchModel.load(is_binary=is_binary, model_name=model_name, directory=directory).get_clone_model(
                {}
            )
        raise ValueError("Unknown model type")

    def evaluate(self, data_adapter: BaseDataAdapter) -> dict[str, float]:
        test_data = data_adapter.data
        if isinstance(self, SklearnModel):
            X_test, y_test = test_data
            y_pred = self.predict(data_adapter)

            if data_adapter.is_bin_classification:
                metrics = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred),
                    "recall": recall_score(y_test, y_pred),
                    "f1": f1_score(y_test, y_pred),
                }
                if hasattr(self.model, "predict_proba"):
                    y_proba = self.predict_proba(data_adapter)
                    # metrics["roc_auc"] = roc_auc_score(y_test, y_proba[:, 1] if y_proba.ndim > 1 else y_proba)
            else:
                metrics = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision_macro": precision_score(y_test, y_pred, average="macro"),
                    "recall_macro": recall_score(y_test, y_pred, average="macro"),
                    "f1_macro": f1_score(y_test, y_pred, average="macro"),
                }

            return metrics

        else:
            return self._evaluate_pytorch(data_adapter)

    def _evaluate_pytorch(self, data_adapter: BaseDataAdapter) -> dict[str, float]:
        if not isinstance(self, PyTorchModel):
            raise TypeError("Method only applicable for PyTorch models")

        raise NotImplementedError("Should be implemented in PyTorch model classes")


class SklearnModel(BaseModel, BaseEstimator, ClassifierMixin, abc.ABC):
    _data_adapter_type = "sklearn"

    def fit(self, train_adapter: BaseDataAdapter, test_adapter: Optional[BaseDataAdapter] = None) -> "SklearnModel":
        X, y = train_adapter.data
        if self.model is None:
            raise ValueError("Model has not been initialized")

        self.model.fit(X, y)
        return self

    def predict(self, data_adapter: BaseDataAdapter) -> np.ndarray:
        X, y = data_adapter.data
        if self.model is None:
            raise ValueError("Model has not been initialized")
        y_pred = self.model.predict(X)
        return y_pred

    def predict_proba(self, data_adapter: BaseDataAdapter) -> np.ndarray:
        X, y = data_adapter.data
        if self.model is None:
            raise ValueError("Model has not been initialized")
        return self.model.predict_proba(X)

    def get_clone_model(self, params: dict[str, Any]) -> Any:
        if self.model is None:
            raise ValueError("Model has not been initialized")

        from sklearn.base import clone

        new_model = clone(self.model)

        if params:
            new_model.set_params(**params)

        return new_model

    def _update_params(self, params: dict[str, Any]) -> None:
        self.model.set_params(**params)

    def save(self, directory: str = None) -> None:
        if directory is None:
            directory = BaseModel.DEFAULT_BINARY_MODEL_DIR if self.is_binary else BaseModel.DEFAULT_MULTICLASS_MODEL_DIR
        config_path = os.path.join(directory, f"{type(self).name}.pkl")
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(is_binary: bool, model_name: str, directory: str) -> "BaseModel":
        config_path = os.path.join(directory, f"{model_name}.pkl")
        with open(config_path, "rb") as f:
            loaded_model = pickle.load(f)

        return loaded_model


class PyTorchModel(BaseModel, abc.ABC):

    def __init__(self, model_params: Optional[dict[str, Any]] = None, is_binary: bool = True):
        super().__init__(model_params, is_binary)

        if model_params is None:
            self.model_params = {
                "epoch": 10,
                "optim": {"name": "adam", "lr": 0.0001},
                "criterion": {"name": "focal"},
            }
        else:
            self.model_params = model_params

            if "optim" not in self.model_params:
                self.model_params["optim"] = {"name": "adam", "lr": 0.0001}
            elif "name" not in self.model_params["optim"]:
                self.model_params["optim"]["name"] = "adam"

            if "epoch" not in self.model_params:
                self.model_params["epoch"] = 10

            if "criterion" not in self.model_params:
                self.model_params["criterion"] = {"name": "focal"}

        optimizer_dict = self.model_params["optim"]
        criterion_dict = self.model_params["criterion"]
        optimizer_name = optimizer_dict["name"]
        criterion_name = criterion_dict["name"]
        if optimizer_name not in OPTIMIZER:
            raise ValueError(
                f"Optimizer {optimizer_name} not supported. Available optimizers: {list(OPTIMIZER.keys())}"
            )

        criterion_dict = BINARY_CRITERION if is_binary else MULTICLASS_CRITERION
        if criterion_name not in criterion_dict:
            raise ValueError(
                f"Criterion {criterion_name} not supported. Available criteria: {list(criterion_dict.keys())}"
            )

        self.optimizer_name = optimizer_name
        self.criterion_name = criterion_name
        self.optimizer = OPTIMIZER[self.optimizer_name]
        self.criterion = (
            BINARY_CRITERION[self.criterion_name] if is_binary else MULTICLASS_CRITERION[self.criterion_name]
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.history: dict[str, Any] = {"train_loss": [], "val_loss": [], "metrics": {}}

        self.preprocessing: list[Callable] = []

    @abc.abstractmethod
    def _create_model(self) -> nn.Module:
        pass

    def _initialize_preprocessing(self) -> list[Callable]:
        return []

    def _prepare_input(self, batch_data: Any) -> tuple[Any, torch.Tensor]:
        if isinstance(batch_data, dict):
            X = batch_data["pixels"].to(self.device)
            y = batch_data["label"].to(self.device) if "label" in batch_data else torch.tensor([])
        elif isinstance(batch_data, (list, tuple)) and len(batch_data) >= 2:
            X, y = batch_data[0].to(self.device), (
                batch_data[1].to(self.device) if batch_data[1] is not None else torch.tensor([])
            )
        else:
            raise ValueError(f"Unsupported batch data format: {type(batch_data)}")

        return X, y

    def _apply_preprocessing(self, X: torch.Tensor) -> torch.Tensor:
        for transform in self.preprocessing:
            X = transform(X)
        return X

    def fit(self, train_adapter: BaseDataAdapter, test_adapter: Optional[BaseDataAdapter] = None) -> "PyTorchModel":
        train_loader = train_adapter.data

        self.model = self.model.to(self.device)

        optim_params = dict(filter(lambda item: item[0] != "name", self.model_params["optim"].items()))
        optimizer = self.optimizer(self.model.parameters(), **optim_params)
        criterion = self.criterion()
        epochs = self.model_params.get("epoch", 10)
        for epoch in range(epochs):
            train_loss = self.train_loop(train_loader, self.model, criterion, optimizer)

            if test_adapter is not None:
                metrics = self.validate(test_adapter)
                val_loss = metrics["loss"]
            else:
                val_loss, metrics = 0.0, {}

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            for metric_name, metric_value in metrics.items():
                if metric_name not in self.history["metrics"]:
                    self.history["metrics"][metric_name] = []
                self.history["metrics"][metric_name].append(metric_value)

            # print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
            # for metric_name, metric_value in metrics.items():
            #     print(f" - {metric_name}: {metric_value:.4f}")

        return self

    def train_loop(
        self, dataloader: DataLoader, model: nn.Module, loss_fn: nn.Module, optimizer: torch.optim.Optimizer
    ) -> float:
        size = len(dataloader.dataset)  # type: ignore
        model.train()
        running_loss = 0.0

        for batch, batch_data in enumerate(dataloader):
            X, y = self._prepare_input(batch_data)
            optimizer.zero_grad()

            pred = model(X)

            if self.is_binary and pred.shape != y.shape:
                y = y.view(pred.shape)

            loss = loss_fn(pred, y)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        return running_loss / size

    def validate(self, data_adapter: BaseDataAdapter) -> dict[str, float]:
        return self._evaluate_pytorch(data_adapter)

    def predict_proba(self, data_adapter: BaseDataAdapter) -> np.ndarray:
        test_loader = data_adapter.data
        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for batch_data in test_loader:
                X, _ = self._prepare_input(batch_data)
                outputs = self.model(X)

                if self.is_binary:
                    probs = torch.sigmoid(outputs).cpu().numpy()
                else:
                    probs = torch.softmax(outputs, dim=1).cpu().numpy()

                all_preds.append(probs)

        return np.vstack(all_preds)

    def predict(self, data_adapter: BaseDataAdapter) -> np.ndarray:
        proba = self.predict_proba(data_adapter)

        if self.is_binary:
            return (proba > 0.5).astype(int)
        else:
            return np.argmax(proba, axis=1)

    def _evaluate_pytorch(self, data_adapter: BaseDataAdapter) -> dict[str, float]:
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_targets = []
        criterion = self.criterion()

        test_loader = data_adapter.data
        with torch.no_grad():
            for batch_data in test_loader:
                X, y = self._prepare_input(batch_data)
                outputs = self.model(X)

                if self.is_binary and outputs.shape != y.shape:
                    y_loss = y.view(outputs.shape)
                else:
                    y_loss = y

                loss = criterion(outputs, y_loss)
                running_loss += loss.item()

                if self.is_binary:
                    preds = torch.sigmoid(outputs) > 0.5
                else:
                    preds = torch.argmax(outputs, dim=1)

                all_preds.append(preds.cpu().numpy())
                all_targets.append(y.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        metrics = {"loss": running_loss / len(test_loader.dataset), "accuracy": accuracy_score(all_targets, all_preds)}

        if self.is_binary:
            metrics.update(
                {
                    "precision": precision_score(all_targets, all_preds, zero_division=0),
                    "recall": recall_score(all_targets, all_preds, zero_division=0),
                    "f1": f1_score(all_targets, all_preds, zero_division=0),
                }
            )

            try:
                proba = self.predict_proba(data_adapter)
                # if proba.shape[1] == 1:
                #     metrics["roc_auc"] = roc_auc_score(all_targets, proba)
                # else:
                #     metrics["roc_auc"] = roc_auc_score(all_targets, proba[:, 1])
            except:
                pass
        else:
            metrics.update(
                {
                    "precision_macro": precision_score(all_targets, all_preds, average="macro"),
                    "recall_macro": recall_score(all_targets, all_preds, average="macro"),
                    "f1_macro": f1_score(all_targets, all_preds, average="macro"),
                }
            )

        return metrics

    def get_clone_model(self, params: Optional[dict[str, Any]] = None) -> "PyTorchModel":
        clone = copy.deepcopy(self)

        clone.history = {"train_loss": [], "val_loss": [], "metrics": {}}

        if params is not None:
            clone._update_params(params)

        if clone.model is not None:
            for layer in clone.model.modules():
                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                    if hasattr(layer, "reset_parameters"):
                        layer.reset_parameters()

        return clone

    def _update_params(self, params: dict[str, Any]) -> None:
        """
        Update model parameters safely handling both nested and flat parameter structures.

        Args:
            params: Dictionary of parameters to update
        """
        for key, value in params.items():
            if key == "optim":
                if isinstance(value, dict):
                    if "optim" not in self.model_params:
                        self.model_params["optim"] = {}

                    for opt_key, opt_value in value.items():
                        self.model_params["optim"][opt_key] = opt_value
                else:
                    self.model_params["optim"]["name"] = value
            elif key == "criterion":
                criterion_name = value["name"]
                criterion_dict = BINARY_CRITERION if self.is_binary else MULTICLASS_CRITERION
                if criterion_name in criterion_dict:
                    self.criterion_name = criterion_name
                    self.criterion = criterion_dict[criterion_name]
                else:
                    raise ValueError(f"Criterion {criterion_name} not supported")
            else:
                self.model_params[key] = value

    def save(self, directory: str = None) -> None:
        state_dict = {
            "model_state_dict": self.model.state_dict(),
            "model_params": self.model_params,
            "optimizer_name": self.optimizer_name,
            "criterion_name": self.criterion_name,
            "is_binary": self.is_binary,
        }
        if directory is None:
            directory = BaseModel.DEFAULT_BINARY_MODEL_DIR if self.is_binary else BaseModel.DEFAULT_MULTICLASS_MODEL_DIR

        config_path = os.path.join(directory, f"{type(self).name}.pth")
        additional_state = self._get_additional_state()
        if additional_state:
            state_dict.update(additional_state)

        torch.save(state_dict, config_path)

    def _get_additional_state(self) -> dict[str, Any]:
        return {}

    @staticmethod
    def load(is_binary: bool, model_name: str, directory: str) -> "BaseModel":
        from src.training_module.model_core.model_registry import ModelRegistry

        model_instance = ModelRegistry.get_model(model_name)
        model_instance.is_binary = is_binary
        if isinstance(model_instance, PyTorchModel):
            config_path = os.path.join(directory, f"{model_name}.pth")
            checkpoint = torch.load(config_path, map_location=model_instance.device)

            model_instance.model_params = checkpoint["model_params"]
            model_instance.optimizer_name = checkpoint["optimizer_name"]
            model_instance.criterion_name = checkpoint["criterion_name"]
            model_instance.is_binary = checkpoint["is_binary"]

            model_instance.optimizer = OPTIMIZER[model_instance.optimizer_name]
            criterion_dict = BINARY_CRITERION if model_instance.is_binary else MULTICLASS_CRITERION
            model_instance.criterion = criterion_dict[model_instance.criterion_name]

            model_instance._load_additional_state(checkpoint)

            model_instance.model.load_state_dict(checkpoint["model_state_dict"])
            model_instance.model = model_instance.model.to(model_instance.device)

        return model_instance

    def _load_additional_state(self, checkpoint: dict[str, Any]) -> None:
        pass
