import abc
import copy
import random
from abc import ABC
from typing import Any, Callable, Dict, Optional, Tuple, Type

import numpy as np
import optuna
import torch
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split

from src.training_module import root_loger
from src.training_module.data_model_bridge.data_adapters import BaseDataAdapter, PytorchDataAdapter, SklearnDataAdapter
from src.training_module.model_core.base_models import BaseTuner, PyTorchModel, SklearnModel


class TunerFactory:
    _registry: dict = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Type], Type]:
        def decorator(tunner_class: Type) -> Type:
            if name in cls._registry:
                raise ValueError(f"Tuner '{name}' already registered")
            cls._registry[name] = tunner_class
            return tunner_class

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> BaseTuner:
        if name not in cls._registry:
            raise ValueError(f"Tuner {name} is not registered")
        return cls._registry[name](**kwargs)


class OptunaTuner(BaseTuner, ABC):
    """Base class for Optuna-based tuners."""

    @staticmethod
    def _suggest_param(trial: optuna.Trial, config: Dict[str, Any]) -> Any:
        param_type = config.get("type")

        if param_type == "float":
            return trial.suggest_float(config["name"], config["low"], config["high"], log=config.get("log", False))
        elif param_type == "int":
            return trial.suggest_int(config["name"], config["low"], config["high"], log=config.get("log", False))
        elif param_type == "categorical":
            return trial.suggest_categorical(config["name"], config["options"])
        raise ValueError(f"Unknown parameter type '{param_type}'")


@TunerFactory.register("sklearn_optuna")
class SklearnOptunaTuner(OptunaTuner):
    """Optuna-based hyperparameter tuner for scikit-learn models."""

    def __init__(
        self,
        scoring: str = "f1",
        direction: str = "maximize",
        n_trials: int = 100,
        cv: int = 3,
        random_state: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(scoring)
        self.n_trials = n_trials
        self.direction = direction
        self.cv = cv
        self.random_state = random_state
        self.kwargs = kwargs

    def tune(
        self, model: SklearnModel, data_adapter: BaseDataAdapter, param_space: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any]]:
        X, y = data_adapter.data
        sampler = optuna.samplers.TPESampler(seed=self.random_state) if self.random_state else None
        study = optuna.create_study(direction=self.direction, sampler=sampler)

        def objective(trial: optuna.Trial) -> float:
            params = {}
            for param_name, param_config in param_space.items():
                param_config = param_config.copy()
                param_config["name"] = param_name
                params[param_name] = self._suggest_param(trial, param_config)

            cloned_model = model.get_clone_model(params)

            cv_scores = cross_val_score(cloned_model, X, y, cv=self.cv, scoring=self.scoring, n_jobs=-1)

            return cv_scores.mean()

        study.optimize(objective, n_trials=self.n_trials, **self.kwargs)

        return study.best_value, study.best_params


@TunerFactory.register("sklearn_gridsearch")
class SklearnGridSearchTuner(BaseTuner):
    """Grid search hyperparameter tuner for scikit-learn models."""

    def __init__(self, cv: int = 3, scoring: str = "f1", n_jobs: int = -1) -> None:
        super().__init__(scoring)
        self.cv: int = cv
        self.n_jobs: int = n_jobs

    def tune(self, model: Any, data: Any, param_space: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        model = model.model
        searcher = GridSearchCV(model, param_space, scoring=self.scoring, cv=self.cv, n_jobs=self.n_jobs, verbose=1)
        X, y = data.data
        searcher.fit(X, y)

        return searcher.best_score_, searcher.best_params_


class DataSplitterMixin:
    """Mixin class that adds functionality to split a data adapter into train and validation parts."""

    def split_data_adapter(
        self, data_adapter: BaseDataAdapter, val_size: float = 0.2, random_state: Optional[int] = None
    ) -> Tuple[BaseDataAdapter, BaseDataAdapter]:

        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)
            torch.manual_seed(random_state)

        if isinstance(data_adapter, PytorchDataAdapter):
            return self._split_pytorch_adapter(data_adapter, val_size, random_state)
        else:
            raise TypeError(f"Unsupported data adapter type: {type(data_adapter)}")

    @staticmethod
    def _split_pytorch_adapter(
        data_adapter: PytorchDataAdapter, val_size: float, random_state: Optional[int]
    ) -> Tuple[PytorchDataAdapter, PytorchDataAdapter]:

        if not hasattr(data_adapter, "_preprocessed") or not data_adapter._preprocessed:
            data_adapter.prepare()

        images = data_adapter.images

        n_samples = len(images)
        n_val = int(n_samples * val_size)
        n_train = n_samples - n_val

        indices = list(range(n_samples))
        if random_state is not None:
            random.Random(random_state).shuffle(indices)
        else:
            random.shuffle(indices)

        train_indices, val_indices = indices[:n_train], indices[n_train:]

        adapter_class = data_adapter.__class__

        train_adapter = adapter_class(
            images=[images[i] for i in train_indices],
            is_bin_classification=data_adapter._is_bin_classification,
            batch_size=data_adapter.batch_size,
            label=data_adapter.label,
        )

        val_adapter = adapter_class(
            images=[images[i] for i in val_indices],
            is_bin_classification=data_adapter._is_bin_classification,
            batch_size=data_adapter.batch_size,
            label=data_adapter.label,
        )

        train_adapter.transforms = copy.deepcopy(data_adapter.transforms)
        val_adapter.transforms = copy.deepcopy(data_adapter.transforms)

        train_adapter.prepare()
        val_adapter.prepare()

        return train_adapter, val_adapter


@TunerFactory.register("pytorch_optuna")
class AutoSplitPyTorchTuner(OptunaTuner, DataSplitterMixin):
    """
    Optuna-based hyperparameter tuner for PyTorch models that automatically splits
    the provided data into training and validation sets.
    """

    def __init__(
        self,
        scoring: str = "f1",
        n_trials: int = 50,
        direction: str = "maximize",
        val_size: float = 0.2,
        random_state: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(scoring)
        self.n_trials = n_trials
        self.direction = direction
        self.val_size = val_size
        self.random_state = random_state
        self.kwargs = kwargs

    def tune(
        self, model: PyTorchModel, data_adapter: BaseDataAdapter, param_space: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any]]:

        train_adapter, val_adapter = self.split_data_adapter(
            data_adapter, val_size=self.val_size, random_state=self.random_state
        )

        root_loger.info(
            f"Auto-split data: {len(train_adapter.images)} training samples, "
            f"{len(val_adapter.images)} validation samples"
        )

        sampler = optuna.samplers.TPESampler(seed=self.random_state) if self.random_state else None
        study = optuna.create_study(direction=self.direction, sampler=sampler)

        def objective(trial: optuna.Trial) -> float:
            params: dict[str, dict[str, Any]] = {}

            for param_name, param_config in param_space.items():
                param_config = param_config.copy()
                if "name" not in param_config:
                    param_config["name"] = param_name

                suggested_value = self._suggest_param(trial, param_config)

                if "." in param_name:
                    parent_key, child_key = param_name.split(".", 1)
                    if parent_key not in params:
                        params[parent_key] = {}
                    params[parent_key][child_key] = suggested_value
                else:
                    params[param_name] = suggested_value
            cloned_model = model.get_clone_model(params)

            cloned_model.fit(train_adapter, val_adapter)

            eval_score = cloned_model.evaluate(val_adapter)

            if model.is_binary:
                score = eval_score.get(self.scoring, eval_score.get("f1", 0.0))
            else:
                score = eval_score.get(f"{self.scoring}_macro", eval_score.get("f1_macro", 0.0))

            return score

        study.optimize(objective, n_trials=self.n_trials, n_jobs=3, **self.kwargs)

        root_loger.info(f"Best trial: Value={study.best_value}, Params={study.best_params}")

        return study.best_value, study.best_params
