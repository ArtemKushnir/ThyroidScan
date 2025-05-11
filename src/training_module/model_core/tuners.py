import abc
from abc import ABC
from typing import Any, Callable, Dict, Optional, Tuple, Type

import optuna
from sklearn.model_selection import GridSearchCV

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
        self, scoring: str = "f1", n_trials: int = 100, cv: int = 5, random_state: Optional[int] = None
    ) -> None:
        super().__init__(scoring)
        self.n_trials = n_trials
        self.cv = cv
        self.random_state = random_state

    def tune(self, model: SklearnModel, data: Any, param_space: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        sampler = optuna.samplers.TPESampler(seed=self.random_state) if self.random_state else None
        study = optuna.create_study(direction="maximize", sampler=sampler)

        def objective(trial: optuna.Trial) -> float:
            params = {}
            for param_name, param_config in param_space.items():
                param_config = param_config.copy()
                param_config["name"] = param_name
                params[param_name] = self._suggest_param(trial, param_config)

            cloned_model = model.get_clone_model(params)
            cloned_model.fit(data)
            eval_score = cloned_model.evaluate(data)

            return eval_score["f1"] if "f1" in eval_score else eval_score["f1_macro"]

        study.optimize(objective, n_trials=self.n_trials)

        return study.best_value, study.best_params


@TunerFactory.register("pytorch_optuna")
class PyTorchTuner(OptunaTuner):
    """Optuna-based hyperparameter tuner for PyTorch models."""

    def __init__(
        self, scoring: str = "f1", n_trials: int = 50, direction: str = "maximize", random_state: Optional[int] = None
    ) -> None:
        super().__init__(scoring)
        self.n_trials = n_trials
        self.direction = direction
        self.random_state = random_state

    def tune(self, model: PyTorchModel, data: Any, param_space: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        sampler = optuna.samplers.TPESampler(seed=self.random_state) if self.random_state else None
        study = optuna.create_study(direction=self.direction, sampler=sampler)

        def objective(trial: optuna.Trial) -> float:
            params = {}
            for param_name, param_config in param_space.items():
                param_config = param_config.copy()
                param_config["name"] = param_name
                params[param_name] = self._suggest_param(trial, param_config)

            cloned_model = model.get_clone_model(params)

            cloned_model.fit(data)
            eval_score = cloned_model.evaluate(data)

            return eval_score["f1"] if "f1" in eval_score else eval_score["f1_macro"]

        study.optimize(objective, n_trials=self.n_trials)

        return study.best_value, study.best_params


@TunerFactory.register("sklearn_gridsearch")
class SklearnGridSearchTuner(BaseTuner):
    """Grid search hyperparameter tuner for scikit-learn models."""

    def __init__(self, cv: int = 3, scoring: str = "f1", n_jobs: int = -1) -> None:
        super().__init__(scoring)
        self.cv: int = cv
        self.n_jobs: int = n_jobs

    def tune(self, model: Any, data: Any, param_space: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        searcher = GridSearchCV(model, param_space, scoring=self.scoring, cv=self.cv, n_jobs=self.n_jobs)
        X, y = data.data
        searcher.fit(X, y)

        return searcher.best_score_, searcher.best_params_
