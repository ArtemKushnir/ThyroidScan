import copy
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, Union

import yaml

from src.training_module.data_model_bridge.data_adapters import BaseDataAdapter
from src.training_module.model_core.model_registry import ModelRegistry
from src.training_module.model_core.tunner_module.tuners import TunerFactory


class TunerType(Enum):
    SKLEARN_GRIDSEARCH = "sklearn_gridsearch"
    SKLEARN_OPTUNA = "sklearn_optuna"
    PYTORCH_OPTUNA = "pytorch_optuna"


@dataclass
class HyperparameterConfig:
    type: Optional[str] = None
    low: Optional[Union[int, float]] = None
    high: Optional[Union[int, float]] = None
    options: Optional[list[Any]] = None
    log: bool = False

    values: Optional[list[Any]] = None

    def to_optuna_param_space(self) -> dict[str, Any]:
        param_space: dict[str, Any] = {}
        if self.type:
            param_space["type"] = self.type
        if self.low is not None:
            param_space["low"] = self.low
        if self.high is not None:
            param_space["high"] = self.high
        if self.options:
            param_space["options"] = self.options
        if self.log:
            param_space["log"] = self.log

        return param_space

    def to_gridsearch_param_grid(self) -> list[Any]:
        if self.values:
            return self.values

        if self.type == "categorical":
            return self.options if self.options is not None else []
        elif self.type in ["float", "int"] and self.log is not None and self.high is not None:
            if self.low is None or self.high is None:
                return []
            if self.type == "int":
                step = max(1, (self.high - self.low) // 5)
                return list(range(int(self.low), int(self.high + 1), int(step)))
            else:
                step = (self.high - self.low) / 5
                values = []
                current = self.low
                while current <= self.high:
                    values.append(round(current, 6))
                    current += step
                return values

        return []


@dataclass
class ModelHyperparameterSpace:
    model_name: str
    parameters: dict[str, HyperparameterConfig] = field(default_factory=dict)
    tuner_configs: dict[TunerType, Any] = field(default_factory=dict)

    def get_param_space_for_tuner(self, tuner_type: TunerType) -> dict[str, Any]:
        if tuner_type == TunerType.SKLEARN_GRIDSEARCH:
            return self._get_gridsearch_space()
        else:
            return self._get_optuna_space()

    def _get_gridsearch_space(self) -> dict[str, Any]:
        param_space = {}

        for param_name, value in self.parameters.items():
            param_space[param_name] = value.to_gridsearch_param_grid()
        return param_space

    def _get_optuna_space(self) -> dict[str, Any]:
        param_space = {}

        for param_name, value in self.parameters.items():
            param_space[param_name] = value.to_optuna_param_space()
        return param_space

    def get_tuner_kwargs(self, tuner_type: TunerType) -> dict[str, Any]:
        return self.tuner_configs.get(tuner_type, {})


class HyperParameterManger:

    def __init__(self, config_dir: Union[str, Path] = "../../config/hyperparameters"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self._model_spaces: dict[str, ModelHyperparameterSpace] = {}
        self._load_all_configs()

    def _load_all_configs(self) -> None:
        for config_file in self.config_dir.glob("*.yaml"):
            try:
                self._load_config(config_file)
            except Exception as e:
                print(f"Download error {config_file}: {e}")

    def _load_config(self, config_path: Path) -> None:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        model_name = config["model_name"]

        parameters = {}
        for param_name, param_config in config.get("parameters", {}).items():
            parameters[param_name] = HyperparameterConfig(**param_config)

        tuner_configs = {}
        for tuner_name, tuner_config in config.get("tuner_configs", {}).items():
            try:
                tuner_type = TunerType(tuner_name)
                tuner_configs[tuner_type] = tuner_config
            except ValueError:
                print(f"Unknown tuner type: {tuner_name}")

        self._model_spaces[model_name] = ModelHyperparameterSpace(
            model_name=model_name, parameters=parameters, tuner_configs=tuner_configs
        )

    def register_model_space(self, model_space: ModelHyperparameterSpace) -> None:
        self._model_spaces[model_space.model_name] = model_space
        self._save_config(model_space)

    def _save_config(self, model_space: ModelHyperparameterSpace) -> None:
        config: dict[str, Any] = {"model_name": model_space.model_name, "parameters": {}, "tuner_configs": {}}

        for param_name, param_config in model_space.parameters.items():
            config["parameters"][param_name] = {
                "type": param_config.type,
                "low": param_config.low,
                "high": param_config.high,
                "options": param_config.options,
                "log": param_config.log,
                "values": param_config.values,
            }
            config["parameters"][param_name] = {
                k: v for k, v in config["parameters"][param_name].items() if v is not None
            }

        for tuner_type, tuner_config in model_space.tuner_configs.items():
            config["tuner_configs"][tuner_type.value] = tuner_config

        config_path = self.config_dir / f"{model_space.model_name}.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    def get_param_space(self, model_name: str, tuner_type: TunerType) -> dict[str, Any]:
        if model_name not in self._model_spaces:
            raise ValueError(f"Model {model_name} not found in configurations")

        model_space = self._model_spaces[model_name]
        return model_space.get_param_space_for_tuner(tuner_type)

    def get_tuner_kwargs(self, model_name: str, tuner_type: TunerType) -> dict[str, Any]:
        if model_name not in self._model_spaces:
            return {}

        model_space = self._model_spaces[model_name]
        return model_space.get_tuner_kwargs(tuner_type)

    def list_models(self) -> list[str]:
        return list(self._model_spaces.keys())

    def get_model_space(self, model_name: str) -> Optional[ModelHyperparameterSpace]:
        return self._model_spaces.get(model_name)


class TuningOrchestrator:
    def __init__(self, hyperparameter_manager: HyperParameterManger):
        self.hp_manager = hyperparameter_manager

    def tune_model(
        self,
        model_name: str,
        tuner_name: str,
        data_adapter: BaseDataAdapter,
        is_update: bool = True,
        override_params: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:

        tuner_type_map = {
            "sklearn_gridsearch": TunerType.SKLEARN_GRIDSEARCH,
            "sklearn_optuna": TunerType.SKLEARN_OPTUNA,
            "pytorch_optuna": TunerType.PYTORCH_OPTUNA,
        }

        if tuner_name not in tuner_type_map:
            raise ValueError(f"Unsupportable tuner: {tuner_name}")

        tuner_type = tuner_type_map[tuner_name]

        param_space = self.hp_manager.get_param_space(model_name, tuner_type)

        if override_params:
            param_space = self._merge_param_spaces(param_space, override_params)

        tuner_kwargs = self.hp_manager.get_tuner_kwargs(model_name, tuner_type)

        tuner = TunerFactory.create(tuner_name, **tuner_kwargs)

        model_instance = ModelRegistry.get_model(model_name)
        result = model_instance.tune_hyperparameters(
            data_adapter=data_adapter, tuner=tuner, param_space=param_space, is_update=is_update
        )

        return result

    @staticmethod
    def _merge_param_spaces(base_space: dict[str, Any], override_space: dict[str, Any]) -> dict[str, Any]:
        merged = copy.deepcopy(base_space)
        merged.update(override_space)
        return merged

    def compare_tuners(
        self, model_name: str, tuner_names: list[str], data_adapter: BaseDataAdapter, n_runs: int = 1
    ) -> dict[str, dict[str, Any]]:
        results: dict[str, Any] = {}

        for tuner_name in tuner_names:
            tuner_results = []

            for run in range(n_runs):
                print(f"Start {tuner_name}, trial {run + 1}/{n_runs}")

                try:
                    result = self.tune_model(model_name=model_name, tuner_name=tuner_name, data_adapter=data_adapter)
                    tuner_results.append(result)

                except Exception as e:
                    print(f"Error in {tuner_name}, trial {run + 1}: {e}")
                    tuner_results.append({"error": str(e)})

            results[tuner_name] = tuner_results

        return results
