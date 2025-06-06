from typing import Any, Optional

from lightgbm import LGBMClassifier

from src.training_module.model_core.base_models import SklearnModel
from src.training_module.model_core.model_registry import ModelRegistry


@ModelRegistry.register("lgb")
class LightGBMModel(SklearnModel):
    name = "lgb"

    def __init__(self, model_params: Optional[dict[str, Any]] = None, is_binary: bool = True) -> None:
        super().__init__(model_params, is_binary)
        if "verbose" not in self.model_params:
            self.model_params["verbose"] = -1
        self.model = LGBMClassifier(**self.model_params)
