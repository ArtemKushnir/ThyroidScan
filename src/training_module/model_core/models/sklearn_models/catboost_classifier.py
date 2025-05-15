from typing import Any, Optional

from catboost import CatBoostClassifier

from src.training_module.model_core.base_models import SklearnModel
from src.training_module.model_core.model_registry import ModelRegistry


@ModelRegistry.register("catboost")
class CatBoostModel(SklearnModel):
    name = "catboost"

    def __init__(self, model_params: Optional[dict[str, Any]] = None, is_binary: bool = True) -> None:
        super().__init__(model_params, is_binary)
        if "verbose" not in self.model_params:
            self.model_params["verbose"] = 0
        self.model = CatBoostClassifier(**self.model_params)
