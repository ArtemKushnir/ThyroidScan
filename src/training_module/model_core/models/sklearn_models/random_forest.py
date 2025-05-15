from typing import Any, Optional

from sklearn.ensemble import RandomForestClassifier

from src.training_module.model_core.base_models import SklearnModel
from src.training_module.model_core.model_registry import ModelRegistry


@ModelRegistry.register("random_forest")
class RandomForest(SklearnModel):
    name = "random_forest"

    def __init__(self, model_params: Optional[dict[str, Any]] = None, is_binary: bool = True) -> None:
        super().__init__(model_params, is_binary)
        self.model = RandomForestClassifier(**self.model_params)
