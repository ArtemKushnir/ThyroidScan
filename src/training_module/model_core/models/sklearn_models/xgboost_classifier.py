from typing import Any, Optional

from xgboost import XGBClassifier

from src.training_module.model_core.base_models import SklearnModel
from src.training_module.model_core.model_registry import ModelRegistry


@ModelRegistry.register("xgboost")
class XGBoostModel(SklearnModel):

    def __init__(self, model_params: Optional[dict[str, Any]] = None, is_binary: bool = True) -> None:
        super().__init__(model_params, is_binary)
        self.model = XGBClassifier(**self.model_params)
