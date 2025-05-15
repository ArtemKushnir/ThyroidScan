from typing import Any, Optional

from sklearn.svm import SVC

from src.training_module.model_core.base_models import SklearnModel
from src.training_module.model_core.model_registry import ModelRegistry


@ModelRegistry.register("svm")
class SVM(SklearnModel):
    name = "svm"

    def __init__(self, model_params: Optional[dict[str, Any]] = None, is_binary: bool = True) -> None:
        super().__init__(model_params, is_binary)
        self.model = SVC(**self.model_params)
