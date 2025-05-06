from typing import Any, Optional

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.training_module.model_core.base_models import SklearnModel


class SVM(SklearnModel):

    def __init__(self, model_params: Optional[dict[str, Any]] = None, is_binary: bool = True) -> None:
        super().__init__(model_params, is_binary)
        self.model = SVC(**self.model_params)


class RandomForest(SklearnModel):

    def __init__(self, model_params: Optional[dict[str, Any]] = None, is_binary: bool = True) -> None:
        super().__init__(model_params, is_binary)
        self.model = RandomForestClassifier(**self.model_params)


class CatBoostModel(SklearnModel):

    def __init__(self, model_params: Optional[dict[str, Any]] = None, is_binary: bool = True) -> None:
        super().__init__(model_params, is_binary)
        if "verbose" not in self.model_params:
            self.model_params["verbose"] = 0
        self.model = CatBoostClassifier(**self.model_params)


class XGBoostModel(SklearnModel):

    def __init__(self, model_params: Optional[dict[str, Any]] = None, is_binary: bool = True) -> None:
        super().__init__(model_params, is_binary)
        self.model = XGBClassifier(**self.model_params)


class LightGBMModel(SklearnModel):

    def __init__(self, model_params: Optional[dict[str, Any]] = None, is_binary: bool = True) -> None:
        super().__init__(model_params, is_binary)
        self.model = LGBMClassifier(**self.model_params)
