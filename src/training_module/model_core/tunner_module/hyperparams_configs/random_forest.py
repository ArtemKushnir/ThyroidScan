from src.training_module.model_core.tunner_module.hyperparameter_selection_system import (
    HyperparameterConfig,
    ModelHyperparameterSpace,
    TunerType,
)

random_forest_space = ModelHyperparameterSpace(
    model_name="random_forest",
    parameters={
        "n_estimators": HyperparameterConfig(type="int", low=50, high=500, values=[50, 100, 200, 300, 500]),
        "max_depth": HyperparameterConfig(type="int", low=3, high=20, values=[5, 10, 15, 20]),
        "min_samples_split": HyperparameterConfig(type="int", low=2, high=20, values=[2, 5, 10, 15]),
        "min_samples_leaf": HyperparameterConfig(type="int", low=1, high=10, values=[1, 2, 4, 8]),
        "max_features": HyperparameterConfig(type="categorical", options=["sqrt", "log2"], values=["sqrt", "log2"]),
        "bootstrap": HyperparameterConfig(type="categorical", options=[True, False], values=[True, False]),
    },
    tuner_configs={
        TunerType.SKLEARN_GRIDSEARCH: {"cv": 3, "scoring": "f1", "n_jobs": -1},
        TunerType.SKLEARN_OPTUNA: {"n_trials": 300, "direction": "maximize", "scoring": "f1", "cv": 3},
    },
)
