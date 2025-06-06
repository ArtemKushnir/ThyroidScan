from src.training_module.model_core.tunner_module.hyperparameter_selection_system import (
    HyperparameterConfig,
    ModelHyperparameterSpace,
    TunerType,
)

svm_space = ModelHyperparameterSpace(
    model_name="svm",
    parameters={
        "C": HyperparameterConfig(type="float", low=0.1, high=100.0, log=True, values=[0.1, 1.0, 10.0, 100.0]),
        "kernel": HyperparameterConfig(
            type="categorical", options=["linear", "poly", "rbf", "sigmoid"], values=["linear", "rbf", "poly"]
        ),
        "gamma": HyperparameterConfig(type="categorical", options=["scale", "auto"], values=["scale", "auto"]),
        "degree": HyperparameterConfig(type="int", low=2, high=5, values=[2, 3, 4]),
        "probability": HyperparameterConfig(type="categorical", options=[True, False], values=[True]),
    },
    tuner_configs={
        TunerType.SKLEARN_GRIDSEARCH: {"cv": 3, "scoring": "f1", "n_jobs": -1},
        TunerType.SKLEARN_OPTUNA: {"n_trials": 500, "direction": "maximize", "scoring": "f1_macro", "cv": 5},
    },
)
