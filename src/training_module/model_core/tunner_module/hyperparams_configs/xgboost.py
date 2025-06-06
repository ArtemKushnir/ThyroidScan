from src.training_module.model_core.tunner_module.hyperparameter_selection_system import (
    HyperparameterConfig,
    ModelHyperparameterSpace,
    TunerType,
)

xgboost_space = ModelHyperparameterSpace(
    model_name="xgboost",
    parameters={
        "n_estimators": HyperparameterConfig(type="int", low=100, high=1000, values=[100, 300, 500, 700, 1000]),
        "max_depth": HyperparameterConfig(type="int", low=3, high=10, values=[3, 5, 7, 10]),
        "learning_rate": HyperparameterConfig(
            type="float", low=0.01, high=0.3, log=True, values=[0.01, 0.05, 0.1, 0.2]
        ),
        "subsample": HyperparameterConfig(type="float", low=0.6, high=1.0, values=[0.6, 0.8, 1.0]),
        "colsample_bytree": HyperparameterConfig(type="float", low=0.6, high=1.0, values=[0.6, 0.8, 1.0]),
        "reg_lambda": HyperparameterConfig(type="float", low=0.0, high=10.0, values=[0.0, 1.0, 5.0, 10.0]),
    },
    tuner_configs={
        TunerType.SKLEARN_GRIDSEARCH: {"cv": 3, "scoring": "f1", "n_jobs": -1},
        TunerType.SKLEARN_OPTUNA: {
            "n_trials": 100,
            "direction": "maximize",
            "scoring": "f1",
            "cv": 5,
            "timeout": 3600,
        },
    },
)
