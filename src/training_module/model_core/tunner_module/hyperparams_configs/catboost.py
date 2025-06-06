from src.training_module.model_core.tunner_module.hyperparameter_selection_system import (
    HyperparameterConfig,
    ModelHyperparameterSpace,
    TunerType,
)

catboost_space = ModelHyperparameterSpace(
    model_name="catboost",
    parameters={
        "iterations": HyperparameterConfig(type="int", low=100, high=1000, values=[100, 300, 500, 700, 1000]),
        "depth": HyperparameterConfig(type="int", low=4, high=10, values=[4, 6, 8, 10]),
        "learning_rate": HyperparameterConfig(
            type="float", low=0.001, high=0.3, log=True, values=[0.001, 0.01, 0.03, 0.1, 0.2]
        ),
        "l2_leaf_reg": HyperparameterConfig(type="float", low=1.0, high=10.0, values=[1.0, 3.0, 5.0, 10.0]),
        "subsample": HyperparameterConfig(type="float", low=0.6, high=1.0, values=[0.6, 0.8, 1.0]),
        "random_strength": HyperparameterConfig(type="float", low=0.0, high=5.0, values=[0.0, 1.0, 2.0, 5.0]),
    },
    tuner_configs={
        TunerType.SKLEARN_GRIDSEARCH: {
            "cv": 5,
            "scoring": "f1",
            "n_jobs": -1,
        },
        TunerType.SKLEARN_OPTUNA: {
            "n_trials": 100,
            "direction": "maximize",
            "scoring": "f1",
            "cv": 5,
        },
    },
)
