from src.training_module.model_core.tunner_module.hyperparameter_selection_system import (
    HyperparameterConfig,
    ModelHyperparameterSpace,
    TunerType,
)

se_resnet_space = ModelHyperparameterSpace(
    model_name="se_resnet_image",
    parameters={
        "batch_size": HyperparameterConfig(type="categorical", options=[8, 16, 32, 64], values=[8, 16, 32, 64]),
        "epochs": HyperparameterConfig(type="int", low=10, high=100, values=[10, 20, 30, 50, 75, 100]),
        "weight_decay": HyperparameterConfig(
            type="float", low=1e-6, high=1e-2, log=True, values=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
        ),
        "optimizer.name": HyperparameterConfig(
            type="categorical", options=["adam", "adamw", "sgd"], values=["adam", "adamw", "sgd"]
        ),
        "optimizer.lr": HyperparameterConfig(
            type="float", low=1e-5, high=1e-2, log=True, values=[1e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 5e-3, 1e-2]
        ),
        "scheduler.type": HyperparameterConfig(
            type="categorical",
            options=["StepLR", "CosineAnnealingLR", "ReduceLROnPlateau", "ExponentialLR"],
            values=["StepLR", "CosineAnnealingLR", "ReduceLROnPlateau", "ExponentialLR"],
        ),
        "scheduler.step_size": HyperparameterConfig(type="int", low=5, high=30, values=[5, 10, 15, 20, 25, 30]),
        "scheduler.gamma": HyperparameterConfig(type="float", low=0.1, high=0.9, values=[0.1, 0.3, 0.5, 0.7, 0.9]),
        "dropout_rate": HyperparameterConfig(type="float", low=0.0, high=0.5, values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
        "criterion.name": HyperparameterConfig(
            type="categorical",
            options=["focal", "cross_entropy"],
            values=["focal", "cross_entropy"],
        ),
    },
    tuner_configs={
        TunerType.PYTORCH_OPTUNA: {
            "n_trials": 1,
            "direction": "maximize",
            "scoring": "f1",
            "val_size": 0.2,
            "random_state": 42,
        }
    },
)
