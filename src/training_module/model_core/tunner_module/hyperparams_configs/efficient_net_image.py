from src.training_module.model_core.tunner_module.hyperparameter_selection_system import (
    HyperparameterConfig,
    ModelHyperparameterSpace,
    TunerType,
)

efficient_net_image_space = ModelHyperparameterSpace(
    model_name="efficient_net_image",
    parameters={
        "batch_size": HyperparameterConfig(type="categorical", options=[4, 8, 16, 32], values=[4, 8, 16, 32]),
        "epochs": HyperparameterConfig(type="int", low=15, high=120, values=[15, 25, 40, 60, 80, 100, 120]),
        "weight_decay": HyperparameterConfig(
            type="float", low=1e-6, high=1e-1, log=True, values=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1]
        ),
        "optim.name": HyperparameterConfig(
            type="categorical", options=["adam", "adamw", "sgd", "rmsprop"], values=["adam", "adamw", "sgd", "rmsprop"]
        ),
        "optim.lr": HyperparameterConfig(
            type="float",
            low=1e-6,
            high=1e-2,
            log=True,
            values=[1e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 5e-3, 1e-2],
        ),
        "model_name": HyperparameterConfig(
            type="categorical",
            options=["efficientnet-b2", "efficientnet-b3", "efficientnet-b4"],
            values=["efficientnet-b2", "efficientnet-b3", "efficientnet-b4"],
        ),
        "scheduler.type": HyperparameterConfig(
            type="categorical",
            options=[
                "CosineAnnealingLR",
                "CosineAnnealingWarmRestarts",
                "ReduceLROnPlateau",
                "StepLR",
                "ExponentialLR",
            ],
            values=["CosineAnnealingLR", "CosineAnnealingWarmRestarts", "ReduceLROnPlateau", "StepLR", "ExponentialLR"],
        ),
        "scheduler.step_size": HyperparameterConfig(type="int", low=5, high=25, values=[5, 8, 10, 15, 20, 25]),
        "scheduler.gamma": HyperparameterConfig(
            type="float", low=0.05, high=0.9, values=[0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
        ),
        "scheduler.T_max": HyperparameterConfig(type="int", low=10, high=50, values=[10, 20, 30, 40, 50]),
        "dropout_rate": HyperparameterConfig(
            type="float", low=0.0, high=0.6, values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        ),
        "criterion.name": HyperparameterConfig(type="categorical", options=["focal"], values=["focal"]),
        "criterion.focal_alpha": HyperparameterConfig(
            type="float", low=0.1, high=2.0, values=[0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
        ),
        "criterion.focal_gamma": HyperparameterConfig(
            type="float", low=0.5, high=3.0, values=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        ),
    },
    tuner_configs={
        TunerType.PYTORCH_OPTUNA: {
            "n_trials": 75,
            "direction": "maximize",
            "scoring": "f1",
            "val_size": 0.2,
            "random_state": 42,
            "timeout": 10800,
        }
    },
)
