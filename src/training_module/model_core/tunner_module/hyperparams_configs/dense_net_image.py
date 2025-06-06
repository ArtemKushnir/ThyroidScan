from src.training_module.model_core.tunner_module.hyperparameter_selection_system import (
    HyperparameterConfig,
    ModelHyperparameterSpace,
    TunerType,
)

dense_net_image_space = ModelHyperparameterSpace(
    model_name="dense_net_image",
    parameters={
        "epoch": HyperparameterConfig(type="int", low=10, high=50, values=[10, 20, 30, 40, 50]),
        "optim.lr": HyperparameterConfig(
            type="float", low=1e-5, high=1e-2, log=True, values=[1e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 5e-3, 1e-2]
        ),
        "optim.name": HyperparameterConfig(
            type="categorical", options=["adam", "adamw", "sgd"], values=["adam", "adamw", "sgd"]
        ),
        "criterion.name": HyperparameterConfig(type="categorical", options=["focal"], values=["focal"]),
        "batch_size": HyperparameterConfig(type="categorical", options=[8, 16, 32, 64], values=[8, 16, 32, 64]),
        "dropout_rate": HyperparameterConfig(type="float", low=0.0, high=0.5, values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
        "scheduler": HyperparameterConfig(
            type="categorical",
            options=["cosine", "onecycle", "plateau", None],
            values=["cosine", "onecycle", "plateau", None],
        ),
        "mixup_alpha": HyperparameterConfig(type="float", low=0.0, high=0.4, values=[0.0, 0.1, 0.2, 0.3, 0.4]),
        "grad_clip_norm": HyperparameterConfig(type="float", low=0.5, high=2.0, values=[0.5, 1.0, 1.5, 2.0]),
        "intensity_normalization.subtrahend": HyperparameterConfig(
            type="float", low=0.3, high=0.6, values=[0.3, 0.4, 0.485, 0.5, 0.6]
        ),
        "intensity_normalization.divisor": HyperparameterConfig(
            type="float", low=0.1, high=0.4, values=[0.1, 0.2, 0.229, 0.3, 0.4]
        ),
    },
    tuner_configs={
        TunerType.PYTORCH_OPTUNA: {
            "n_trials": 15,
            "direction": "maximize",
            "scoring": "f1",
            "val_size": 0.2,
            "random_state": 42,
            "timeout": 3600,
        },
    },
)
