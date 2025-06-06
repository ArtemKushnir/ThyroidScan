from src.training_module.model_core.tunner_module.hyperparameter_selection_system import (
    HyperparameterConfig,
    ModelHyperparameterSpace,
    TunerType,
)

swin_image_space = ModelHyperparameterSpace(
    model_name="swin_image",
    parameters={
        "epoch": HyperparameterConfig(type="int", low=10, high=60, values=[10, 20, 30, 40, 50, 60]),
        "optim.lr": HyperparameterConfig(
            type="float", low=1e-5, high=5e-3, log=True, values=[1e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3]
        ),
        "optim.name": HyperparameterConfig(
            type="categorical", options=["adam", "adamw", "sgd"], values=["adam", "adamw", "sgd"]
        ),
        "optim.weight_decay": HyperparameterConfig(
            type="float", low=1e-6, high=1e-2, log=True, values=[1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
        ),
        "criterion.name": HyperparameterConfig(
            type="categorical", options=["focal", "cross_entropy"], values=["focal", "cross_entropy"]
        ),
        "batch_size": HyperparameterConfig(type="categorical", options=[2, 4], values=[2, 4]),
        "scheduler": HyperparameterConfig(
            type="categorical",
            options=["cosine", "onecycle", "plateau", "warmup_cosine", None],
            values=["cosine", "onecycle", "plateau", "warmup_cosine", None],
        ),
        "dropout_rate": HyperparameterConfig(
            type="float", low=0.0, high=0.6, values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        ),
        "mixup_alpha": HyperparameterConfig(type="float", low=0.0, high=0.6, values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.6]),
        "embed_dim": HyperparameterConfig(type="categorical", options=[96, 128, 192, 256], values=[96, 128, 192, 256]),
        "attention_dropout": HyperparameterConfig(
            type="float", low=0.0, high=0.3, values=[0.0, 0.05, 0.1, 0.15, 0.2, 0.3]
        ),
        "path_dropout": HyperparameterConfig(type="float", low=0.0, high=0.4, values=[0.0, 0.1, 0.2, 0.3, 0.4]),
    },
    tuner_configs={
        TunerType.PYTORCH_OPTUNA: {
            "n_trials": 20,
            "direction": "maximize",
            "scoring": "f1",
            "val_size": 0.2,
            "random_state": 42,
            "timeout": 7200,
        }
    },
)
