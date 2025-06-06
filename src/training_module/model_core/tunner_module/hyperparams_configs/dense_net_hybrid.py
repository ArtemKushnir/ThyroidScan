from src.training_module.model_core.tunner_module.hyperparameter_selection_system import (
    HyperparameterConfig,
    ModelHyperparameterSpace,
    TunerType,
)

dense_net_hybrid_space = ModelHyperparameterSpace(
    model_name="dense_net_hybrid",
    parameters={
        "epoch": HyperparameterConfig(type="int", low=10, high=50, values=[10, 20, 30, 40, 50]),
        "optim.lr": HyperparameterConfig(
            type="float", low=1e-5, high=1e-2, log=True, values=[1e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 5e-3, 1e-2]
        ),
        "optim.name": HyperparameterConfig(
            type="categorical", options=["adam", "adamw", "sgd"], values=["adam", "adamw", "sgd"]
        ),
        "optim.weight_decay": HyperparameterConfig(
            type="float", low=1e-6, high=1e-3, log=True, values=[1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
        ),
        "img_size": HyperparameterConfig(type="categorical", options=[224, 256, 288, 320], values=[224, 256, 288, 320]),
        "criterion.name": HyperparameterConfig(type="categorical", options=["focal"], values=["focal"]),
        "normalize": HyperparameterConfig(type="categorical", options=[True, False], values=[True, False]),
        "batch_size": HyperparameterConfig(type="categorical", options=[8, 16, 32, 64], values=[8, 16, 32, 64]),
        "scheduler": HyperparameterConfig(
            type="categorical",
            options=["cosine", "onecycle", "plateau", None],
            values=["cosine", "onecycle", "plateau", None],
        ),
        "patience": HyperparameterConfig(type="int", low=3, high=15, values=[3, 5, 7, 10, 15]),
        "grad_clip_norm": HyperparameterConfig(type="float", low=0.5, high=2.0, values=[0.5, 1.0, 1.5, 2.0]),
        "onecycle.max_lr_multiplier": HyperparameterConfig(
            type="float", low=5.0, high=20.0, values=[5.0, 10.0, 15.0, 20.0]
        ),
        "plateau.factor": HyperparameterConfig(type="float", low=0.1, high=0.8, values=[0.1, 0.3, 0.5, 0.8]),
        "plateau.patience": HyperparameterConfig(type="int", low=2, high=8, values=[2, 3, 5, 8]),
        "augmentation_strength": HyperparameterConfig(
            type="categorical",
            options=["none", "light", "medium", "strong"],
            values=["none", "light", "medium", "strong"],
        ),
        "dropout_rate": HyperparameterConfig(type="float", low=0.0, high=0.5, values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
        "mixup_alpha": HyperparameterConfig(type="float", low=0.0, high=0.4, values=[0.0, 0.1, 0.2, 0.3, 0.4]),
        "feature_fusion_dropout": HyperparameterConfig(type="float", low=0.0, high=0.3, values=[0.0, 0.1, 0.2, 0.3]),
        "statistical_features_weight": HyperparameterConfig(
            type="float", low=0.1, high=1.0, values=[0.1, 0.3, 0.5, 0.7, 1.0]
        ),
        "dense_block_config": HyperparameterConfig(
            type="categorical", options=["standard", "compact", "wide"], values=["standard", "compact", "wide"]
        ),
        "growth_rate": HyperparameterConfig(type="int", low=16, high=48, values=[16, 24, 32, 40, 48]),
        "compression": HyperparameterConfig(type="float", low=0.3, high=0.8, values=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
    },
    tuner_configs={
        TunerType.PYTORCH_OPTUNA: {
            "n_trials": 15,
            "direction": "maximize",
            "scoring": "f1",
            "val_size": 0.2,
            "random_state": 42,
            "timeout": 3600,
        }
    },
)
