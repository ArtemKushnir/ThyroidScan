from src.training_module.model_core.tunner_module.hyperparameter_selection_system import (
    HyperparameterConfig,
    ModelHyperparameterSpace,
    TunerType,
)

efficient_net_hybrid_space = ModelHyperparameterSpace(
    model_name="efficient_net_hybrid",
    parameters={
        "epoch": HyperparameterConfig(type="int", low=10, high=50, values=[10, 20, 30, 40, 50]),
        "optim.lr": HyperparameterConfig(
            type="float", low=1e-5, high=1e-2, log=True, values=[1e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 5e-3, 1e-2]
        ),
        "optim.name": HyperparameterConfig(
            type="categorical", options=["adam", "adamw", "sgd"], values=["adam", "adamw", "sgd"]
        ),
        "model_variant": HyperparameterConfig(
            type="categorical", options=["b0", "b1", "b2", "b3", "b4"], values=["b0", "b1", "b2", "b3", "b4"]
        ),
        "img_size": HyperparameterConfig(type="categorical", options=[224, 256, 288, 320], values=[224, 256, 288, 320]),
        "mixup_alpha": HyperparameterConfig(type="float", low=0.0, high=0.4, values=[0.0, 0.1, 0.2, 0.3, 0.4]),
        "criterion.name": HyperparameterConfig(type="categorical", options=["focal"], values=["focal"]),
        "scheduler": HyperparameterConfig(
            type="categorical",
            options=["cosine", "onecycle", "plateau", None],
            values=["cosine", "onecycle", "plateau", None],
        ),
        "patience": HyperparameterConfig(type="int", low=3, high=15, values=[3, 5, 7, 10, 15]),
        "batch_size": HyperparameterConfig(type="categorical", options=[8, 16, 32, 64], values=[8, 16, 32, 64]),
        "normalize": HyperparameterConfig(type="categorical", options=[True, False], values=[True, False]),
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
        "feature_fusion_dropout": HyperparameterConfig(type="float", low=0.0, high=0.3, values=[0.0, 0.1, 0.2, 0.3]),
        "statistical_features_weight": HyperparameterConfig(
            type="float", low=0.1, high=1.0, values=[0.1, 0.3, 0.5, 0.7, 1.0]
        ),
    },
    tuner_configs={
        TunerType.PYTORCH_OPTUNA: {
            "n_trials": 10,
            "direction": "maximize",
            "scoring": "f1",
            "val_size": 0.2,
            "random_state": 42,
        }
    },
)
