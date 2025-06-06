from src.training_module.model_core.tunner_module.hyperparameter_selection_system import (
    HyperparameterConfig,
    ModelHyperparameterSpace,
    TunerType,
)

cnn_mlp_hybrid_space = ModelHyperparameterSpace(
    model_name="cnn_mlp_hybrid",
    parameters={
        "epoch": HyperparameterConfig(type="int", low=10, high=50, values=[10, 20, 30, 40, 50]),
        "optim.lr": HyperparameterConfig(
            type="float", low=1e-5, high=1e-2, log=True, values=[1e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 5e-3, 1e-2]
        ),
        "optim.name": HyperparameterConfig(
            type="categorical", options=["adam", "adamw", "sgd"], values=["adam", "adamw", "sgd"]
        ),
        "criterion.name": HyperparameterConfig(type="categorical", options=["focal"], values=["focal"]),
        "img_size": HyperparameterConfig(type="categorical", options=[64, 128, 224, 256], values=[64, 128, 224, 256]),
        "dropout_rate": HyperparameterConfig(
            type="float", low=0.0, high=0.7, values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        ),
        "cnn_base_filters": HyperparameterConfig(type="categorical", options=[16, 32, 64], values=[16, 32, 64]),
        "cnn_depth": HyperparameterConfig(type="categorical", options=[3, 4, 5, 6], values=[3, 4, 5, 6]),
        "cnn_filter_multiplier": HyperparameterConfig(
            type="categorical", options=[1.5, 2.0, 2.5], values=[1.5, 2.0, 2.5]
        ),
        "batch_norm_momentum": HyperparameterConfig(type="float", low=0.05, high=0.2, values=[0.05, 0.1, 0.15, 0.2]),
        "activation_function": HyperparameterConfig(
            type="categorical",
            options=["relu", "leaky_relu", "elu", "swish"],
            values=["relu", "leaky_relu", "elu", "swish"],
        ),
        "batch_size": HyperparameterConfig(
            type="categorical", options=[8, 16, 32, 64, 128], values=[8, 16, 32, 64, 128]
        ),
        "normalize": HyperparameterConfig(type="categorical", options=[True, False], values=[True, False]),
        "scheduler": HyperparameterConfig(
            type="categorical",
            options=["cosine", "onecycle", "plateau", "step", None],
            values=["cosine", "onecycle", "plateau", "step", None],
        ),
        "patience": HyperparameterConfig(type="int", low=3, high=15, values=[3, 5, 7, 10, 15]),
        "onecycle.max_lr_multiplier": HyperparameterConfig(
            type="float", low=5.0, high=20.0, values=[5.0, 10.0, 15.0, 20.0]
        ),
        "plateau.factor": HyperparameterConfig(type="float", low=0.1, high=0.8, values=[0.1, 0.3, 0.5, 0.8]),
        "plateau.patience": HyperparameterConfig(type="int", low=2, high=8, values=[2, 3, 5, 8]),
        "step.step_size": HyperparameterConfig(type="int", low=5, high=20, values=[5, 10, 15, 20]),
        "step.gamma": HyperparameterConfig(type="float", low=0.1, high=0.9, values=[0.1, 0.3, 0.5, 0.7, 0.9]),
        "grad_clip_norm": HyperparameterConfig(type="float", low=0.5, high=2.0, values=[0.5, 1.0, 1.5, 2.0]),
        "augmentation_strength": HyperparameterConfig(
            type="categorical",
            options=["none", "light", "medium", "strong"],
            values=["none", "light", "medium", "strong"],
        ),
        "mixup_alpha": HyperparameterConfig(type="float", low=0.0, high=0.4, values=[0.0, 0.1, 0.2, 0.3, 0.4]),
        "feature_fusion_strategy": HyperparameterConfig(
            type="categorical",
            options=["concat", "attention", "gated", "weighted_sum"],
            values=["concat", "attention", "gated", "weighted_sum"],
        ),
        "cnn_branch_weight": HyperparameterConfig(type="float", low=0.3, high=0.9, values=[0.3, 0.5, 0.7, 0.9]),
        "stat_branch_weight": HyperparameterConfig(type="float", low=0.1, high=0.7, values=[0.1, 0.3, 0.5, 0.7]),
        "l2_regularization": HyperparameterConfig(
            type="float", low=1e-6, high=1e-3, log=True, values=[1e-6, 1e-5, 1e-4, 1e-3]
        ),
        "early_stopping_patience": HyperparameterConfig(type="int", low=5, high=20, values=[5, 10, 15, 20]),
        "pooling_strategy": HyperparameterConfig(
            type="categorical",
            options=["max", "avg", "adaptive_max", "adaptive_avg"],
            values=["max", "avg", "adaptive_max", "adaptive_avg"],
        ),
        "label_smoothing": HyperparameterConfig(type="float", low=0.0, high=0.2, values=[0.0, 0.05, 0.1, 0.15, 0.2]),
    },
    tuner_configs={
        TunerType.PYTORCH_OPTUNA: {
            "n_trials": 20,
            "direction": "maximize",
            "scoring": "f1",
            "val_size": 0.2,
            "random_state": 42,
            "timeout": 7200,
        },
    },
)
