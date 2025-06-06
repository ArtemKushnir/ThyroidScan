from src.training_module.model_core.tunner_module.hyperparameter_selection_system import HyperParameterManger
from src.training_module.model_core.tunner_module.hyperparams_configs.catboost import catboost_space
from src.training_module.model_core.tunner_module.hyperparams_configs.cnn_mlp_hybrid import cnn_mlp_hybrid_space
from src.training_module.model_core.tunner_module.hyperparams_configs.dense_net_hybrid import dense_net_hybrid_space
from src.training_module.model_core.tunner_module.hyperparams_configs.dense_net_image import dense_net_image_space
from src.training_module.model_core.tunner_module.hyperparams_configs.efficient_net_hybrid import (
    efficient_net_hybrid_space,
)
from src.training_module.model_core.tunner_module.hyperparams_configs.efficient_net_image import (
    efficient_net_image_space,
)
from src.training_module.model_core.tunner_module.hyperparams_configs.lgb import lgb_space
from src.training_module.model_core.tunner_module.hyperparams_configs.random_forest import random_forest_space
from src.training_module.model_core.tunner_module.hyperparams_configs.resnet_image import resnet_image_space
from src.training_module.model_core.tunner_module.hyperparams_configs.se_resnet_image import se_resnet_space
from src.training_module.model_core.tunner_module.hyperparams_configs.svm import svm_space
from src.training_module.model_core.tunner_module.hyperparams_configs.swin_image import swin_image_space
from src.training_module.model_core.tunner_module.hyperparams_configs.xgboost import xgboost_space


def create_configs(hp_manager: HyperParameterManger) -> None:
    hp_manager.register_model_space(catboost_space)
    hp_manager.register_model_space(lgb_space)
    hp_manager.register_model_space(random_forest_space)
    hp_manager.register_model_space(svm_space)
    hp_manager.register_model_space(xgboost_space)

    hp_manager.register_model_space(se_resnet_space)
    hp_manager.register_model_space(resnet_image_space)
    hp_manager.register_model_space(dense_net_image_space)
    hp_manager.register_model_space(swin_image_space)
    hp_manager.register_model_space(efficient_net_image_space)

    hp_manager.register_model_space(efficient_net_hybrid_space)
    hp_manager.register_model_space(cnn_mlp_hybrid_space)
    hp_manager.register_model_space(dense_net_hybrid_space)


if __name__ == "__main__":
    hp_manager = HyperParameterManger()
    create_configs(hp_manager)
