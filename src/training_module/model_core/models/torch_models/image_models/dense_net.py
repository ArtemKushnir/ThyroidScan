from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
from monai.networks.nets import DenseNet121, EfficientNetBN, SEResNet50, ViT
from monai.transforms import NormalizeIntensity, Resize, ScaleIntensity

from src.training_module.model_core.base_models import PyTorchModel
from src.training_module.model_core.model_registry import ModelRegistry


@ModelRegistry.register("dense_net_image")
class ThyroidDenseNetModel(PyTorchModel):
    name = "dense_net_image"

    def __init__(
        self,
        model_params: Optional[dict[str, Any]] = None,
        is_binary: bool = True,
        pretrained: bool = True,
        spatial_dims: int = 2,
        n_input_channels: int = 1,
        num_classes: int = 1,
        growth_rate: int = 32,
        block_config: Tuple[int, int, int, int] = (6, 12, 24, 16),
    ):
        super().__init__(model_params, is_binary)

        self.pretrained = pretrained
        self.spatial_dims = spatial_dims
        self.n_input_channels = n_input_channels
        self.num_classes = num_classes
        self.growth_rate = growth_rate
        self.block_config = block_config

        self.model = self._create_model()
        self.preprocessing = self._initialize_preprocessing()

    def _create_model(self) -> nn.Module:
        out_classes = 1 if self.is_binary else self.num_classes

        model = DenseNet121(
            spatial_dims=self.spatial_dims,
            in_channels=self.n_input_channels,
            out_channels=out_classes,
            pretrained=self.pretrained,
            growth_rate=self.growth_rate,
            block_config=self.block_config,
        )

        return model

    def _initialize_preprocessing(self) -> List[Any]:
        input_size = (224, 224)

        transforms = [
            lambda x: x.unsqueeze(1) if x.dim() == 3 else x,
            Resize(spatial_size=input_size),
            ScaleIntensity(),
            NormalizeIntensity(subtrahend=np.array(0.485), divisor=np.array(0.229)),
        ]

        return transforms

    def _get_additional_state(self) -> Dict[str, Any]:
        return {
            "pretrained": self.pretrained,
            "spatial_dims": self.spatial_dims,
            "n_input_channels": self.n_input_channels,
            "num_classes": self.num_classes,
            "growth_rate": self.growth_rate,
            "block_config": self.block_config,
        }

    def _load_additional_state(self, checkpoint: Dict[str, Any]) -> None:
        self.pretrained = checkpoint.get("pretrained", True)
        self.spatial_dims = checkpoint.get("spatial_dims", 2)
        self.n_input_channels = checkpoint.get("n_input_channels", 1)
        self.num_classes = checkpoint.get("num_classes", 1)
        self.growth_rate = checkpoint.get("growth_rate", 32)
        self.block_config = checkpoint.get("block_config", (6, 12, 24, 16))

        if self.model is None:
            self.model = self._create_model()
            self.preprocessing = self._initialize_preprocessing()
