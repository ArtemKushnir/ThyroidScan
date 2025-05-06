from typing import Any, Callable, Optional

import numpy as np
from monai.networks.nets import EfficientNetBN
from monai.transforms import NormalizeIntensity, Resize, ScaleIntensity
from torch import nn

from src.training_module.model_core.base_models import PyTorchModel


class ThyroidEfficientNetModel(PyTorchModel):
    def __init__(
        self,
        optimizer: str = "adam",
        criterion: str = "bce_logit",
        model_params: Optional[dict[str, Any]] = None,
        is_binary: bool = True,
        model_name: str = "efficientnet-b0",
        pretrained: bool = True,
        spatial_dims: int = 2,
        n_input_channels: int = 1,
        num_classes: int = 1,
    ):
        super().__init__(optimizer, criterion, model_params, is_binary)

        self.model_name = model_name
        self.pretrained = pretrained
        self.spatial_dims = spatial_dims
        self.n_input_channels = n_input_channels
        self.num_classes = num_classes

        self.model = self._create_model()
        self.preprocessing = self._initialize_preprocessing()

    def _create_model(self) -> nn.Module:
        out_classes = 1 if self.is_binary else self.num_classes

        model = EfficientNetBN(
            model_name=self.model_name,
            spatial_dims=self.spatial_dims,
            in_channels=self.n_input_channels,
            num_classes=out_classes,
            pretrained=self.pretrained,
        )

        return model

    def _initialize_preprocessing(self) -> list[Any]:
        if self.model_name in ["efficientnet-b0", "efficientnet-b1"]:
            input_size = (224, 224)
        elif self.model_name in ["efficientnet-b2", "efficientnet-b3"]:
            input_size = (256, 256)
        elif self.model_name == "efficientnet-b4":
            input_size = (320, 320)
        else:
            input_size = (384, 384)

        transforms = [
            lambda x: x.unsqueeze(1) if x.dim() == 3 else x,
            Resize(spatial_size=input_size),
            ScaleIntensity(),
            NormalizeIntensity(subtrahend=np.array(0.5), divisor=np.array(0.5)),
        ]

        return transforms

    def _get_additional_state(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "pretrained": self.pretrained,
            "spatial_dims": self.spatial_dims,
            "n_input_channels": self.n_input_channels,
            "num_classes": self.num_classes,
        }

    def _load_additional_state(self, checkpoint: dict[str, Any]) -> None:
        self.model_name = checkpoint.get("model_name", "efficientnet-b0")
        self.pretrained = checkpoint.get("pretrained", True)
        self.spatial_dims = checkpoint.get("spatial_dims", 2)
        self.n_input_channels = checkpoint.get("n_input_channels", 1)
        self.num_classes = checkpoint.get("num_classes", 1)

        if self.model is None:
            self.model = self._create_model()
            self.preprocessing = self._initialize_preprocessing()
