from typing import Any, Callable, Dict, List, Optional

import torch.nn as nn
from monai.networks.nets import resnet18, resnet34, resnet50
from monai.transforms import Resize, ScaleIntensity

from src.training_module.model_core.base_models import PyTorchModel
from src.training_module.model_core.model_registry import ModelRegistry


@ModelRegistry.register("resnet_image")
class ThyroidResNetModel(PyTorchModel):
    def __init__(
        self,
        optimizer: str = "adam",
        criterion: str = "bce_logit",
        model_params: Optional[dict[str, Any]] = None,
        is_binary: bool = True,
        resnet_model_name: str = "resnet18",
        pretrained: bool = True,
        spatial_dims: int = 2,
        n_input_channels: int = 1,
        num_classes: int = 1,
    ):
        super().__init__(optimizer, criterion, model_params, is_binary)

        self.resnet_model_name = resnet_model_name
        self.pretrained = pretrained
        self.spatial_dims = spatial_dims
        self.n_input_channels = n_input_channels
        self.num_classes = num_classes

        self.model = self._create_model()

        self.preprocessing = self._initialize_preprocessing()

    def _create_model(self) -> nn.Module:
        out_classes = 1 if self.is_binary else self.num_classes

        if self.resnet_model_name == "resnet18":
            model = resnet18(
                pretrained=self.pretrained,
                spatial_dims=self.spatial_dims,
                n_input_channels=self.n_input_channels,
                num_classes=out_classes,
            )
        elif self.resnet_model_name == "resnet34":
            model = resnet34(
                pretrained=self.pretrained,
                spatial_dims=self.spatial_dims,
                n_input_channels=self.n_input_channels,
                num_classes=out_classes,
            )
        elif self.resnet_model_name == "resnet50":
            model = resnet50(
                pretrained=self.pretrained,
                spatial_dims=self.spatial_dims,
                n_input_channels=self.n_input_channels,
                num_classes=out_classes,
            )
        else:
            raise ValueError(f"Unsupported ResNet model name: {self.resnet_model_name}")

        return model

    def _initialize_preprocessing(self) -> List[Any]:
        input_size = (224, 224)

        transforms = [
            lambda x: x.unsqueeze(1) if x.dim() == 3 else x,
            Resize(spatial_size=input_size),
            ScaleIntensity(),
        ]

        return transforms

    def _get_additional_state(self) -> Dict[str, Any]:
        return {
            "resnet_model_name": self.resnet_model_name,
            "pretrained": self.pretrained,
            "spatial_dims": self.spatial_dims,
            "n_input_channels": self.n_input_channels,
            "num_classes": self.num_classes,
        }

    def _load_additional_state(self, checkpoint: Dict[str, Any]) -> None:
        self.resnet_model_name = checkpoint.get("resnet_model_name", "resnet18")
        self.pretrained = checkpoint.get("pretrained", True)
        self.spatial_dims = checkpoint.get("spatial_dims", 2)
        self.n_input_channels = checkpoint.get("n_input_channels", 1)
        self.num_classes = checkpoint.get("num_classes", 1)

        if self.model is None:
            self.model = self._create_model()
            self.preprocessing = self._initialize_preprocessing()
