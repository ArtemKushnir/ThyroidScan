from typing import Any, Optional

import torch
import torch.nn as nn
import torchvision.models as tv_models
from monai.transforms import Resize, ScaleIntensity

from src.training_module.model_core.base_models import PyTorchModel
from src.training_module.model_core.model_registry import ModelRegistry


@ModelRegistry.register("resnet_image")
class ResNetImageModel(PyTorchModel):
    name = "resnet_image"
    _data_adapter_type = "image"

    def __init__(
        self,
        model_params: Optional[dict[str, Any]] = None,
        is_binary: bool = True,
        resnet_model_name: str = "resnet18",
        pretrained: bool = True,
    ):
        super().__init__(model_params, is_binary)

        self.resnet_model_name = resnet_model_name
        self.pretrained = pretrained
        self.n_input_channels = 1
        self.num_classes = 1 if is_binary else 6

        self.model = self._create_model()

        self.preprocessing = self._initialize_preprocessing()

    def _create_model(self) -> nn.Module:
        out_classes = 1 if self.is_binary else self.num_classes

        # Use torchvision models instead of MONAI for 2D medical images
        if self.resnet_model_name == "resnet18":
            model = tv_models.resnet18(weights="DEFAULT" if self.pretrained else None)
        elif self.resnet_model_name == "resnet34":
            model = tv_models.resnet34(weights="DEFAULT" if self.pretrained else None)
        elif self.resnet_model_name == "resnet50":
            model = tv_models.resnet50(weights="DEFAULT" if self.pretrained else None)
        else:
            raise ValueError(f"Unsupported ResNet model name: {self.resnet_model_name}")

        if self.n_input_channels != 3:
            original_layer = model.conv1
            model.conv1 = nn.Conv2d(
                self.n_input_channels,
                original_layer.out_channels,
                kernel_size=original_layer.kernel_size,
                stride=original_layer.stride,
                padding=original_layer.padding,
                bias=original_layer.bias is not None,
            )

            if self.pretrained:
                with torch.no_grad():
                    model.conv1.weight = nn.Parameter(
                        original_layer.weight.mean(dim=1, keepdim=True).repeat(1, self.n_input_channels, 1, 1)
                    )

        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, out_classes)

        return model

    def _initialize_preprocessing(self) -> list[Any]:
        input_size = (224, 224)

        transforms = [
            lambda x: x.unsqueeze(1) if x.dim() == 3 else x,
            Resize(spatial_size=input_size),
            ScaleIntensity(),
        ]

        return transforms

    def _get_additional_state(self) -> dict[str, Any]:
        return {
            "resnet_model_name": self.resnet_model_name,
            "pretrained": self.pretrained,
            "n_input_channels": self.n_input_channels,
            "num_classes": self.num_classes,
        }

    def _load_additional_state(self, checkpoint: dict[str, Any]) -> None:
        self.resnet_model_name = checkpoint.get("resnet_model_name", "resnet18")
        self.pretrained = checkpoint.get("pretrained", True)
        self.n_input_channels = checkpoint.get("n_input_channels", 1)
        self.num_classes = checkpoint.get("num_classes", 1)

        if self.model is None:
            self.model = self._create_model()
            self.preprocessing = self._initialize_preprocessing()
