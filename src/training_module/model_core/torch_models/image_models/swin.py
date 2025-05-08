from typing import Any, Callable, Optional

import numpy as np
import torchvision.models as tv_models
from monai.transforms import NormalizeIntensity, Resize, ScaleIntensity
from torch import nn

from src.training_module.model_core.base_models import PyTorchModel


class ThyroidSwinModel(PyTorchModel):
    """
    Swin Transformer implementation for thyroid ultrasound classification.
    Swin Transformer uses shifted windows for self-attention, which is effective
    for capturing both local and global features in medical images.
    """

    def __init__(
        self,
        optimizer: str = "adam",
        criterion: str = "bce_logit",
        model_params: Optional[dict[str, Any]] = None,
        is_binary: bool = True,
        pretrained: bool = True,
        n_input_channels: int = 1,
        num_classes: int = 1,
    ):
        super().__init__(optimizer, criterion, model_params, is_binary)

        self.pretrained = pretrained
        self.n_input_channels = n_input_channels
        self.num_classes = num_classes

        self.model = self._create_model()
        self.preprocessing = self._initialize_preprocessing()

    def _create_model(self) -> nn.Module:
        out_classes = 1 if self.is_binary else self.num_classes

        # Load pretrained Swin-T
        model = tv_models.swin_t(pretrained=self.pretrained)

        # Modify the first conv layer to accept the desired number of input channels
        if self.n_input_channels != 3:
            weight = model.features[0][0].weight.detach()
            model.features[0][0] = nn.Conv2d(
                self.n_input_channels, 96, kernel_size=(4, 4), stride=(4, 4)  # Swin-T first layer output channels
            )
            if self.pretrained:
                # For single-channel input, use the mean of the original weights
                if self.n_input_channels == 1:
                    model.features[0][0].weight.data = weight.sum(dim=1, keepdim=True)
                else:
                    # For other channel counts, initialize new channels
                    model.features[0][0].weight.data[:, : min(3, self.n_input_channels)] = weight[
                        :, : min(3, self.n_input_channels)
                    ]

        # Modify the classifier head
        model.head = nn.Linear(model.head.in_features, out_classes)

        return model

    def _initialize_preprocessing(self) -> list[Any]:
        # Swin-T uses 224x224 input
        input_size = (224, 224)

        transforms = [
            lambda x: x.unsqueeze(1) if x.dim() == 3 else x,
            Resize(spatial_size=input_size),
            ScaleIntensity(),
            NormalizeIntensity(subtrahend=np.array(0.485), divisor=np.array(0.229)),  # ImageNet stats
        ]

        return transforms

    def _get_additional_state(self) -> dict[str, Any]:
        return {
            "pretrained": self.pretrained,
            "n_input_channels": self.n_input_channels,
            "num_classes": self.num_classes,
        }

    def _load_additional_state(self, checkpoint: dict[str, Any]) -> None:
        self.pretrained = checkpoint.get("pretrained", True)
        self.n_input_channels = checkpoint.get("n_input_channels", 1)
        self.num_classes = checkpoint.get("num_classes", 1)

        if self.model is None:
            self.model = self._create_model()
            self.preprocessing = self._initialize_preprocessing()
