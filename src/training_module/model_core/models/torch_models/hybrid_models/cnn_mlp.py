import copy
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader

from src.training_module.model_core.base_models import PyTorchModel
from src.training_module.model_core.model_registry import ModelRegistry


class CustomNet(nn.Module):
    """
    Neural network for thyroid disease classification using both image and statistical features.
    The model processes images through a CNN branch and statistical features through an MLP branch,
    then combines them for final prediction.
    """

    def __init__(
        self,
        img_channels: int = 1,
        img_size: int = 224,
        num_statistical_features: int = 100,
        num_classes: int = 1,
        dropout_rate: float = 0.5,
    ):
        super(CustomNet, self).__init__()
        self.num_statistical_features = num_statistical_features

        self.cnn_branch = nn.Sequential(
            nn.Conv2d(img_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        cnn_output_size = 256 * (img_size // (2**4)) * (img_size // (2**4))

        self.stat_branch = nn.Sequential(
            nn.Linear(num_statistical_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        self.combined_layers = nn.Sequential(
            nn.Linear(cnn_output_size + 128, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: Any) -> Any:
        if isinstance(x, dict):
            img = x["pixels"]
            stat_features = x["features"]
        elif isinstance(x, tuple) and len(x) == 2:
            img, stat_features = x
        else:
            raise ValueError("Input must be either a dict with 'pixels' and 'features' keys or a tuple (img, features)")

        img_features = self.cnn_branch(img)
        img_features = img_features.view(img_features.size(0), -1)  # Flatten

        stat_output = self.stat_branch(stat_features)

        combined_features = torch.cat((img_features, stat_output), dim=1)

        output = self.combined_layers(combined_features)

        return output


@ModelRegistry.register("cnn_mlp_hybrid")
class CNNMLPModel(PyTorchModel):
    """
    PyTorch model for thyroid disease classification using both US images and statistical features.
    This model implements the PyTorchModel interface.
    """

    name = "cnn_mlp_hybrid"
    _data_adapter_type = "hybrid"

    def __init__(
        self,
        model_params: Optional[Dict[str, Any]] = None,
        is_binary: bool = True,
        img_channels: int = 1,
        img_size: int = 64,
        num_statistical_features: int = 100,
    ):
        super().__init__(model_params, is_binary)

        self.img_channels = img_channels
        self.img_size = img_size
        self.num_statistical_features = num_statistical_features
        self.num_classes = 1 if is_binary else 6  # Assuming TIRADS 0-5 for multiclass

        self.image_transform = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),  # Resize images to expected dimensions
                transforms.Lambda(lambda x: x if x.shape[1] == self.img_channels else x.mean(dim=1, keepdim=True)),
            ]
        )

        self.model = self._create_model()
        self.preprocessing = self._initialize_preprocessing()

    def _create_model(self) -> nn.Module:
        return CustomNet(
            img_channels=self.img_channels,
            img_size=self.img_size,
            num_statistical_features=self.num_statistical_features,
            num_classes=self.num_classes,
            dropout_rate=self.model_params.get("dropout_rate", 0.5),
        )

    def _initialize_preprocessing(self) -> List[Callable]:
        preprocessing = []

        if self.model_params.get("normalize", True):
            preprocessing.append(lambda x: self._normalize_images(x) if isinstance(x, dict) else x)

        return preprocessing

    def _resize_image(self, img: torch.Tensor) -> torch.Tensor:
        batch_size = img.shape[0]
        resized_imgs = []

        for i in range(batch_size):
            single_img = img[i : i + 1]
            resized_img = self.image_transform(single_img)
            resized_imgs.append(resized_img)

        return torch.cat(resized_imgs, dim=0)

    @staticmethod
    def _normalize_images(batch_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if "pixels" in batch_data:
            if batch_data["pixels"].max() > 1.0:
                batch_data["pixels"] = batch_data["pixels"] / 255.0
        return batch_data

    def _prepare_input(self, batch_data: Any) -> Tuple[Any, torch.Tensor]:
        if isinstance(batch_data, dict):
            pixels = batch_data["pixels"].to(self.device)
            features = batch_data["features"].to(self.device) if "features" in batch_data else None
            y = batch_data["label"].to(self.device) if "label" in batch_data else torch.tensor([])

            if pixels.shape[2] != self.img_size or pixels.shape[3] != self.img_size:
                pixels = self._resize_image(pixels)

            if pixels.shape[1] != self.img_channels:
                if self.img_channels == 1 and pixels.shape[1] > 1:
                    pixels = pixels.mean(dim=1, keepdim=True)
                elif self.img_channels > 1 and pixels.shape[1] == 1:
                    pixels = pixels.repeat(1, self.img_channels, 1, 1)

            for transform in self.preprocessing:
                pixels = transform({"pixels": pixels})["pixels"]

            X = {"pixels": pixels, "features": features}

        else:
            raise ValueError(f"Unsupported batch data format: {type(batch_data)}")

        return X, y

    def _get_additional_state(self) -> Dict[str, Any]:
        return {
            "img_channels": self.img_channels,
            "img_size": self.img_size,
            "num_statistical_features": self.num_statistical_features,
        }

    def _load_additional_state(self, checkpoint: Dict[str, Any]) -> None:
        self.img_channels = checkpoint.get("img_channels", self.img_channels)
        self.img_size = checkpoint.get("img_size", self.img_size)
        self.num_statistical_features = checkpoint.get("num_statistical_features", self.num_statistical_features)

        self.image_transform = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.Lambda(lambda x: x if x.shape[1] == self.img_channels else x.mean(dim=1, keepdim=True)),
            ]
        )
