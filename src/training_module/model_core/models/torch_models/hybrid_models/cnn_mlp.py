import copy
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader

from src.training_module.model_core.base_models import PyTorchModel
from src.training_module.model_core.model_registry import ModelRegistry


class ThyroidNet(nn.Module):
    """
    Neural network for thyroid disease classification using both image and statistical features.
    The model processes images through a CNN branch and statistical features through an MLP branch,
    then combines them for final prediction.
    """

    def __init__(
        self,
        img_channels: int = 1,
        img_size: int = 224,
        num_statistical_features: int = 10,
        num_classes: int = 1,
        dropout_rate: float = 0.5,
    ):
        super(ThyroidNet, self).__init__()

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
class ThyroidClassificationModel(PyTorchModel):
    """
    PyTorch model for thyroid disease classification using both US images and statistical features.
    This model implements the PyTorchModel interface.
    """

    def __init__(
        self,
        optimizer: str = "adam",
        criterion: str = "bce",
        model_params: Optional[Dict[str, Any]] = None,
        is_binary: bool = True,
        img_channels: int = 1,
        img_size: int = 224,
        num_statistical_features: int = 10,
    ):
        super().__init__(optimizer, criterion, model_params, is_binary)

        self.img_channels = img_channels
        self.img_size = img_size
        self.num_statistical_features = num_statistical_features
        self.num_classes = 1 if is_binary else 6  # Assuming TIRADS 0-5 for multiclass

        self.model = self._create_model()

        self.preprocessing = self._initialize_preprocessing()

    def _create_model(self) -> nn.Module:
        return ThyroidNet(
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

            for transform in self.preprocessing:
                pixels = transform({"pixels": pixels})["pixels"]

            X = {"pixels": pixels, "features": features}

        elif isinstance(batch_data, (list, tuple)) and len(batch_data) >= 2:
            X_data, y = batch_data[0], batch_data[1]

            if isinstance(X_data, dict):
                X = {"pixels": X_data["pixels"].to(self.device), "features": X_data["features"].to(self.device)}
                y = y.to(self.device) if y is not None else torch.tensor([])

                for transform in self.preprocessing:
                    X = transform(X)
            else:
                raise ValueError("For tuple input, the first element must be a dict with 'pixels' and 'features'")
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
