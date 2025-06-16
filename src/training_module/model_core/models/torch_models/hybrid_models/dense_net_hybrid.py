from typing import Any, Callable, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training_module.data_model_bridge.data_adapters import BaseDataAdapter
from src.training_module.model_core.base_models import PyTorchModel
from src.training_module.model_core.model_registry import ModelRegistry
from src.training_module.model_core.models.torch_models.hybrid_models.hybrid_blocks import DenseNet121Thyroid


@ModelRegistry.register("dense_net_hybrid")
class DenseNetHybridModel(PyTorchModel):
    """
    DenseNet121-based model for thyroid disease classification
    """

    name = "dense_net_hybrid"
    _data_adapter_type = "hybrid"

    def __init__(
        self,
        model_params: Optional[dict[str, Any]] = None,
        is_binary: bool = True,
        img_channels: int = 1,
        img_size: int = 224,
        num_statistical_features: int = 37,
        pretrained: bool = True,
    ):
        default_params = {
            "epoch": 10,
            "optim": {"name": "adam", "lr": 0.0001, "weight_decay": 1e-5},
            "criterion": {"name": "focal"},
            "normalize": True,
        }

        if model_params:
            default_params.update(model_params)

        super().__init__(default_params, is_binary)

        self.img_channels = img_channels
        self.img_size = img_size
        self.num_statistical_features = num_statistical_features
        self.num_classes = 1 if is_binary else 6  # Assuming TIRADS 0-5
        self.pretrained = pretrained

        self.model = self._create_model()

        self.preprocessing = self._initialize_preprocessing()

    def _create_model(self) -> nn.Module:
        return DenseNet121Thyroid(
            img_channels=self.img_channels,
            num_statistical_features=self.num_statistical_features,
            num_classes=self.num_classes,
            pretrained=self.pretrained,
        )

    def _initialize_preprocessing(self) -> list[Callable]:
        preprocessing = []

        if self.model_params.get("normalize", True):

            def normalize_fn(x: Any) -> Any:
                if isinstance(x, dict) and "pixels" in x:
                    if self.pretrained and x["pixels"].max() <= 1.0:
                        if self.img_channels == 1 and x["pixels"].size(1) == 1:
                            x["pixels"] = x["pixels"].repeat(1, 3, 1, 1)

                        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x["pixels"].device)
                        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x["pixels"].device)
                        x["pixels"] = (x["pixels"] - mean) / std
                    elif x["pixels"].max() > 1.0:
                        x["pixels"] = x["pixels"] / 255.0
                return x

            preprocessing.append(normalize_fn)

        return preprocessing

    def _prepare_input(self, batch_data: Any) -> tuple[dict[str, Any], torch.Tensor]:
        if isinstance(batch_data, dict):
            X_data = {"pixels": batch_data["pixels"], "features": batch_data["features"]}
            y = batch_data["label"].to(self.device) if "label" in batch_data else torch.tensor([])
        else:
            raise ValueError(f"Unsupported batch data format: {type(batch_data)}")

        return X_data, y

    def fit(self, train_adapter: "BaseDataAdapter", test_adapter: Optional["BaseDataAdapter"] = None) -> "PyTorchModel":
        train_loader = train_adapter.data

        self.model = self.model.to(self.device)

        optim_params = dict(filter(lambda item: item[0] != "name", self.model_params["optim"].items()))

        optimizer = self.optimizer(self.model.parameters(), **optim_params)
        criterion = self.criterion()
        epochs = self.model_params.get("epoch", 10)

        for epoch in range(epochs):
            train_loss = self.train_loop(train_loader, self.model, criterion, optimizer)

            if test_adapter is not None:
                metrics = self.validate(test_adapter)
                val_loss = metrics["loss"]
            else:
                val_loss, metrics = 0.0, {}

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            for metric_name, metric_value in metrics.items():
                if metric_name not in self.history["metrics"]:
                    self.history["metrics"][metric_name] = []
                self.history["metrics"][metric_name].append(metric_value)

            print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
            for metric_name, metric_value in metrics.items():
                print(f" - {metric_name}: {metric_value:.4f}")

        return self

    def _get_additional_state(self) -> dict[str, Any]:
        return {
            "img_channels": self.img_channels,
            "img_size": self.img_size,
            "num_statistical_features": self.num_statistical_features,
            "pretrained": self.pretrained,
        }

    def _load_additional_state(self, checkpoint: dict[str, Any]) -> None:
        self.img_channels = checkpoint.get("img_channels", self.img_channels)
        self.img_size = checkpoint.get("img_size", self.img_size)
        self.num_statistical_features = checkpoint.get("num_statistical_features", self.num_statistical_features)
        self.pretrained = checkpoint.get("pretrained", self.pretrained)
