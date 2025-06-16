import copy
from typing import Any, Callable, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.training_module.data_model_bridge.data_adapters import BaseDataAdapter
from src.training_module.model_core.base_models import PyTorchModel
from src.training_module.model_core.model_registry import ModelRegistry
from src.training_module.model_core.models.torch_models.hybrid_models.hybrid_blocks import EfficientNetThyroid


@ModelRegistry.register("efficient_net_hybrid")
class EfficientNetHybridModel(PyTorchModel):
    """
    EfficientNet-based model for thyroid disease classification
    """

    name = "efficient_net_hybrid"
    _data_adapter_type = "hybrid"

    def __init__(
        self,
        model_params: Optional[dict[str, Any]] = None,
        is_binary: bool = True,
        img_channels: int = 1,
        img_size: int = 224,
        num_statistical_features: int = 37,
        model_variant: str = "b0",
        pretrained: bool = True,
        mixup_alpha: float = 0.2,
    ):
        # Update default model params with SOTA settings
        default_params = {
            "epoch": 40,
            "optim": {"name": "sgd", "lr": 0.0002, "weight_decay": 1e-4},
            "criterion": {"name": "focal"},
            "normalize": True,
            "scheduler": "cosine",
        }

        if model_params:
            default_params.update(model_params)

        super().__init__(default_params, is_binary)

        self.img_channels = img_channels
        self.img_size = img_size
        self.num_statistical_features = num_statistical_features
        self.num_classes = 1 if is_binary else 6  # Assuming TIRADS 0-5
        self.model_variant = model_variant
        self.pretrained = pretrained
        self.mixup_alpha = mixup_alpha

        self.model = self._create_model()

        self.preprocessing = self._initialize_preprocessing()

    def _create_model(self) -> nn.Module:
        return EfficientNetThyroid(
            img_channels=self.img_channels,
            num_statistical_features=self.num_statistical_features,
            num_classes=self.num_classes,
            model_variant=self.model_variant,
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

    @staticmethod
    def _mixup_data(
        x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    @staticmethod
    def _mixup_criterion(
        criterion: Callable, pred: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor, lam: float
    ) -> float:
        """Mixup loss calculation"""
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    def _prepare_input(self, batch_data: Any) -> tuple[Any, torch.Tensor]:
        if isinstance(batch_data, dict):
            X_data = {"pixels": batch_data["pixels"], "features": batch_data["features"]}
            y = batch_data["label"].to(self.device) if "label" in batch_data else torch.tensor([])
        else:
            raise ValueError(f"Unsupported batch data format: {type(batch_data)}")

        return X_data, y

    def train_loop(
        self, dataloader: DataLoader, model: nn.Module, loss_fn: nn.Module, optimizer: torch.optim.Optimizer
    ) -> float:
        size = len(dataloader.dataset)  # type: ignore
        model.train()
        running_loss = 0.0
        use_mixup = self.mixup_alpha > 0

        for batch, batch_data in enumerate(dataloader):
            X, y = self._prepare_input(batch_data)

            # Apply mixup if enabled
            if use_mixup and isinstance(X, dict) and "pixels" in X:
                # Only apply mixup to images, not to statistical features
                pixels_mixed, y_a, y_b, lam = self._mixup_data(X["pixels"], y, self.mixup_alpha)
                X["pixels"] = pixels_mixed
                optimizer.zero_grad()
                pred = model(X)

                if self.is_binary and pred.shape != y_a.shape:
                    y_a = y_a.view(pred.shape)
                    y_b = y_b.view(pred.shape)

                loss = self._mixup_criterion(loss_fn, pred, y_a, y_b, lam)
            else:
                optimizer.zero_grad()
                pred = model(X)

                if self.is_binary and pred.shape != y.shape:
                    y = y.view(pred.shape)

                loss = loss_fn(pred, y)

            loss.backward()  # type: ignore

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            running_loss += loss.item()  # type: ignore

        return running_loss / size

    def fit(self, train_adapter: "BaseDataAdapter", test_adapter: Optional["BaseDataAdapter"] = None) -> "PyTorchModel":

        train_loader = train_adapter.data

        self.model = self.model.to(self.device)

        optim_params = dict(filter(lambda item: item[0] != "name", self.model_params["optim"].items()))

        optimizer = self.optimizer(self.model.parameters(), **optim_params)
        criterion = self.criterion()
        epochs = self.model_params.get("epoch", 10)

        # Initialize scheduler
        scheduler_type = self.model_params.get("scheduler", None)
        if scheduler_type == "cosine":
            scheduler: Any = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        elif scheduler_type == "onecycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.model_params["optim"]["lr"] * 10,
                epochs=epochs,
                steps_per_epoch=len(train_loader),
            )
        elif scheduler_type == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
        else:
            scheduler = None

        # Early stopping parameters
        patience = self.model_params.get("patience", 7)
        best_loss = float("inf")
        early_stop_counter = 0
        best_model_state = None

        for epoch in range(epochs):
            train_loss = self.train_loop(train_loader, self.model, criterion, optimizer)

            if test_adapter is not None:
                metrics = self.validate(test_adapter)
                val_loss = metrics["loss"]

                if val_loss < best_loss:
                    best_loss = val_loss
                    early_stop_counter = 0
                    best_model_state = copy.deepcopy(self.model.state_dict())
                else:
                    early_stop_counter += 1

                if early_stop_counter >= patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    if best_model_state:
                        self.model.load_state_dict(best_model_state)
                    break
            else:
                val_loss, metrics = 0.0, {}

            # Update scheduler
            if scheduler is not None:
                if scheduler_type == "plateau" and test_adapter is not None:
                    scheduler.step(val_loss)
                elif scheduler_type == "onecycle":
                    # OneCycleLR is updated after each batch, not epoch
                    pass
                else:
                    scheduler.step()

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
            "model_variant": self.model_variant,
            "pretrained": self.pretrained,
            "mixup_alpha": self.mixup_alpha,
        }

    def _load_additional_state(self, checkpoint: dict[str, Any]) -> None:
        self.img_channels = checkpoint.get("img_channels", self.img_channels)
        self.img_size = checkpoint.get("img_size", self.img_size)
        self.num_statistical_features = checkpoint.get("num_statistical_features", self.num_statistical_features)
        self.model_variant = checkpoint.get("model_variant", self.model_variant)
        self.pretrained = checkpoint.get("pretrained", self.pretrained)
        self.mixup_alpha = checkpoint.get("mixup_alpha", self.mixup_alpha)
