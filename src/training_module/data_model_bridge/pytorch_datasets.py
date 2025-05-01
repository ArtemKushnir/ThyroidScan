from typing import Any, Callable, Optional

import torch
from torch.utils.data import Dataset

from src.image_data.image_data import Image


class PytorchDatasetMixin:
    def __init__(self, images: list[Image], transform: Optional[Callable] = None) -> None:
        self.images = images
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def _get_pixels(self, index: int) -> torch.Tensor:
        img = self.images[index]

        image_tensor = torch.from_numpy(img.org_image).float()
        if self.transform is not None:
            image_tensor = self.transform(image_tensor)
        return image_tensor

    def _get_features(self, index: int) -> torch.Tensor:
        img = self.images[index]
        if img.features is None:
            raise ValueError("Missing features")
        features = torch.tensor(list(img.features.values())).float()
        return features

    def _get_label(self, index: int) -> torch.Tensor:
        img = self.images[index]
        if img.metadata is None:
            raise ValueError("Missing metadata")
        label = torch.tensor(img.metadata["tirads"])
        return label


class HybridDataset(Dataset, PytorchDatasetMixin):
    def __init__(self, images: list[Image], transform: Optional[Callable] = None) -> None:
        PytorchDatasetMixin.__init__(self, images, transform)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "pixels": self._get_pixels(index),
            "features": self._get_features(index),
            "label": self._get_label(index),
        }


class ImageDataset(Dataset, PytorchDatasetMixin):
    def __init__(self, images: list[Image], transform: Optional[Callable] = None) -> None:
        PytorchDatasetMixin.__init__(self, images, transform)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {"pixels": self._get_pixels(index), "label": self._get_label(index)}
