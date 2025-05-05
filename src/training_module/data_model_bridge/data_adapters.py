import abc
from typing import Any, Callable, Optional, Type, Union

import numpy as np
from monai.transforms import Compose, RandFlip, RandGaussianNoise
from torch.utils.data import DataLoader

from src.image_data.image_data import Image
from src.training_module.data_model_bridge.pytorch_datasets import HybridDataset, ImageDataset

CustomDataset = Union[HybridDataset, ImageDataset]


class DataAdapterFactory:
    _registry: dict = {}

    @classmethod
    def register(cls, adapter_type: str) -> Callable[[Type], Type]:
        def decorator(data_adapter_cls: Type) -> Type:
            if adapter_type in cls._registry:
                raise ValueError(f"DataAdapter '{adapter_type}' already registered")
            cls._registry[adapter_type] = data_adapter_cls
            return data_adapter_cls

        return decorator

    @classmethod
    def create(cls, adapter_type: str, **kwargs: Any) -> "BaseDataAdapter":
        if adapter_type not in cls._registry:
            raise ValueError(f"DataAdapter {adapter_type} is not registered")
        return cls._registry[adapter_type](**kwargs)


class BaseDataAdapter(abc.ABC):
    """Convert list[Images] to the desired format"""

    def __init__(self, images: list[Image], is_bin_classification: bool = True, label: bool = True) -> None:
        self.images: list[Image] = images
        self._is_bin_classification: bool = is_bin_classification
        self._preprocessed: bool = False
        self._data: Any = None
        self.label: bool = label
        self._check_constraint_images()
        if label:
            self._convert_image_labels()

    @abc.abstractmethod
    def prepare(self) -> None:
        pass

    @property
    def data(self) -> Any:
        return self._data

    def _is_prepare(self) -> None:
        if not self._preprocessed:
            raise ValueError("Data is not prepare")

    def _convert_tirads(self, tirads_string: str) -> int:
        tirads_digits = "".join([ch for ch in tirads_string if ch.isdigit()])
        if tirads_digits == "":
            raise ValueError("'tirads' must contain a number")
        tirads_numeric = int(tirads_digits)
        if tirads_numeric < 0 or tirads_numeric > 5:
            raise ValueError("'tirads' should be between 0 and 5")
        if self._is_bin_classification:
            return 1 if tirads_numeric >= 4 else 0
        else:
            return tirads_numeric

    def _convert_image_labels(self) -> None:
        for img in self.images:
            img.metadata["tirads"] = self._convert_tirads(img.metadata["tirads"])  # type: ignore

    def _check_constraint_images(self) -> None:
        for img in self.images:
            if img.features is None:
                raise ValueError("Missing features")
            if img.metadata is None:
                raise ValueError("Missing metadata")


@DataAdapterFactory.register("sklearn")
class SklearnDataAdapter(BaseDataAdapter):
    """DataAdapter for sklearn compatible models"""

    def prepare(self) -> None:
        X = np.array([img.features.values() for img in self.images])  # type: ignore
        if self.label:
            y = np.array([self._convert_tirads(img.metadata["tirads"]) for img in self.images])  # type: ignore
        else:
            y = None
        self._data = (X, y)
        self._preprocessed = True


class PytorchDataAdapter(BaseDataAdapter):
    dataset_cls: Optional[Type[CustomDataset]] = None

    def __init__(self, batch_size: int, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.transforms: Compose = Compose(
            [RandFlip(spatial_axis=1, prob=0.5), RandGaussianNoise(prob=0.2, mean=0.0, std=0.1)]
        )

    def prepare(self) -> None:
        if self.dataset_cls is None:
            raise ValueError("Missing dataset")
        dataset = self.dataset_cls(images=self.images, transform=self.transforms, label=self.label)
        self._data = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)


@DataAdapterFactory.register("image")
class ImageDataAdapter(PytorchDataAdapter):
    dataset_cls = ImageDataset


@DataAdapterFactory.register("hybrid")
class PytorchHybridDataAdapter(PytorchDataAdapter):
    dataset_cls = HybridDataset
