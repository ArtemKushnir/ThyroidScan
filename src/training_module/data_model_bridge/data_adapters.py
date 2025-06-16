import abc
import importlib
import inspect
import os
import pkgutil
from typing import Any, Callable, Optional, Type, Union

import numpy as np
from monai.transforms import Compose, EnsureChannelFirst, Lambda, Resize
from torch.utils.data import DataLoader

from src.image_data.image_data import Image
from src.training_module.data_model_bridge.pytorch_datasets import HybridDataset, ImageDataset

CustomDataset = Union[HybridDataset, ImageDataset]


class DataAdapterRegistry:
    _registry: dict = {}
    _initialized = False

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

    @classmethod
    def discover_adapters(cls) -> None:
        """
        Discover and register all data adapters in the adapters directory.
        """
        if cls._initialized:
            return

        adapter_pkg = "src.training_module.data_model_bridge"

        try:
            package = importlib.import_module(adapter_pkg)
            package_dir = os.path.dirname(inspect.getfile(package))

            for _, name, is_pkg in pkgutil.iter_modules([package_dir]):
                if name == "__init__":
                    continue

                importlib.import_module(f"{adapter_pkg}.{name}")

                if is_pkg:
                    cls._import_submodules(f"{adapter_pkg}.{name}")

            cls._initialized = True
        except ImportError as e:
            print(f"Error discovering adapters: {e}")

    @classmethod
    def _import_submodules(cls, package_name: str) -> None:
        """
        Recursively import all submodules of a package.
        """
        package = importlib.import_module(package_name)
        if not hasattr(package, "__path__"):
            return

        package_dir = os.path.dirname(inspect.getfile(package))
        for _, name, is_pkg in pkgutil.iter_modules([package_dir]):
            if name == "__init__":
                continue

            module_name = f"{package_name}.{name}"
            importlib.import_module(module_name)

            if is_pkg:
                cls._import_submodules(module_name)


class BaseDataAdapter(abc.ABC):
    """Convert list[images] to the desired format"""

    def __init__(self, images: list[Image], is_bin_classification: bool = True, label: bool = True) -> None:
        self.images: list[Image] = images
        self._is_bin_classification: bool = is_bin_classification
        self._preprocessed: bool = False
        self._data: Any = None
        self.label: bool = label
        self._check_constraint_images()
        # if label:
        #     self._convert_image_labels()

    def prepare(self) -> None:
        self._prepare()
        self._preprocessed = True

    @abc.abstractmethod
    def _prepare(self) -> None:
        pass

    @property
    def data(self) -> Any:
        if not self._preprocessed:
            raise ValueError("Call 'prepare' before receiving the data")
        return self._data

    def _convert_target(self, target: Any) -> int:
        if isinstance(target, int):
            return target
        elif isinstance(target, str):
            target_digits = "".join([ch for ch in target if ch.isdigit()])
            if target_digits == "":
                raise ValueError("'target' must contain a number")
            target_numeric = int(target_digits)
            if target_numeric < 0 or target_numeric > 5:
                raise ValueError("'target' should be between 0 and 5")
            if self._is_bin_classification:
                return 1 if target_numeric >= 4 else 0
            else:
                return target_numeric
        else:
            raise TypeError(f"Unsupportable type {type(target)}")

    def _convert_image_labels(self) -> None:
        for img in self.images:
            img.metadata["target"] = self._convert_target(img.metadata["target"])  # type: ignore

    def _check_constraint_images(self) -> None:
        for img in self.images:
            if img.features is None:
                raise ValueError("Missing features")
            if img.metadata is None:
                raise ValueError("Missing metadata")


@DataAdapterRegistry.register("sklearn")
class SklearnDataAdapter(BaseDataAdapter):
    """DataAdapter for sklearn compatible models"""

    def _prepare(self) -> None:
        X = np.array([list(img.features.values()) for img in self.images])  # type: ignore
        if self.label:
            y = np.array([img.metadata["target"] for img in self.images])  # type: ignore
        else:
            y = None
        self._data = (X, y)
        self._preprocessed = True


class PytorchDataAdapter(BaseDataAdapter):
    dataset_cls: Optional[Type[CustomDataset]] = None

    def __init__(self, batch_size: int = 32, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.transforms = Compose(
            [
                Lambda(lambda img: img[np.newaxis, ...]),
                Resize(spatial_size=(256, 256)),
            ]
        )

    def _prepare(self) -> None:
        if self.dataset_cls is None:
            raise ValueError("Missing dataset")
        dataset = self.dataset_cls(images=self.images, transform=self.transforms, label=self.label)
        self._data = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)


@DataAdapterRegistry.register("image")
class ImageDataAdapter(PytorchDataAdapter):
    dataset_cls = ImageDataset


@DataAdapterRegistry.register("hybrid")
class PytorchHybridDataAdapter(PytorchDataAdapter):
    dataset_cls = HybridDataset
