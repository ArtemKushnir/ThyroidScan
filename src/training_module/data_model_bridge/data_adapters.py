import abc
from typing import Any

import numpy as np
from monai.transforms import Compose, RandFlip, RandGaussianNoise
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from src.image_data.image_data import Image
from src.training_module.data_model_bridge.image_dataset import ImageDataset


class BaseDataAdapter(abc.ABC):
    """Convert list[Images] to the desired format
    Splitting the dataset into train and test
    """

    def __init__(self, images: list[Image], is_bin_classification: bool = True, test_size: float = 0.2) -> None:
        self.images: list[Image] = images
        self.test_size: float = test_size
        self._is_bin_classification: bool = is_bin_classification
        self._preprocessed: bool = False
        self._train_data: Any = None
        self._test_data: Any = None
        self._check_constraint_images()
        self._convert_image_labels()

    @abc.abstractmethod
    def prepare(self) -> None:
        pass

    @abc.abstractmethod
    def get_train_data(self) -> Any:
        pass

    @abc.abstractmethod
    def get_test_data(self) -> Any:
        pass

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


class SklearnDataAdapter(BaseDataAdapter):
    """DataAdapter for sklearn compatible models"""

    def prepare(self) -> None:
        X = np.array([img.features.values() for img in self.images])  # type: ignore
        y = np.array([self._convert_tirads(img.metadata["tirads"]) for img in self.images])  # type: ignore

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, stratify=y)

        scaler = StandardScaler()
        scaler.fit(X_train)
        self._train_data = (scaler.transform(X_train), y_train)
        self._test_data = (scaler.transform(X_test), y_test)
        self._preprocessed = True

    def get_train_data(self) -> Any:
        self._is_prepare()
        return self._train_data

    def get_test_data(self) -> Any:
        self._is_prepare()
        return self._test_data


class PytorchDataAdapter(BaseDataAdapter):
    """DataAdapter for pytorch compatible models"""

    def __init__(self, batch_size: int = 8, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.batch_size: int = batch_size
        self.transforms: Compose = Compose(
            [RandFlip(spatial_axis=1, prob=0.5), RandGaussianNoise(prob=0.2, mean=0.0, std=0.1)]
        )

    def prepare(self) -> None:
        train_idx, test_idx = train_test_split(
            range(len(self.images)), test_size=self.test_size, stratify=[img.metadata["tirads"] for img in self.images]  # type: ignore
        )

        self._train_data = ImageDataset(images=[self.images[i] for i in train_idx], transform=self.transforms)

        self._test_data = ImageDataset(
            images=[self.images[i] for i in test_idx],
        )

    def get_train_data(self) -> Any:
        return DataLoader(dataset=self._train_data, batch_size=self.batch_size, shuffle=True)

    def get_test_data(self) -> Any:
        return DataLoader(
            dataset=self._test_data,
            batch_size=self.batch_size,
        )


class DataAdapterFactory:
    @staticmethod
    def create(adapter_type: str, **kwargs: Any) -> BaseDataAdapter:
        adapters = {
            "sklearn": SklearnDataAdapter,
            "pytorch": PytorchDataAdapter,
        }
        return adapters[adapter_type](**kwargs)
