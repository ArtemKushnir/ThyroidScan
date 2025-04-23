from typing import Optional

import numpy as np
from numpy.typing import NDArray


class Image:
    """
    A class for storing and managing information about an ultrasound image,
    including its original image, masks, cropping data, features, and associated metadata.
    """

    def __init__(
        self, image_name: str, image: NDArray, true_mask: Optional[NDArray] = None, metadata: Optional[dict] = None
    ) -> None:
        self.name = self.validate_image_name(image_name)
        self.org_image = self.validate_image(image)
        self.metadata = metadata
        self._true_mask = true_mask

        self._cropped_image: Optional[NDArray] = None
        self._crop_points: Optional[NDArray] = None
        self._segmented_masks: Optional[list[NDArray]] = None
        self._features: Optional[dict] = None

    @property
    def cropped_image(self) -> Optional[NDArray]:
        return self._cropped_image

    @cropped_image.setter
    def cropped_image(self, cropped_image: NDArray) -> None:
        if not isinstance(cropped_image, np.ndarray):
            raise TypeError("Cropped image must be a NumPy array.")
        self._cropped_image = cropped_image

    @property
    def crop_points(self) -> Optional[NDArray]:
        return self._crop_points

    @crop_points.setter
    def crop_points(self, points: NDArray) -> None:
        if not isinstance(points, np.ndarray):
            raise TypeError("Crop points must be a NumPy array.")
        self._crop_points = points

    @property
    def true_mask(self) -> Optional[NDArray]:
        return self._true_mask

    @true_mask.setter
    def true_mask(self, mask: NDArray) -> None:
        if not isinstance(mask, np.ndarray):
            raise TypeError("True mask must be a NumPy array.")
        self._true_mask = mask

    @property
    def segmented_masks(self) -> Optional[list[NDArray]]:
        return self._segmented_masks

    @segmented_masks.setter
    def segmented_masks(self, masks: list[NDArray]) -> None:
        if not isinstance(masks, list) or not all(isinstance(m, np.ndarray) for m in masks):
            raise TypeError("Segmented masks must be a list of NumPy arrays.")
        self._segmented_masks = masks

    @property
    def features(self) -> Optional[dict]:
        return self._features

    @features.setter
    def features(self, features: dict) -> None:
        if not isinstance(features, dict) or features is None:
            raise TypeError("Features must be a not null dictionary.")
        self._features = features

    def __repr__(self) -> str:
        shape = self.org_image.shape if self.org_image is not None else "None"
        metadata_keys = list(self.metadata.keys()) if self.metadata else []
        return f"Image(name={self.name}, shape={shape}, metadata_keys={metadata_keys})"

    @staticmethod
    def validate_image(image: NDArray) -> NDArray:
        if not isinstance(image, np.ndarray) or image is None:
            raise TypeError("Image mast be NumPy array.")
        return image

    @staticmethod
    def validate_image_name(image_name: str) -> str:
        if not isinstance(image_name, str) or str is None:
            raise TypeError("Image name mast be string.")
        return image_name
