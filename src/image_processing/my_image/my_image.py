from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from numpy.typing import NDArray


class Image:
    """
    Image implementation.
    """

    def __init__(self, image_path: str) -> None:
        validated_path = self._path_validation(image_path)
        self.name = validated_path.name.split(".")[0]
        self.org_image = self._image_read(str(validated_path))
        self.cropped_image: Optional[list[NDArray]] = None
        self._crop_points: Optional[NDArray] = None
        self._true_mask: Optional[list[NDArray]] = None
        self.segmented_masks: Optional[list[NDArray]] = None

    @staticmethod
    def _image_read(path: str) -> NDArray:
        image = cv2.imread(path)
        if image is None or len(image) == 0:
            raise ValueError(f"Image file is empty.")
        return image

    @staticmethod
    def _path_validation(path: str) -> Path:
        if not Path(path).exists():
            raise FileNotFoundError(f"File not found. Check image path.")
        return Path(path)

    def get_cropped(self) -> Optional[list[NDArray]]:
        return self.cropped_image

    def set_cropped(self, image: list[NDArray]) -> None:
        self.cropped_image = image

    def get_crop_points(self) -> Optional[NDArray]:
        return self._crop_points

    def set_crop_points(self, points: NDArray) -> None:
        self._crop_points = points

    def get_true_mask(self) -> Optional[list[NDArray]]:
        return self._true_mask

    def set_true_mask(self, mask: list[NDArray]) -> None:
        self._true_mask = mask

    def get_segmented_mask(self) -> Optional[list[NDArray]]:
        return self.segmented_masks

    def set_segmented_mask(self, mask: list[NDArray]) -> None:
        self.segmented_masks = mask

    cropped = property(get_cropped, set_cropped)
    crop_points = property(get_crop_points, set_crop_points)
    true_mask = property(get_true_mask, set_true_mask)
    masks = property(get_segmented_mask, set_segmented_mask)


if __name__ == "__main__":
    my_image = Image("../images/images/106_4.png")
    print(my_image.name)
    my_image.cropped = np.array([1, 2, 3])
    print(my_image.cropped)
    print(my_image.org_image)
