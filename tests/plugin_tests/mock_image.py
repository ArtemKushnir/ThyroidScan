from typing import Optional

from numpy.typing import NDArray


class Image:
    def __init__(
        self, image_name: str, image: NDArray, true_mask: Optional[NDArray] = None, metadata: Optional[dict] = None
    ):
        self.name = image_name
        self.org_image = image
        self.metadata = metadata
        self._true_mask = true_mask
        self._cropped_image: Optional[NDArray] = None
        self._segmented_masks: Optional[list[NDArray]] = None
        self._features: Optional[dict] = None

    @property
    def cropped_image(self) -> Optional[NDArray]:
        return self._cropped_image

    @cropped_image.setter
    def cropped_image(self, cropped_image: NDArray) -> None:
        self._cropped_image = cropped_image

    @property
    def segmented_masks(self) -> Optional[list[NDArray]]:
        return self._segmented_masks

    @segmented_masks.setter
    def segmented_masks(self, masks: list[NDArray]) -> None:
        self._segmented_masks = masks

    @property
    def features(self) -> Optional[dict]:
        return self._features

    @features.setter
    def features(self, features: dict) -> None:
        self._features = features
