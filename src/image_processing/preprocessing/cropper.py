from copy import deepcopy
from typing import Any, Optional

import cv2
import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError

from src.image_data.image_data import Image


class Cropper(BaseEstimator, TransformerMixin):
    """
    Crops input images around the largest detected contour and optionally splits double images.
    """

    def __init__(self, crop_radius: int = 10) -> None:
        """
        :param crop_radius: Number of pixels to keep as padding around the detected object (max 50).
        """
        self.crop_radius = self._validate_radius(crop_radius)

    def fit(self, image_list: list[Image], target: Any = None) -> "Cropper":
        """
        Fit method for compatibility with scikit-learn pipelines.

        This method does nothing and simply returns self.

        :param image_list: List of Image objects.
        :param target: Not used.
        :return: Self.
        """
        return self

    def transform(self, image_list: list[Image]) -> list[Image]:
        """
        Applies cropping to each image in the list.

        :param image_list: List of Image objects.
        :return: Image list with cropped image and true mask attributes.
        """
        if not isinstance(image_list, list):
            raise TypeError("A list of images was expected.")

        copy_list = deepcopy(image_list)
        cropped_list = []
        for image in copy_list:
            image = self._validate_image(image)
            cropped_list.extend(self._crop_image(image))

        return cropped_list

    def _crop_image(self, image: Image) -> list[Image]:
        """
        Crops an image around the largest contour and splits if double objects are found.
        """
        gray_image = cv2.cvtColor(image.org_image, cv2.COLOR_BGR2GRAY)
        denoised_image = cv2.medianBlur(gray_image, 3)

        threshold_value, binary_image = cv2.threshold(denoised_image, 5, 255, cv2.THRESH_BINARY)

        contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 10]
        max_contour = max(filtered_contours, key=cv2.contourArea)

        x, y, wight, height = cv2.boundingRect(max_contour)
        crop_points = np.array(
            [
                y + self.crop_radius,
                y + height - self.crop_radius,
                x + self.crop_radius + 5,
                x + wight - self.crop_radius,
            ]
        )
        cropped_image = gray_image[crop_points[0] : crop_points[1], crop_points[2] : crop_points[3]]

        image.crop_points = crop_points
        image.cropped_image = cropped_image

        result = self._check_for_double(cropped_image)
        if result is None:
            if image.true_mask is not None:
                return [self._crop_mask(image)]
            return [image]

        left_part, right_part = result

        left_image = deepcopy(image)
        left_image.cropped_image = left_part

        right_image = deepcopy(image)
        right_image.name = f"{image.name.split('.')[0]}_1.jpg"
        right_image.cropped_image = right_part

        if image.true_mask is not None:
            split_column = left_part.shape[1]
            return [
                self._crop_mask(left_image, True, split_column, "left"),
                self._crop_mask(right_image, True, split_column, "right"),
            ]

        return [left_image, right_image]

    @staticmethod
    def _check_for_double(cropped_image: NDArray) -> Optional[tuple[NDArray, NDArray]]:
        """
        Detects whether the cropped image contains two objects side by side.
        """
        EPS = 25

        sobel_x = cv2.Sobel(cropped_image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_x = np.absolute(sobel_x)
        sobel_x = (255 * sobel_x / np.max(sobel_x)).astype(np.uint8)

        column_edges = np.sum(sobel_x, axis=0)
        peak_arg = np.argmax(column_edges)
        peak_value = column_edges[peak_arg]
        median = np.median(column_edges)

        if peak_value < 2 * median:
            return None
        left_part = cropped_image[:, :peak_arg]
        right_part = cropped_image[:, peak_arg:]

        min_size = cropped_image.shape[1] / 2 - EPS

        if left_part.shape[1] < min_size or right_part.shape[1] < min_size:
            return None

        return left_part, right_part

    @staticmethod
    def _crop_mask(
        image: Image, is_double: bool = False, split_column: Optional[int] = None, image_type: Optional[str] = None
    ) -> Image:
        """
        Crops the ground truth mask corresponding to the cropped image.

        Handles single and double-object cases by splitting the cropped mask
        accordingly.

        :param image: Image object with `true_mask` and `crop_points`.
        :param is_double: Whether the image is split into two parts.
        :param split_column: X-coordinate for vertical split.
        :param image_type: "left" or "right" for double images.
        :return: Image object with cropped true_mask.
        """
        if image.true_mask is None:
            raise ValueError("Image must have true mask.")
        if image.crop_points is None:
            raise ValueError("Image after cropping should have crop points.")

        gray_mask = image.true_mask
        cropped_mask = gray_mask[
            image.crop_points[0] : image.crop_points[1], image.crop_points[2] : image.crop_points[3]
        ]

        if is_double:
            if image_type == "left":
                mask_left = cropped_mask[:, :split_column]
                image.true_mask = mask_left
            else:
                mask_right = cropped_mask[:, split_column:]
                image.true_mask = mask_right
        else:
            image.true_mask = cropped_mask

        return image

    @staticmethod
    def write_cropped_to_file(image: Image) -> None:
        """
        Write cropped image into file.
        """
        if image.cropped_image is None:
            raise NotFittedError("Should use cropper before.")
        cropped_image_path = "cropped_" + image.name
        cv2.imwrite(cropped_image_path, image.cropped_image)

    @staticmethod
    def _validate_radius(radius: int) -> int:
        """
        Validates that the crop radius is an integer in an acceptable range.
        """
        CONST = 50
        if not isinstance(radius, int):
            raise TypeError("Radius must be int.")
        if radius < 0 or radius > CONST:
            raise ValueError("Radius must be positive and <= 50.")
        return radius

    @staticmethod
    def _validate_image(image: Image) -> Image:
        """
        Validates that the input is a non-null Image object.
        """
        if not isinstance(image, Image):
            raise TypeError("Image must be object of Image.")
        if image is None:
            raise ValueError("Image can't be null.")
        return image
