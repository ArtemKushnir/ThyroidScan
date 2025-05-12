from copy import deepcopy
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray
from skimage.measure import regionprops
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.metrics import jaccard_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.image_data.image_data import Image


class MaskSelector(BaseEstimator, TransformerMixin):
    """
    Selects the best segmentation masks based on shape and intensity features.

    This class ranks segmentation masks for each image using basic geometric
    and intensity-based features. Optionally, it can use K-Nearest Neighbors
    to select masks that are most similar to a reference set of top masks.
    """

    def __init__(self, mask_number: int = 10, use_knn: bool = True):
        """
        :param mask_number: Number of masks to retain for each image.
        :param use_knn: Whether to use KNN-based similarity to select masks.
        """
        self.mask_number = mask_number
        self.use_knn = use_knn
        self._scaler = StandardScaler()
        self._knn = NearestNeighbors(n_neighbors=1)
        self._reference_features: Optional[NDArray] = None

    def fit(self, images_with_masks: list[Image], target: Any = None) -> "MaskSelector":
        """
        Fits the mask selector on a list of segmented images.

        For each mask in each image, extracts features and computes the IoU
        (Jaccard score) with the true mask. The top-N masks with highest IoU
        are used to build the reference set for KNN matching.

        :param images_with_masks: List of Image objects with segmented_masks and true_mask.
        :param target: Ignored. Exists for compatibility with scikit-learn.
        :return: Self.
        """
        all_features: list[NDArray] = []

        for image in images_with_masks:
            if image.segmented_masks is None:
                raise NotFittedError("Use ImageSegmenter before.")
            if image.cropped_image is None:
                raise NotFittedError("Use Cropper before.")

            valid_masks = self._validate_masks(image.segmented_masks)

            image_features = []
            for mask in valid_masks:
                if image.true_mask is None:
                    raise ValueError("True_mask can't be None. Use XMLHandler before.")

                features = self._extract_features(mask, image.cropped_image)
                iou = jaccard_score(image.true_mask.ravel(), mask.ravel(), pos_label=255)
                image_features.append((iou, features))

            image_features.sort(reverse=True, key=lambda x: x[0])
            top_features = np.array([x[1] for x in image_features[: self.mask_number]])
            all_features.extend(top_features)

        self._reference_features = np.array(all_features)

        if self.use_knn:
            self._scaler.fit(self._reference_features)
            scaled_features = self._scaler.transform(self._reference_features)
            self._knn.fit(scaled_features)

        return self

    def transform(self, image_list: list[Image]) -> list[Image]:
        """
        Selects the best segmentation masks for each image.

        :param image_list: List of Image objects with segmented_masks.
        :return: List of images with updated segmented_masks.
        """
        if self._reference_features is None:
            raise NotFittedError("Selector must be fitted before calling transform.")

        copy_list = deepcopy(image_list)

        for image in copy_list:
            if image.segmented_masks is None:
                raise NotFittedError("Use ImageSegmenter before.")
            if image.cropped_image is None:
                raise NotFittedError("Use Cropper before.")

            valid_masks = self._validate_masks(image.segmented_masks)

            features = np.array([self._extract_features(mask, image.cropped_image) for mask in valid_masks])
            scaled_features = self._scaler.transform(features)
            distances, _ = self._knn.kneighbors(scaled_features)

            best_indices = np.argsort(distances.ravel())[: self.mask_number]
            image.segmented_masks = [valid_masks[i] for i in best_indices]

        return copy_list

    def _validate_masks(self, masks: list[NDArray], min_pixels: int = 16) -> list[NDArray]:
        """
        Filters masks based on quality heuristics.

        Selects masks with low "badness" weight. If not enough good masks are found,
        fills the result with the least bad ones.

        :param masks: List of mask arrays to evaluate.
        :param min_pixels: Minimum number of pixels required to consider a mask valid.
        :return: List of selected masks.
        """
        good_masks = []
        bad_masks = []

        for mask in masks:
            binary_mask = (mask > 0).astype(np.uint8)
            weight = self._get_weights(binary_mask, min_pixels)
            if weight == 0:
                good_masks.append((weight, mask))
            else:
                bad_masks.append((weight, mask))

        if len(good_masks) >= self.mask_number:
            return [mask for _, mask in good_masks]

        bad_masks.sort(key=lambda x: x[0])

        add_mask_num = self.mask_number - len(good_masks)
        good_masks += [(weight, mask) for weight, mask in bad_masks[:add_mask_num]]

        return [mask for _, mask in good_masks]

    @staticmethod
    def _extract_features(mask: NDArray, image: NDArray) -> NDArray:
        """
        Extracts shape and intensity-based features from a mask.

        :param mask: Binary mask array.
        :param image: Corresponding image array.
        :return: 1D NumPy array of features.
        """
        features: list[float] = []
        binary_mask = (mask > 0).astype(np.uint8)

        area = np.sum(binary_mask)
        features.append(float(area))

        props = regionprops(binary_mask)
        if props:
            perimeter = props[0].perimeter
            compactness = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0.0
            features.append(compactness)

            bbox = props[0].bbox
            aspect_ratio = (bbox[3] - bbox[1]) / (bbox[2] - bbox[0]) if (bbox[2] - bbox[0]) > 0 else 0
            features.append(aspect_ratio)

            eccentricity = props[0].eccentricity
            features.append(eccentricity)
        else:
            features.extend([0, 0, 0])

        masked_pixels = image[binary_mask > 0]

        if len(masked_pixels) > 0:
            mean_intensity = np.mean(masked_pixels)
            std_intensity = np.std(masked_pixels)
            features.extend([mean_intensity, std_intensity])

            q25, q50, q75 = np.percentile(masked_pixels, [25, 50, 75])
            features.extend([q25, q50, q75])

            hist, _ = np.histogram(masked_pixels, bins=5)
            features.extend(hist / hist.sum())
        else:
            features.extend([0] * 10)

        return np.array(features)

    @staticmethod
    def _get_weights(mask: NDArray, min_pixels: int = 16, max_coverage: float = 0.43) -> float:
        """
        Calculates a heuristic weight for a mask based on pixel count and coverage.

        :param mask: Binary mask array.
        :param min_pixels: Minimum pixel threshold.
        :param max_coverage: Maximum fraction of mask coverage before penalty.
        :return: Weight value.
        """
        weight = 0.0

        if not np.any(mask):
            weight += min_pixels + 3

        pixel_num = np.sum(mask)
        if pixel_num <= min_pixels:
            weight -= pixel_num

        coverage = np.mean(mask)
        if coverage > max_coverage:
            weight -= min_pixels / 2

        return weight
