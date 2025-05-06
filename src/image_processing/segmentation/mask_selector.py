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

    def __init__(self, mask_number: int = 10, use_knn: bool = True):
        self.mask_number = mask_number
        self.use_knn = use_knn
        self._scaler = StandardScaler()
        self._knn = NearestNeighbors(n_neighbors=1)
        self._reference_features: Optional[NDArray] = None

    def fit(self, images_with_masks: list[Image], target: Any = None) -> "MaskSelector":
        all_features: list[NDArray] = []

        for image in images_with_masks:
            if image.segmented_masks is not None:
                valid_masks = self._validate_masks(image.segmented_masks)
            else:
                raise ValueError("Segmented mask can't be None.")

            image_features = []
            for mask in valid_masks:
                if image.cropped_image is not None and image.true_mask is not None:
                    features = self._extract_features(mask, image.cropped_image)
                    iou = jaccard_score(image.true_mask.ravel(), mask.ravel())
                    image_features.append((iou, features))
                else:
                    raise ValueError("Cropped image and true_mask can't be None.")

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
        if self._reference_features is None:
            raise NotFittedError("Selector must be fitted before calling transform.")

        for image in image_list:
            if image.segmented_masks is not None:
                valid_masks = self._validate_masks(image.segmented_masks)
            else:
                raise ValueError("Segmented mask can't be None.")

            if image.cropped_image is not None:
                features = np.array([self._extract_features(mask, image.cropped_image) for mask in valid_masks])
            else:
                raise ValueError("Cropped image can't be None.")

            scaled_features = self._scaler.transform(features)

            distances, _ = self._knn.kneighbors(scaled_features)

            best_indices = np.argsort(distances.ravel())[: self.mask_number]
            image.segmented_masks = [valid_masks[i] for i in best_indices]

        return image_list

    def _extract_features(self, mask: NDArray, image: NDArray) -> NDArray:
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

    def _validate_masks(self, masks: list[NDArray], min_pixels: int = 15) -> list[NDArray]:
        good_masks = []
        bad_masks = []

        for mask in masks:
            weight = self._get_weights(mask, min_pixels)
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
    def _get_weights(mask: NDArray, min_pixels: int = 15) -> float:
        weight = 0
        MAX_COVERAGE = 0.9

        if not np.any(mask):
            weight += 2

        if np.sum(mask) <= min_pixels:
            weight += 1

        coverage = np.mean(mask)
        if coverage > MAX_COVERAGE:
            weight += 1
        return weight
