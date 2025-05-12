from copy import deepcopy
from typing import Any

import cv2
import numpy as np
import pywt
from numpy.typing import NDArray
from scipy.stats import kurtosis, skew
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.measure import shannon_entropy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError

from src.image_data.image_data import Image


class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        lbp_radius: int = 2,
        lbp_points: int = 8,
        glcm_distances: list[int] = [1],
        glcm_angles: list[float] = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        hist_bins: int = 5,
        quantiles: list[float] = [0.1, 0.25, 0.5, 0.75, 0.9],
        wavelet_level: int = 3,
    ):
        self.lbp_radius = lbp_radius
        self.lbp_points = lbp_points
        self.glcm_distances = glcm_distances
        self.glcm_angles = glcm_angles
        self.hist_bins = hist_bins
        self.quantiles = quantiles
        self.wavelet_level = wavelet_level

    def fit(self, image_list: list[Image], target: Any = None) -> "FeatureExtractor":
        return self

    def transform(self, image_list: list[Image]) -> list[Image]:
        """Extract features for each image and its masks."""
        copy_list = deepcopy(image_list)

        for image in copy_list:
            if image.cropped_image is None:
                raise NotFittedError("Should use Cropper before.")

            features = {}
            features.update(self._prefix_features(self._extract_common_features(image.cropped_image), prefix="image"))
            features.update(
                self._prefix_features(self._calc_statistic_features(image.cropped_image.ravel()), prefix="image")
            )

            masks = image.segmented_masks
            if masks is None:
                raise NotFittedError("Should use MaskSelector before.")

            for i, mask in enumerate(masks):
                features.update(
                    self._prefix_features(self._extract_mask_features(mask, image.cropped_image), prefix=f"mask{i}")
                )

            image.features = features

        return copy_list

    def _extract_common_features(self, region: NDArray) -> dict[str, float]:
        """Extract common features for any region (image or mask area)."""
        if region is None:
            raise ValueError("Region can't be None.")

        features = {}
        features.update(self._calc_contour_features(region))

        if self.lbp_points > 0:
            features.update(self._calc_lbp_features(region))

        if len(self.glcm_distances) > 0:
            features.update(self._calc_glcm_features(region))

        if self.wavelet_level > 0:
            features.update(self._calc_wavelet_features(region))

        return features

    def _calc_statistic_features(self, region: NDArray) -> dict[str, float]:
        """Calculates statistic features."""
        if region is None:
            raise ValueError("Region can't be None.")
        if len(region.shape) != 1:
            raise ValueError("Region dimension must be 1.")

        features = {
            "mean": float(np.mean(region)),
            "std": float(np.std(region)),
            "skewness": float(skew(region)),
            "kurtosis": float(kurtosis(region)),
            "entropy": float(shannon_entropy(region)),
        }

        features.update(self._calc_quantile_features(region))
        features.update(self._calc_histogram_features(region))

        return features

    def _extract_mask_features(self, mask: NDArray, image: NDArray) -> dict[str, float]:
        """Extracts features of a mask and its corresponding region."""
        binary_mask = (mask > 0).astype(np.uint8)
        masked_region = image[binary_mask > 0]

        features = self._calc_geometric_features(binary_mask)
        features.update(self._calc_statistic_features(masked_region))
        features.update(self._extract_common_features(image * binary_mask))

        return features

    def _calc_quantile_features(self, region: NDArray) -> dict[str, float]:
        """Calculates quantiles."""
        if region is None:
            raise ValueError("Region can't be None.")

        quantile_values = np.quantile(region, self.quantiles)

        return {f"q_{int(q * 100)}": float(val) for q, val in zip(self.quantiles, quantile_values)}

    def _calc_histogram_features(self, region: NDArray) -> dict[str, float]:
        """Calculates the histogram of intensities."""
        if region is None:
            raise ValueError("Region can't be None.")

        hist = np.histogram(region, bins=self.hist_bins, range=(0, 1))[0]
        hist_norm = hist / hist.sum() if hist.sum() > 0 else hist

        return {f"hist_bin_{i}": float(val) for i, val in enumerate(hist_norm)}

    def _calc_lbp_features(self, region: NDArray) -> dict[str, float]:
        """Calculates Local Binary Pattern features."""
        if region is None:
            raise ValueError("Region can't be None.")

        lbp = local_binary_pattern(region, P=self.lbp_points, R=self.lbp_radius, method="uniform")
        hist, _ = np.histogram(lbp, bins=10, range=(0, 10))
        hist_norm = hist / hist.sum() if hist.sum() > 0 else hist

        return {f"lbp_{i}": float(val) for i, val in enumerate(hist_norm)}

    def _calc_glcm_features(self, region: NDArray) -> dict[str, float]:
        """Calculates GLCM features."""
        if region is None:
            raise ValueError("Region can't be None.")

        glcm = graycomatrix(
            region.astype(np.uint8),
            distances=self.glcm_distances,
            angles=self.glcm_angles,
            levels=256,
            symmetric=True,
        )

        return {
            "glcm_contrast": float(np.mean(graycoprops(glcm, "contrast"))),
            "glcm_energy": float(np.mean(graycoprops(glcm, "energy"))),
            "glcm_homogeneity": float(np.mean(graycoprops(glcm, "homogeneity"))),
            "glcm_correlation": float(np.mean(graycoprops(glcm, "correlation"))),
        }

    def _calc_wavelet_features(self, region: NDArray) -> dict[str, float]:
        """Calculate wavelet transform statistics."""
        features = {}
        coeffs = pywt.wavedec2(region, "haar", level=self.wavelet_level)

        cA = coeffs[0]
        features.update(
            {
                "wavelet_approx_mean": float(np.mean(cA)),
                "wavelet_approx_std": float(np.std(cA)),
                "wavelet_approx_energy": float(np.sum(cA**2)),
                "wavelet_approx_min": float(np.min(cA)),
                "wavelet_approx_max": float(np.max(cA)),
            }
        )

        for i in range(1, len(coeffs)):
            cH, cV, cD = coeffs[i]
            prefix = f"wavelet_level{self.wavelet_level - i + 1}_"

            features.update(
                {
                    f"{prefix}horiz_mean": float(np.mean(cH)),
                    f"{prefix}horiz_std": float(np.std(cH)),
                    f"{prefix}vert_mean": float(np.mean(cV)),
                    f"{prefix}vert_std": float(np.std(cV)),
                    f"{prefix}diag_mean": float(np.mean(cD)),
                    f"{prefix}diag_std": float(np.std(cD)),
                    f"{prefix}horiz_energy": float(np.sum(cH**2)),
                    f"{prefix}vert_energy": float(np.sum(cV**2)),
                    f"{prefix}diag_energy": float(np.sum(cD**2)),
                }
            )

        return features

    @staticmethod
    def _prefix_features(features: dict[str, float], prefix: str) -> dict[str, float]:
        """Adds a prefix to the keys of the feature dictionary."""
        return {f"{prefix}_{k}": v for k, v in features.items()}

    @staticmethod
    def _calc_contour_features(region: NDArray) -> dict[str, float]:
        """Calculates contour features (Laplacian and Sobel)."""
        laplacian = cv2.Laplacian(region.astype(np.float32), cv2.CV_32F)
        sobel_x = cv2.Sobel(region, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(region, cv2.CV_32F, 0, 1, ksize=3)
        sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)

        return {
            "laplacian_mean": float(np.mean(laplacian)),
            "laplacian_std": float(np.std(laplacian)),
            "sobel_magnitude": float(np.mean(sobel_mag)),
        }

    @staticmethod
    def _calc_geometric_features(mask: NDArray) -> dict[str, float]:
        """Calculates geometric features of a mask."""
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if not contours:
            return {
                "area": 0.0,
                "perimeter": 0.0,
                "compactness": 0.0,
                "aspect_ratio": 0.0,
                "circularity": 0.0,
                "convexity": 0.0,
            }

        cnt = contours[0]
        perimeter = cv2.arcLength(cnt, True)
        area = cv2.contourArea(cnt)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)

        return {
            "area": float(area),
            "perimeter": float(perimeter),
            "compactness": (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0.0,
            "aspect_ratio": float(cv2.boundingRect(cnt)[2]) / max(cv2.boundingRect(cnt)[3], 1e-5),
            "convexity": float(hull_area / area) if area > 0 else 0.0,
        }
