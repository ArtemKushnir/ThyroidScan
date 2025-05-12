from copy import deepcopy
from typing import Any

from sklearn.base import BaseEstimator, TransformerMixin

from src.image_data.image_data import Image
from src.image_processing.segmentation.algorithms.segmentation_interfaces import PointFinder, SegmentationAlgorithm


class ImageSegmenter(BaseEstimator, TransformerMixin):
    """
     A transformer that performs image segmentation based on detected points of interest.

    This class is designed to work as part of a scikit-learn pipeline. It takes a list
    of images, uses a point-finding strategy to locate points of interest within each
    image, and then applies a segmentation algorithm to extract masks around those points.

    The segmentation results are stored in each image's `segmented_masks` attribute.
    """

    def __init__(self, algorithm: SegmentationAlgorithm, point_finder: PointFinder) -> None:
        """
        :param algorithm: Segmentation algorithm that implements the `segment(image, point)` method.
        :param point_finder: Algorithm that implements the `find_points(image)` method to detect points of interest.
        """
        self.algorithm = algorithm
        self.point_finder = point_finder

    def fit(self, image_list: list[Image], target: Any = None) -> "ImageSegmenter":
        """
        Fit method for compatibility with scikit-learn pipelines.

        This method does nothing and simply returns self.

        :param image_list: List of input Image objects.
        :param target: Ignored. Exists for compatibility with scikit-learn.
        :return: Self.
        """
        return self

    def transform(self, image_list: list[Image]) -> list[Image]:
        """
        Apply the segmentation algorithm to each image in the list.

        For each image, detects points of interest using the point finder,
        then generates segmented masks at those points using the segmentation algorithm.

        :param image_list: List of Image objects to transform.
        :return: List of modified Image objects with populated `segmented_masks`.
        """
        if not isinstance(image_list, list):
            raise TypeError("A list of images was expected.")
        copy_list = deepcopy(image_list)

        for image in copy_list:
            if image.cropped_image is not None:
                points = self.point_finder.find_points(image.cropped_image)
                image.segmented_masks = [self.algorithm.segment(image.cropped_image, point) for point in points]
            else:
                raise ValueError("Cropped image can't be None.")
        return copy_list
