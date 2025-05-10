from typing import Any

from sklearn.base import BaseEstimator, TransformerMixin

from src.image_data.image_data import Image
from src.image_processing.segmentation.algorithms.segmentation_interfaces import PointFinder, SegmentationAlgorithm


class ImageSegmenter(BaseEstimator, TransformerMixin):

    def __init__(self, algorithm: SegmentationAlgorithm, point_finder: PointFinder) -> None:
        self.algorithm = algorithm
        self.point_finder = point_finder

    def fit(self, image_list: list[Image], target: Any = None) -> "ImageSegmenter":
        return self

    def transform(self, image_list: list[Image]) -> list[Image]:
        if not isinstance(image_list, list):
            raise TypeError("A list of images was expected.")

        for image in image_list:
            if image.cropped_image is not None:
                points = self.point_finder.find_points(image.cropped_image)
                image.segmented_masks = [self.algorithm.segment(image.cropped_image, point) for point in points]
            else:
                raise ValueError("Cropped image can't be None.")
        return image_list
