from collections import deque

import numpy as np
from numpy.typing import NDArray

from src.image_processing.segmentation.algorithms.segmentation_interfaces import SegmentationAlgorithm


class WaveSegmenter(SegmentationAlgorithm):
    """
    Wavefront-based image segmentation algorithm that implements the SegmentationAlgorithm interface.
    Expands a region from a starting point based on intensity similarity.
    """

    def __init__(self, threshold: int = 15, connectivity: int = 4) -> None:
        """
        Initialize the wave segmenter.

        :param threshold: Maximum allowed intensity difference between neighboring pixels (0-255).
        :param connectivity: Pixel connectivity mode ('4' for 4-way, '8' for 8-way).
        """
        self._validate_params(threshold, connectivity)
        self.threshold = threshold
        self.connectivity = connectivity
        self.directions = self._get_directions()

    def segment(self, image: NDArray, start_point: tuple[int, int]) -> NDArray:
        """
        Perform image segmentation from the starting point.

        :param image: 2D grayscale image.
        :param start_point: (x,y) coordinate for region seed.
        :return: Segmented mask.
        """

        self._validate_input(image, start_point)

        rows, cols = image.shape
        mask = np.zeros_like(image, dtype=np.uint8)
        visited = np.zeros_like(image, dtype=bool)
        x, y = start_point
        reference_intensity = image[y, x]

        queue = deque([start_point])
        visited[y, x] = True

        while queue:
            x, y = queue.popleft()
            mask[y, x] = 255

            for dx, dy in self.directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < cols and 0 <= ny < rows and not visited[ny, nx]:
                    if abs(int(image[ny, nx]) - int(reference_intensity)) <= self.threshold:
                        queue.append((nx, ny))
                        visited[ny, nx] = True

        return mask

    def _get_directions(self) -> list[tuple[int, int]]:
        """Get expansion directions based on connectivity."""
        four_way = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        return four_way if self.connectivity == 4 else four_way + [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    @staticmethod
    def _validate_params(threshold, connectivity) -> None:
        """Validate constructor parameters."""
        if connectivity not in [4, 8]:
            raise ValueError("Connectivity must be either '4' or '8'.")
        if not 0 <= threshold <= 255:
            raise ValueError("Threshold must be between 0 and 255.")

    @staticmethod
    def _validate_input(image: np.ndarray, start_point: tuple[int, int]) -> None:
        """Validate input image and start point."""
        if len(image.shape) != 2:
            raise ValueError("Input must be a 2D grayscale image.")
        if not (0 <= start_point[0] < image.shape[1] and 0 <= start_point[1] < image.shape[0]):
            raise ValueError("Start point coordinates are out of image bounds.")
