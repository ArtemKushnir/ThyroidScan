import cv2
import numpy as np
from numpy.typing import NDArray

from src.image_processing.segmentation.algorithms.segmentation_interfaces import PointFinder


class WavePointFinder(PointFinder):
    """
    Detects points in regions with maximum statistical deviation from global image characteristics.

    This implementation slides a window across the image and selects points where local statistics
    (mean, variance, standard deviation) differ most from the global image characteristics.
    """

    def __init__(self, points_number: int = 20, window_size: int = 80, step: int = 40) -> None:
        """
        Initialize the point finder.

        :param points_number: Number of points to return.
        :param window_size: Size of the sliding window (pixels).
        :param step: Step size for sliding window movement (pixels).
        """
        self._validate_params(points_number, window_size, step)

        self.points_number = points_number
        self.window_size = window_size
        self.step = step

    def find_points(self, image: NDArray) -> list[tuple[int, int]]:
        """
        Find points with maximum statistical deviation.

        :param image: Input grayscale image (2D numpy array).
        :return: List of (x,y) coordinates ordered by deviation significance.
        """
        self._validate_image(image)

        global_mean, global_variance, global_std_dev = self._calculate_region_stats(image)
        global_params = np.array([global_mean, global_variance, global_std_dev])

        height, width = image.shape
        deviations = []

        for y in range(0, height - self.window_size + 1, self.step):
            for x in range(0, width - self.window_size + 1, self.step):
                window = image[y : y + self.window_size, x : x + self.window_size]
                local_params = np.array(self._calculate_region_stats(window))
                deviation = np.linalg.norm(local_params - global_params)
                deviations.append(((x, y), deviation))

        deviations.sort(key=lambda x: x[1], reverse=True)

        return [point for point, _ in deviations[: self.points_number]]

    @staticmethod
    def visualize_points(
        image: NDArray, points: list[tuple[int, int]], color: tuple[int, int, int] = (0, 255, 0), radius: int = 3
    ) -> NDArray:
        """
        Draw detected points on the original image.
        """
        output_image = image.copy()
        for x, y in points:
            cv2.circle(output_image, (x, y), radius, color, -1)
        return output_image

    @staticmethod
    def _calculate_region_stats(region: NDArray) -> tuple[float, float, float]:
        """Calculate basic statistics for an image region."""
        return np.mean(region), np.var(region), np.std(region)

    @staticmethod
    def _validate_image(image: NDArray) -> None:
        """Checks input image."""
        if len(image.shape) != 2:
            raise ValueError("Input must be a 2D grayscale image.")

    @staticmethod
    def _validate_params(points_number: int, window_size: int, step: int) -> None:
        if not isinstance(points_number, int):
            raise TypeError("Points number must be int.")
        if not isinstance(window_size, int):
            raise TypeError("Window size must be int.")
        if not isinstance(step, int):
            raise TypeError("Step must be int.")
        if points_number <= 0:
            raise ValueError("Points number must be positive.")
        if window_size <= 0:
            raise ValueError("Window size must be positive.")
        if step <= 0:
            raise ValueError("Step must be positive.")
