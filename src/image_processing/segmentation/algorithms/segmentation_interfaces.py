from abc import ABC, abstractmethod

from numpy.typing import NDArray


class PointFinder(ABC):
    """Abstract base class for seed point detectors."""

    @abstractmethod
    def find_points(self, image: NDArray) -> list[tuple]:
        pass


class SegmentationAlgorithm(ABC):
    """Abstract base class for image segmentation algorithms."""

    @abstractmethod
    def segment(self, image: NDArray, start_point: tuple) -> NDArray:
        pass
