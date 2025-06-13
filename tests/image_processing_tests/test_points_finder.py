import numpy as np
import pytest

from src.image_processing.segmentation.algorithms.wave_algorithm.points_finder import WavePointFinder


class TestWavePointFinder:
    @pytest.fixture
    def sample_image(self):
        img = np.zeros((200, 200), dtype=np.uint8)

        img[50:150, 50:150] = 255

        for i in range(100, 200):
            img[i, 100:200] = i - 100

        return img

    @pytest.fixture
    def uniform_image(self):
        return np.full((100, 100), 100, dtype=np.uint8)

    def test_init_validation(self):
        with pytest.raises(ValueError):
            WavePointFinder(points_number=0)
        with pytest.raises(ValueError):
            WavePointFinder(window_size=0)
        with pytest.raises(ValueError):
            WavePointFinder(step=0)

        finder = WavePointFinder(points_number=10, window_size=50, step=25)
        assert finder.points_number == 10
        assert finder.window_size == 50
        assert finder.step == 25

    def test_image_validation(self):
        finder = WavePointFinder()

        with pytest.raises(ValueError):
            finder.find_points(np.zeros((10, 10, 3)))

        try:
            finder.find_points(np.zeros((10, 10)))
        except ValueError:
            pytest.fail("Valid 2D image raised ValueError")

    def test_find_points_on_uniform_image(self):
        uniform_image = np.full((100, 100), 100, dtype=np.uint8)

        finder = WavePointFinder(points_number=5)
        points = finder.find_points(uniform_image)

        assert len(points) == 1
        assert points[0] == (0, 0)

    def test_find_points_on_heterogeneous_image(self, sample_image):
        finder = WavePointFinder(points_number=3, window_size=50, step=25)
        points = finder.find_points(sample_image)

        assert len(points) == 3

        for x, y in points:
            assert (50 <= x < 150 and 50 <= y < 150) or (100 <= x < 200 and 100 <= y < 200)

    def test_points_number_limit(self, sample_image):
        for n in [1, 5, 10]:
            finder = WavePointFinder(points_number=n)
            points = finder.find_points(sample_image)
            assert len(points) == n

    def test_window_coverage(self, sample_image):
        finder = WavePointFinder(window_size=100, step=50)
        points = finder.find_points(sample_image)

        found_areas = set()
        for x, y in points:
            if x < 100 and y < 100:
                found_areas.add("top-left")
            elif x >= 100 and y < 100:
                found_areas.add("top-right")
            elif x < 100 and y >= 100:
                found_areas.add("bottom-left")
            else:
                found_areas.add("bottom-right")

        assert len(found_areas) >= 3

    def test_statistics_calculation(self):
        finder = WavePointFinder()

        test_region = np.array([[1, 2], [3, 4]], dtype=np.uint8)

        mean, var, std = finder._calculate_region_stats(test_region)
        assert mean == 2.5
        assert var == 1.25
        assert std == pytest.approx(1.11803, rel=1e-5)

    def test_edge_case_window_sizes(self, sample_image):
        finder = WavePointFinder(window_size=200, step=200)
        points = finder.find_points(sample_image)
        assert len(points) == 1
        assert points[0] == (0, 0)
