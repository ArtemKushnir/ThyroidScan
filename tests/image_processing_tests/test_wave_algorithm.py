import numpy as np
import pytest

from src.image_processing.segmentation.algorithms.wave_algorithm.wave_segmenter import WaveSegmenter


class TestWaveSegmenter:
    @pytest.fixture
    def sample_image(self):
        img = np.zeros((100, 100), dtype=np.uint8)
        for i in range(100):
            img[i, :] = i
        return img

    @pytest.fixture
    def uniform_image(self):
        return np.full((50, 50), 100, dtype=np.uint8)

    def test_init_validation(self):
        with pytest.raises(ValueError):
            WaveSegmenter(threshold=-1)
        with pytest.raises(ValueError):
            WaveSegmenter(threshold=256)
        with pytest.raises(ValueError):
            WaveSegmenter(connectivity=5)

        seg = WaveSegmenter(threshold=10, connectivity=8)
        assert seg.threshold == 10
        assert seg.connectivity == 8

    def test_directions(self):
        seg4 = WaveSegmenter(connectivity=4)
        assert len(seg4.directions) == 4
        assert (0, 1) in seg4.directions

        seg8 = WaveSegmenter(connectivity=8)
        assert len(seg8.directions) == 8
        assert (1, 1) in seg8.directions

    def test_input_validation(self, sample_image):
        seg = WaveSegmenter()

        with pytest.raises(ValueError):
            seg.segment(np.zeros((10, 10, 3)), (5, 5))

        with pytest.raises(ValueError):
            seg.segment(sample_image, (-1, 5))
        with pytest.raises(ValueError):
            seg.segment(sample_image, (5, 100))

    def test_segmentation_uniform(self, uniform_image):
        seg = WaveSegmenter(threshold=5)
        mask = seg.segment(uniform_image, (25, 25))

        assert mask.sum() == uniform_image.size * 255
        assert np.array_equal(mask, np.full_like(uniform_image, 255))

    def test_segmentation_gradient(self, sample_image):
        seg = WaveSegmenter(threshold=1)
        start_point = (50, 50)
        mask = seg.segment(sample_image, start_point)

        expected_mask = np.zeros_like(sample_image)
        expected_mask[49:52, :] = 255

        assert np.array_equal(mask, expected_mask)

    def test_segmentation_boundary(self):
        img = np.zeros((10, 10), dtype=np.uint8)
        img[5, 5] = 100

        seg = WaveSegmenter(threshold=10)
        mask = seg.segment(img, (5, 5))

        expected_mask = np.zeros_like(img)
        expected_mask[5, 5] = 255

        assert np.array_equal(mask, expected_mask)

    def test_large_threshold(self, sample_image):
        seg = WaveSegmenter(threshold=200)
        mask = seg.segment(sample_image, (50, 50))

        assert mask.sum() > 0.9 * sample_image.size * 255

    def test_connectivity_difference(self):
        img = np.array([[10, 50, 10], [50, 50, 50], [10, 50, 10]], dtype=np.uint8)

        seg4 = WaveSegmenter(threshold=30, connectivity=4)
        mask4 = seg4.segment(img, (1, 1))

        seg8 = WaveSegmenter(threshold=40, connectivity=8)
        mask8 = seg8.segment(img, (1, 1))

        assert mask4[0, 0] == 0
        assert mask4[2, 2] == 0

        assert mask8[0, 0] == 255
        assert mask8[2, 2] == 255
