from unittest.mock import MagicMock

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from src.image_data.image_data import Image
from src.image_processing.segmentation.algorithms.segmentation_interfaces import PointFinder, SegmentationAlgorithm
from src.image_processing.segmentation.image_segmenter import ImageSegmenter


class TestImageSegmenter:
    @pytest.fixture
    def sample_image(self):
        gray_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        img = Image("test_image.jpg", np.random.rand(100, 100, 3))
        img.cropped_image = gray_image
        return img

    @pytest.fixture
    def mock_algorithm(self):
        mock = MagicMock(spec=SegmentationAlgorithm)
        mock.segment.return_value = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
        return mock

    @pytest.fixture
    def mock_point_finder(self):
        mock = MagicMock(spec=PointFinder)
        mock.find_points.return_value = [(25, 25), (75, 75)]
        return mock

    def test_init(self, mock_algorithm, mock_point_finder):
        segmenter = ImageSegmenter(algorithm=mock_algorithm, point_finder=mock_point_finder)
        assert segmenter.algorithm == mock_algorithm
        assert segmenter.point_finder == mock_point_finder

    def test_fit(self, mock_algorithm, mock_point_finder, sample_image):
        segmenter = ImageSegmenter(mock_algorithm, mock_point_finder)
        result = segmenter.fit([sample_image])
        assert result is segmenter

    def test_transform_input_validation(self, mock_algorithm, mock_point_finder):
        segmenter = ImageSegmenter(mock_algorithm, mock_point_finder)
        with pytest.raises(TypeError):
            segmenter.transform("not_a_list")
        with pytest.raises(TypeError):
            segmenter.transform([123])

    def test_transform_without_cropping(self, mock_algorithm, mock_point_finder):
        segmenter = ImageSegmenter(mock_algorithm, mock_point_finder)
        img = Image("test.jpg", np.random.rand(100, 100, 3))
        with pytest.raises(NotFittedError):
            segmenter.transform([img])

    def test_transform_success(self, mock_algorithm, mock_point_finder, sample_image):
        test_points = [(30, 30), (70, 70)]
        mock_point_finder.find_points.return_value = test_points

        mask1 = np.zeros_like(sample_image.cropped_image, dtype=np.uint8)
        mask1[25:35, 25:35] = 255
        mask2 = np.zeros_like(sample_image.cropped_image, dtype=np.uint8)
        mask2[65:75, 65:75] = 255
        mock_algorithm.segment.side_effect = [mask1, mask2]

        segmenter = ImageSegmenter(mock_algorithm, mock_point_finder)
        result = segmenter.transform([sample_image])

        mock_point_finder.find_points.assert_called_once()
        args, _ = mock_point_finder.find_points.call_args
        assert len(args) == 1
        assert np.array_equal(args[0], sample_image.cropped_image)

        assert mock_algorithm.segment.call_count == 2
        segment_calls = mock_algorithm.segment.call_args_list

        for i, call in enumerate(segment_calls):
            args, _ = call
            assert len(args) == 2
            assert np.array_equal(args[0], sample_image.cropped_image)
            assert args[1] == test_points[i]

        assert len(result) == 1
        processed_image = result[0]
        assert len(processed_image.segmented_masks) == 2
        assert np.array_equal(processed_image.segmented_masks[0], mask1)
        assert np.array_equal(processed_image.segmented_masks[1], mask2)

    def test_transform_empty_points(self, mock_algorithm, mock_point_finder, sample_image):
        mock_point_finder.find_points.return_value = []
        segmenter = ImageSegmenter(mock_algorithm, mock_point_finder)
        result = segmenter.transform([sample_image])

        assert len(result) == 1
        assert result[0].segmented_masks == []

    def test_deepcopy_usage(self, mock_algorithm, mock_point_finder, sample_image):
        segmenter = ImageSegmenter(mock_algorithm, mock_point_finder)
        original_list = [sample_image]
        result = segmenter.transform(original_list)

        assert result is not original_list
        assert result[0] is not original_list[0]
        assert result[0].name == original_list[0].name

    def test_algorithm_not_called_when_no_points(self, mock_algorithm, mock_point_finder, sample_image):
        mock_point_finder.find_points.return_value = []
        segmenter = ImageSegmenter(mock_algorithm, mock_point_finder)
        segmenter.transform([sample_image])

        mock_algorithm.segment.assert_not_called()
