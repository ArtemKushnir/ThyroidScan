from copy import deepcopy

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from src.image_data.image_data import Image
from src.image_processing.segmentation.mask_selector import MaskSelector


class TestMaskSelector:
    @pytest.fixture
    def sample_image(self):
        img = Image("test.jpg", np.random.rand(100, 100, 3))
        img.cropped_image = np.random.rand(100, 100)

        mask1 = np.zeros((100, 100), dtype=np.uint8)
        mask1[20:40, 20:40] = 255
        mask2 = np.zeros((100, 100), dtype=np.uint8)
        mask2[30:70, 30:70] = 255
        mask3 = np.zeros((100, 100), dtype=np.uint8)
        mask3[10:90, 10:90] = 255

        img.segmented_masks = [mask1, mask2, mask3]
        img.true_mask = mask2
        return img

    @pytest.fixture
    def bad_image(self):
        return Image("bad.jpg", np.random.rand(100, 100, 3))

    def test_init(self):
        selector = MaskSelector(mask_number=5)
        assert selector.mask_number == 5
        assert selector._reference_features is None

    def test_fit_validation(self, bad_image):
        selector = MaskSelector()

        bad_img = deepcopy(bad_image)
        with pytest.raises(NotFittedError):
            selector.fit([bad_img])

        bad_img = deepcopy(bad_image)
        bad_img.cropped_image = bad_image.org_image
        with pytest.raises(NotFittedError):
            selector.fit([bad_img])

        bad_img = deepcopy(bad_image)
        bad_img.cropped_image = bad_image.org_image
        bad_img.segmented_masks = [np.array([0, 1]), np.array([1, 0])]
        with pytest.raises(ValueError):
            selector.fit([bad_img])

    def test_fit_success(self, sample_image):
        selector = MaskSelector(mask_number=2)
        selector.fit([sample_image])

        assert selector._reference_features is not None
        assert len(selector._reference_features) == 2

    def test_transform_before_fit(self):
        selector = MaskSelector()
        with pytest.raises(NotFittedError):
            selector.transform([Image("test.jpg", np.zeros((10, 10)))])

    def test_transform_success(self, sample_image):
        selector = MaskSelector(mask_number=2)
        selector.fit([sample_image])

        test_img = deepcopy(sample_image)
        test_img.segmented_masks = [
            np.zeros((100, 100), dtype=np.uint8),
            sample_image.true_mask,
            np.ones((100, 100), dtype=np.uint8),
        ]

        result = selector.transform([test_img])
        assert len(result[0].segmented_masks) == 2

    def test_validate_masks(self, sample_image):
        selector = MaskSelector(mask_number=2)

        good_masks = selector._validate_masks(sample_image.segmented_masks)
        assert len(good_masks) >= 2

    def test_extract_features(self, sample_image):
        selector = MaskSelector()
        features = selector._extract_features(sample_image.segmented_masks[0], sample_image.cropped_image)

        assert isinstance(features, np.ndarray)
        assert features.shape[0] > 5

        empty_mask = np.zeros((100, 100), dtype=np.uint8)
        empty_features = selector._extract_features(empty_mask, sample_image.cropped_image)
        assert np.any(empty_features == 0)

    def test_get_weights(self):
        selector = MaskSelector()

        empty_mask = np.zeros((100, 100), dtype=np.uint8)
        assert selector._get_weights(empty_mask) > 0

        small_mask = np.zeros((100, 100), dtype=np.uint8)
        small_mask[5, 5] = 1
        assert selector._get_weights(small_mask) < 0

        good_mask = np.zeros((100, 100), dtype=np.uint8)
        good_mask[20:40, 20:40] = 1
        assert selector._get_weights(good_mask) == 0.0

    def test_feature_scaling(self, sample_image):
        selector = MaskSelector()
        selector.fit([sample_image])

        assert hasattr(selector._scaler, "mean_")
        assert hasattr(selector._scaler, "scale_")

        assert selector._knn._fit_X is not None

    def test_mask_selection_logic(self, sample_image):
        selector = MaskSelector(mask_number=1)
        selector.fit([sample_image])

        test_img = deepcopy(sample_image)
        result = selector.transform([test_img])

        selected_mask = result[0].segmented_masks[0]
        assert np.array_equal(selected_mask, sample_image.segmented_masks[1])
