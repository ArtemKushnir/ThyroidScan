from copy import deepcopy

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from src.image_data.image_data import Image
from src.image_processing.collecting_features.feature_extractor import FeatureExtractor


class TestFeatureExtractor:
    @pytest.fixture
    def sample_image(self):
        img = Image("test.jpg", (np.random.rand(100, 100, 3) * 255).astype(np.uint8))

        img.cropped_image = (np.random.rand(100, 100) * 255).astype(np.uint8)

        mask1 = np.zeros((100, 100), dtype=np.uint8)
        mask1[20:40, 20:40] = 255
        mask2 = np.zeros((100, 100), dtype=np.uint8)
        mask2[30:70, 30:70] = 255

        img.segmented_masks = [mask1, mask2]
        return img

    @pytest.fixture
    def bad_image(self):
        return Image("bad.jpg", (np.random.rand(100, 100, 3) * 255).astype(np.uint8))

    def test_init(self):
        extractor = FeatureExtractor()
        assert extractor.lbp_radius == 2
        assert extractor.lbp_points == 8
        assert extractor.wavelet_level == 3

        extractor = FeatureExtractor(lbp_radius=3, lbp_points=16, wavelet_level=1)
        assert extractor.lbp_radius == 3
        assert extractor.lbp_points == 16
        assert extractor.wavelet_level == 1

    def test_fit(self, sample_image):
        extractor = FeatureExtractor()
        result = extractor.fit([sample_image])
        assert result is extractor

    def test_transform_validation(self, bad_image):
        extractor = FeatureExtractor()

        with pytest.raises(NotFittedError):
            extractor.transform([bad_image])

        bad_img = deepcopy(bad_image)
        with pytest.raises(NotFittedError):
            extractor.transform([bad_img])

    def test_transform_success(self, sample_image):
        extractor = FeatureExtractor()
        result = extractor.transform([sample_image])

        assert len(result) == 1

        assert result[0].features is not None
        assert len(result[0].features) > 0

        assert any(k.startswith("image_") for k in result[0].features.keys())
        assert any(k.startswith("mask0_") for k in result[0].features.keys())
        assert any(k.startswith("mask1_") for k in result[0].features.keys())

    def test_extract_common_features(self):
        extractor = FeatureExtractor()
        test_region = (np.random.rand(50, 50) * 255).astype(np.uint8)

        features = extractor._extract_common_features(test_region)

        assert "laplacian_mean" in features
        assert "lbp_0" in features
        assert "glcm_contrast" in features
        assert "wavelet_approx_mean" in features

    def test_calc_statistic_features(self):
        extractor = FeatureExtractor()
        test_data = np.random.rand(100)

        features = extractor._calc_statistic_features(test_data)

        assert "mean" in features
        assert "std" in features
        assert "skewness" in features
        assert "kurtosis" in features

        assert "q_10" in features
        assert "q_50" in features
        assert "q_90" in features

        assert "hist_bin_0" in features
        assert f"hist_bin_{extractor.hist_bins - 1}" in features

    def test_extract_mask_features(self, sample_image):
        extractor = FeatureExtractor()
        mask = sample_image.segmented_masks[0]
        image = sample_image.cropped_image

        features = extractor._extract_mask_features(mask, image)

        assert "area" in features
        assert "perimeter" in features
        assert "compactness" in features

        assert "mean" in features
        assert "std" in features

    def test_calc_geometric_features(self):
        extractor = FeatureExtractor()

        empty_mask = np.zeros((100, 100), dtype=np.uint8)
        features = extractor._calc_geometric_features(empty_mask)
        assert features["area"] == 0.0
        assert features["perimeter"] == 0.0

        square_mask = np.zeros((100, 100), dtype=np.uint8)
        square_mask[20:40, 20:40] = 255
        features = extractor._calc_geometric_features(square_mask)
        assert features["area"] > 0
        assert features["perimeter"] > 0
        assert 0 < features["compactness"] <= 1

    def test_prefix_features(self):
        extractor = FeatureExtractor()
        test_features = {"feature1": 1.0, "feature2": 2.0}

        prefixed = extractor._prefix_features(test_features, "test")
        assert "test_feature1" in prefixed
        assert "test_feature2" in prefixed
        assert prefixed["test_feature1"] == 1.0

    def test_feature_completeness(self, sample_image):
        extractor = FeatureExtractor()
        result = extractor.transform([sample_image])[0]
        features = result.features

        assert "image_mean" in features
        assert "image_std" in features
        assert "image_laplacian_mean" in features
        assert "image_glcm_contrast" in features

        assert "mask0_area" in features
        assert "mask0_mean" in features
        assert "mask1_perimeter" in features
        assert "mask1_glcm_energy" in features
