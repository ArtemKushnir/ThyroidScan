import numpy as np
import pytest

from src.image_data.image_data import Image


class TestImage:
    @pytest.fixture
    def sample_image_data(self):
        rgb_image = np.random.rand(100, 100, 3) * 255
        binary_mask = np.random.choice([0, 255], (100, 100))
        grayscale_crop = np.random.rand(100, 100) * 255
        metadata = {"patient_id": "123", "scan_type": "ultrasound"}
        return rgb_image, binary_mask, grayscale_crop, metadata

    def test_initialization(self, sample_image_data):
        rgb_image, binary_mask, _, metadata = sample_image_data
        img_obj = Image("test_image", rgb_image, binary_mask, metadata)

        assert img_obj.name == "test_image"
        assert np.array_equal(img_obj.org_image, rgb_image)
        assert np.array_equal(img_obj.true_mask, binary_mask)
        assert img_obj.metadata == metadata
        assert img_obj.cropped_image is None
        assert img_obj.crop_points is None
        assert img_obj.segmented_masks is None
        assert img_obj.features is None

    def test_invalid_image_name(self):
        with pytest.raises(TypeError):
            Image(None, np.random.rand(10, 10, 3))
        with pytest.raises(TypeError):
            Image(123, np.random.rand(10, 10, 3))

    def test_invalid_image(self):
        with pytest.raises(TypeError):
            Image("test", None)
        with pytest.raises(TypeError):
            Image("test", "not_an_image")

    def test_cropped_image_property(self, sample_image_data):
        rgb_image, _, grayscale_crop, _ = sample_image_data
        img_obj = Image("test", rgb_image)

        img_obj.cropped_image = grayscale_crop
        assert np.array_equal(img_obj.cropped_image, grayscale_crop)

        with pytest.raises(TypeError):
            img_obj.cropped_image = "not_an_array"

    def test_true_mask_property(self, sample_image_data):
        rgb_image, binary_mask, _, _ = sample_image_data
        img_obj = Image("test", rgb_image)

        valid_mask = np.array([[0, 255], [255, 0]])
        img_obj.true_mask = valid_mask
        assert np.array_equal(img_obj.true_mask, valid_mask)

        with pytest.raises(TypeError):
            img_obj.true_mask = "not_an_array"

    def test_segmented_masks_property(self, sample_image_data):
        rgb_image, _, _, _ = sample_image_data
        img_obj = Image("test", rgb_image)

        valid_masks = [np.array([[0, 255], [255, 0]]), np.array([[255, 0], [0, 255]])]
        img_obj.segmented_masks = valid_masks
        assert all(np.array_equal(a, b) for a, b in zip(img_obj.segmented_masks, valid_masks))

        with pytest.raises(TypeError):
            img_obj.segmented_masks = "not_a_list"

    def test_features_property(self, sample_image_data):
        rgb_image, _, _, _ = sample_image_data
        img_obj = Image("test", rgb_image)

        features = {"area": 100, "perimeter": 50}
        img_obj.features = features
        assert img_obj.features == features

        with pytest.raises(TypeError):
            img_obj.features = None
        with pytest.raises(TypeError):
            img_obj.features = "not_a_dict"

    def test_repr(self, sample_image_data):
        rgb_image, binary_mask, _, metadata = sample_image_data
        img_obj = Image("test_image", rgb_image, binary_mask, metadata)

        repr_str = repr(img_obj)
        assert "test_image" in repr_str
        assert str(rgb_image.shape) in repr_str
        assert "patient_id" in repr_str or "metadata_keys" in repr_str
