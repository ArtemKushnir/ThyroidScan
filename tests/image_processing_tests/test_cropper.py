import os

import numpy as np
import pytest

from src.image_data.image_data import Image
from src.image_processing.preprocessing.cropper import Cropper


class TestCropper:
    @pytest.fixture
    def sample_image(self):
        rgb_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        binary_mask = np.random.choice([0, 255], (100, 100)).astype(np.uint8)
        return Image("test_image.jpg", rgb_image, binary_mask)

    @pytest.fixture
    def double_image(self):
        image = np.zeros((100, 200), dtype=np.uint8)
        image[:, :90] = 200
        image[:, 110:] = 200

        for col in range(90, 110):
            image[:, col] = int(200 * (1 - (col - 90) / 20))

        mask = np.zeros((100, 200), dtype=np.uint8)
        mask[:, :100] = 255

        return Image("reliable_double.jpg", np.dstack([image] * 3), mask)

    def test_init_validation(self):
        with pytest.raises(TypeError):
            Cropper(crop_radius="10")
        with pytest.raises(ValueError):
            Cropper(crop_radius=-5)
        with pytest.raises(ValueError):
            Cropper(crop_radius=60)

        cropper = Cropper(crop_radius=10)
        assert cropper.crop_radius == 10

    def test_fit(self, sample_image):
        cropper = Cropper()
        assert cropper.fit([sample_image]) is cropper
        assert cropper.fit([sample_image], target=None) is cropper

    def test_transform_input_validation(self, sample_image):
        cropper = Cropper()
        with pytest.raises(TypeError):
            cropper.transform(sample_image)

    def test_single_image_cropping(self, sample_image):
        cropper = Cropper(crop_radius=5)
        result = cropper.transform([sample_image])

        assert len(result) == 1
        cropped_image = result[0]
        assert cropped_image.cropped_image is not None
        assert cropped_image.crop_points is not None
        assert cropped_image.true_mask is not None
        assert cropped_image.cropped_image.shape == (
            sample_image.org_image.shape[0] - 2 * 5,
            sample_image.org_image.shape[1] - 2 * 5 - 5,
        )

    def test_double_image_splitting(self, double_image):
        cropper = Cropper(crop_radius=5)
        result = cropper.transform([double_image])

        assert len(result) == 2
        left, right = result

        assert left.cropped_image is not None
        assert right.cropped_image is not None
        assert left.cropped_image.shape[1] + right.cropped_image.shape[1] <= double_image.org_image.shape[1]

        assert left.true_mask is not None
        assert right.true_mask is not None
        assert left.true_mask.shape == left.cropped_image.shape
        assert right.true_mask.shape == right.cropped_image.shape

    def test_crop_mask_validation(self, sample_image):
        cropper = Cropper()

        with pytest.raises(ValueError):
            cropper._crop_mask(sample_image)

    def test_write_cropped_to_file(self, sample_image, tmp_path):
        cropper = Cropper()
        cropped_image = cropper.transform([sample_image])[0]

        os.chdir(tmp_path)

        Cropper.write_cropped_to_file(cropped_image)
        assert os.path.exists(f"cropped_{cropped_image.name}")

    def test_check_for_double(self):
        test_image = np.zeros((100, 200), dtype=np.uint8)
        test_image[:, :90] = 255
        test_image[:, 110:] = 255

        cropper = Cropper()
        result = cropper._check_for_double(test_image)
        assert result is not None
        left, right = result
        assert left.shape[1] + right.shape[1] <= test_image.shape[1]

        single_image = np.zeros((100, 100), dtype=np.uint8)
        single_image[20:80, 20:80] = 255
        assert cropper._check_for_double(single_image) is None

    def test_image_validation(self):
        cropper = Cropper()
        with pytest.raises(TypeError):
            cropper._validate_image("not_an_image")
        with pytest.raises(TypeError):
            cropper._validate_image(None)
