import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from src.data_loaders.ddti_loader import DDTILoader


class TestDDTILoader:
    @pytest.fixture
    def setup_dirs(self):
        tmp_dir = tempfile.mkdtemp()
        xml_dir = Path(tmp_dir) / "xml"
        images_dir = Path(tmp_dir) / "images"
        xml_dir.mkdir()
        images_dir.mkdir()

        test_xml = """
        <root>
            <number>123</number>
            <age>45</age>
            <sex>F</sex>
            <tirads>4</tirads>
            <mark>
                <image>1</image>
                <svg>[{"points": [{"x": 10, "y": 10}, {"x": 50, "y": 10}, {"x": 50, "y": 50}, {"x": 10, "y": 50}]}]</svg>
            </mark>
        </root>
        """

        (xml_dir / "test.xml").write_text(test_xml)

        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(images_dir / "123_1.jpg"), test_img)

        yield xml_dir, images_dir

        shutil.rmtree(tmp_dir)

    def test_init_valid_dirs(self, setup_dirs):
        xml_dir, images_dir = setup_dirs
        loader = DDTILoader(str(xml_dir), str(images_dir))
        assert loader.xml_dir == xml_dir
        assert loader.images_dir == images_dir

    def test_init_invalid_dirs(self):
        with pytest.raises(FileNotFoundError):
            DDTILoader("invalid_xml", "invalid_images")

    def test_create_images_with_masks(self, setup_dirs):
        xml_dir, images_dir = setup_dirs
        loader = DDTILoader(str(xml_dir), str(images_dir))

        images = loader.load_dataset()
        assert len(images) == 1
        image = images[0]

        assert image.name == "123_1.jpg"
        assert isinstance(image.org_image, np.ndarray)
        assert isinstance(image.true_mask, np.ndarray)
        assert image.true_mask.sum() > 0

        assert image.metadata["age"] == "45"
        assert image.metadata["sex"] == "F"
        assert image.metadata["target"] == "4"

    def test_skip_no_tirads(self, setup_dirs):
        xml_dir, images_dir = setup_dirs
        xml_path = xml_dir / "test.xml"
        xml_content = xml_path.read_text().replace("<tirads>4</tirads>", "")
        xml_path.write_text(xml_content)

        loader = DDTILoader(str(xml_dir), str(images_dir), only_with_tirads=True)
        images = loader.load_dataset()
        assert len(images) == 0

        loader = DDTILoader(str(xml_dir), str(images_dir), only_with_tirads=False)
        images = loader.load_dataset()
        assert len(images) == 1
        assert images[0].metadata["target"] == "0"

    def test_invalid_svg(self, setup_dirs):
        xml_dir, images_dir = setup_dirs
        xml_path = xml_dir / "test.xml"
        xml_content = xml_path.read_text().replace("points", "invalid_points")
        xml_path.write_text(xml_content)

        loader = DDTILoader(str(xml_dir), str(images_dir))
        with pytest.raises(KeyError):
            loader.load_dataset()

    def test_missing_image_file(self, setup_dirs):
        xml_dir, images_dir = setup_dirs
        (images_dir / "123_1.jpg").unlink()

        loader = DDTILoader(str(xml_dir), str(images_dir))
        with patch("builtins.print") as mock_print:
            images = loader.load_dataset()
            assert len(images) == 0
            mock_print.assert_called_with("There is no 123_1.jpg in directory with images.")

    def test_empty_xml(self, setup_dirs):
        xml_dir, images_dir = setup_dirs
        (xml_dir / "empty.xml").write_text("<root></root>")

        loader = DDTILoader(str(xml_dir), str(images_dir))
        with pytest.raises(ValueError, match="xml must contains <number>."):
            loader._create_images_by_xml(xml_dir / "empty.xml")

    @patch("cv2.imread")
    @patch("cv2.fillPoly")
    def test_image_creation_error(self, mock_fillpoly, mock_imread, setup_dirs):
        xml_dir, images_dir = setup_dirs
        mock_imread.return_value = None

        loader = DDTILoader(str(xml_dir), str(images_dir))
        with patch("builtins.print") as mock_print:
            images = loader.load_dataset()
            assert len(images) == 0
            mock_print.assert_called_with("Failed to read image: 123_1.jpg")

    def test_check_label(self):
        with pytest.raises(TypeError, match="only_with_tirads must be true or false."):
            DDTILoader._check_label("not_a_bool")

        assert DDTILoader._check_label(True) is True
        assert DDTILoader._check_label(False) is False
