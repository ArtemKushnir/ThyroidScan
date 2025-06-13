import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from src.data_loaders.bus_bra_loader import BUSLoader
from src.image_data.image_data import Image


class TestBUSLoader:
    @pytest.fixture
    def setup_test_dir(self):
        tmp_dir = Path(tempfile.mkdtemp())

        (tmp_dir / "images").mkdir()
        (tmp_dir / "masks").mkdir()

        for i in range(1, 4):
            img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
            cv2.imwrite(str(tmp_dir / "images" / f"bus_{i:04d}.png"), img)

            mask = np.zeros((100, 100), dtype=np.uint8)
            mask[20:80, 20:80] = 255
            cv2.imwrite(str(tmp_dir / "masks" / f"mask_{i:04d}.png"), mask)

        csv_data = """ID,Histology,Pathology,BIRADS,Device
bus_0001,Benign,Fibroadenoma,2,GE
bus_0002,Malignant,IDC,5,Siemens
bus_0003,Benign,Cyst,3,Philips"""

        (tmp_dir / "bus_data.csv").write_text(csv_data)

        yield tmp_dir
        shutil.rmtree(tmp_dir)

    def test_init_valid_dir(self, setup_test_dir):
        loader = BUSLoader(str(setup_test_dir))
        assert loader.root_path == setup_test_dir
        assert loader.image_dir == setup_test_dir / "images"
        assert loader.mask_dir == setup_test_dir / "masks"
        assert loader.csv_path == setup_test_dir / "bus_data.csv"
        assert len(loader.metadata) == 3

    def test_init_invalid_dir(self):
        with pytest.raises(FileNotFoundError):
            BUSLoader("/invalid/path")

    def test_load_dataset(self, setup_test_dir):
        loader = BUSLoader(str(setup_test_dir))
        images = loader.load_dataset()

        assert len(images) == 3
        for img in images:
            assert isinstance(img, Image)
            assert img.org_image.shape == (100, 100)
            assert img.true_mask.sum() > 0
            assert img.metadata is not None

    def test_missing_image(self, setup_test_dir):
        (setup_test_dir / "images" / "bus_0001.png").unlink()

        loader = BUSLoader(str(setup_test_dir))
        images = loader.load_dataset()
        assert len(images) == 2

    def test_missing_mask(self, setup_test_dir):
        (setup_test_dir / "masks" / "mask_0002.png").unlink()

        loader = BUSLoader(str(setup_test_dir))
        with patch("builtins.print") as mock_print:
            images = loader.load_dataset()
            assert len(images) == 2
            mock_print.assert_called_with("Failed to load mask: mask_0002.png")

    def test_invalid_csv(self, setup_test_dir):
        (setup_test_dir / "bus_data.csv").write_text("invalid,data")

        with pytest.raises(ValueError):
            BUSLoader(str(setup_test_dir))

    def test_missing_csv_column(self, setup_test_dir):
        csv_data = """ID,Histology,Pathology,Device
bus_0001,Benign,Fibroadenoma,GE"""
        (setup_test_dir / "bus_data.csv").write_text(csv_data)

        with pytest.raises(ValueError, match="Required column missing in CSV: BIRADS"):
            BUSLoader(str(setup_test_dir))

    def test_load_image(self, setup_test_dir):
        loader = BUSLoader(str(setup_test_dir))
        img_path = setup_test_dir / "images" / "bus_0001.png"
        img = loader._load_image(img_path)

        assert isinstance(img, np.ndarray)
        assert img.dtype == np.uint8
        assert img.shape == (100, 100)

    def test_load_mask(self, setup_test_dir):
        loader = BUSLoader(str(setup_test_dir))
        mask = loader._load_mask("0001")

        assert isinstance(mask, np.ndarray)
        assert mask.dtype == np.uint8
        assert mask.shape == (100, 100)
        assert np.all(np.isin(mask, [0, 255]))

    def test_metadata_loading(self, setup_test_dir):
        loader = BUSLoader(str(setup_test_dir))
        metadata = loader.metadata

        assert len(metadata) == 3
        assert metadata["bus_0001"]["histology"] == "Benign"
        assert metadata["bus_0002"]["pathology"] == "IDC"
        assert metadata["bus_0003"]["target"] == "3"
