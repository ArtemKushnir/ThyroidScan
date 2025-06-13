from pathlib import Path
from typing import Optional

import cv2
import pandas as pd
from numpy.typing import NDArray

from src.image_data.image_data import Image


class BUSLoader:
    """
    Loader for Breast Ultrasound (BUS) dataset with images, masks and metadata.

    Expected directory structure:
    - root_dir/
        - images/
        - masks/
        - bus_data.csv

    :param root_dir: Path to the root directory containing the dataset.
    :raises FileNotFoundError: If required directories/files are missing.
    :raises ValueError: If CSV metadata is invalid.
    """

    def __init__(self, root_dir: str) -> None:
        """
        Initialize BUSLoader and validate directory structure.

        :param root_dir: Path to dataset root directory.
        :raises FileNotFoundError: If root directory doesn't exist.
        :raises NotADirectoryError: If root path isn't a directory.
        """
        self.root_path = self._validate_root_dir(root_dir)
        self.image_dir = self._validate_subdir("images")
        self.mask_dir = self._validate_subdir("masks")
        self.csv_path = self._validate_file("bus_data.csv")
        self.metadata = self._load_metadata()

    def load_dataset(self) -> list[Image]:
        """
        Load complete dataset with images, masks and metadata.

        :return: List of Image objects.
        """
        images = []
        for img_path in sorted(self.image_dir.glob("bus_*.png")):
            try:
                img_id = img_path.stem.split("_")[1]
                img_array = self._load_image(img_path)
                if img_array is None:
                    continue

                mask = self._load_mask(img_id)
                if mask is None:
                    continue

                metadata = self.metadata.get("bus_" + img_id)
                if metadata is None:
                    continue

                image = Image(image_name=img_path.name, image=img_array, true_mask=mask, metadata=metadata)
                image.cropped_image = img_array
                images.append(image)

            except Exception as e:
                print(f"Error processing {img_path.name}: {str(e)}.")

        return images

    def _validate_subdir(self, dir_name: str) -> Path:
        """
        Validate subdirectory exists.

        :param dir_name: Subdirectory name (e.g. 'images').
        :return: Validated Path object.
        """
        path = self.root_path / dir_name
        if not path.exists():
            raise FileNotFoundError(f"Required directory not found: {path}")
        if not path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {path}")
        return path

    def _validate_file(self, file_name: str) -> Path:
        """
        Validate file exists.

        :param file_name: File name to validate.
        :return: Validated Path object.
        """
        path = self.root_path / file_name
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not path.is_file():
            raise IsADirectoryError(f"Path is not a file: {path}")
        return path

    def _load_metadata(self) -> dict[str, dict[str, str]]:
        """
        Load and validate metadata from CSV file.

        :return: Dictionary mapping image IDs to metadata.
        :raises ValueError: If required columns are missing.
        """
        df = pd.read_csv(self.csv_path, dtype=str)

        for field in ["Histology", "Pathology", "BIRADS", "Device", "ID"]:
            if field not in df.columns:
                raise ValueError(f"Required column missing in CSV: {field}")

        return {
            row["ID"]: {
                "histology": row["Histology"],
                "pathology": row["Pathology"],
                "target": row["BIRADS"],
                "device": row["Device"],
            }
            for _, row in df.iterrows()
        }

    def _load_mask(self, img_id: str) -> Optional[NDArray]:
        """
        Load corresponding mask for image.

        :param img_id: Image ID (e.g. '0001-l').
        :return: Binary mask as numpy array or None if loading fails.
        """
        mask_path = self.mask_dir / f"mask_{img_id}.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Failed to load mask: {mask_path.name}")
            return None
        return mask

    @staticmethod
    def _load_image(img_path: Path) -> Optional[NDArray]:
        """
        Load single ultrasound image.

        :param img_path: Path to image file.
        :return: Grayscale image as numpy array or None if loading fails.
        """
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Failed to load image: {img_path.name}")
            return None
        return img

    @staticmethod
    def _validate_root_dir(root_dir: str) -> Path:
        """
        Validate root directory exists and is a directory.

        :param root_dir: Path to validate.
        :return: Validated Path object.
        """
        path = Path(root_dir)
        if not path.exists():
            raise FileNotFoundError(f"Root directory doesn't exist: {root_dir}")
        if not path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {root_dir}")
        return path
