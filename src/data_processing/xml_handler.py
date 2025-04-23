import copy
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.image_data.image_data import Image


class XMLHandler:
    """
    Handles XML files containing image annotations and metadata, and generates corresponding mask images.

    This class parses XML files to extract information such as annotation polygons (in SVG format),
    patient metadata (e.g., age, sex, TIRADS score), and links it with source images to create
    representations of images with masks.
    """

    def __init__(self, xml_dir: str, images_dir: str) -> None:
        """
        Initializes the handler with directories containing XML files and source images.

        :param xml_dir: Path to the directory containing XML annotation files.
        :param images_dir: Path to the directory containing source image files.
        """
        self._check_directory(xml_dir, images_dir)

        self.xml_dir = Path(xml_dir)
        self.images_dir = Path(images_dir)

    def create_images_with_masks(self) -> list:
        """
        Parses all XML files in the directory and creates a list of image objects with associated masks and metadata.

        Skips any XML entries that do not include a TIRADS score.

        :return: List of Image objects with masks and metadata.
        """
        images_list = []
        for xml_file in self.xml_dir.glob("*.xml"):
            images_list.extend(self._create_images_by_xml(xml_file))
        return images_list

    def _create_images_by_xml(self, xml_file: Path) -> list:
        """
        Parses a single XML file and extracts all annotated images from it.

        Skips the file if the TIRADS score is missing.

        :param xml_file: Path to the XML file.
        :return: List of Image objects extracted from this XML file.
        """
        tree = ET.parse(xml_file)
        root = tree.getroot()

        num = root.find("number")
        if num is None or num.text is None:
            raise ValueError("xml must contains <number>.")
        case_number = num.text.strip()

        metadata = {
            "age": root.findtext("age") or None,
            "sex": root.findtext("sex") or None,
            "composition": root.findtext("composition") or None,
            "echogenicity": root.findtext("echogenicity") or None,
            "margins": root.findtext("margins") or None,
            "calcifications": root.findtext("calcifications") or None,
            "tirads": root.findtext("tirads") or None,
            "reportbacaf": root.findtext("reportbacaf") or None,
            "reporteco": root.findtext("reporteco") or None,
        }

        if metadata["tirads"] is None:
            print(f"Skipping {xml_file.name}: no TIRADS.")
            return []

        images = []

        for mark in root.findall("mark"):
            image_mark = mark.find("image")
            if image_mark is None or image_mark.text is None:
                continue
            image_number = image_mark.text.strip()

            svg = mark.find("svg")
            if svg is None or svg.text is None:
                continue
            svg_data = svg.text

            image_name = f"{case_number}_{image_number}.jpg"
            image_path = self.images_dir / image_name

            if not image_path.exists():
                print(f"There is no {image_name} in directory with images.")
                continue

            image = self._create_image(image_path, svg_data, image_name, metadata)
            if image is not None:
                images.append(image)

        return images

    def _create_image(self, image_path: Path, svg_data: str, image_name: str, metadata: dict) -> Optional[Image]:
        """
        Creates an Image object with a binary mask generated from the SVG polygon annotation.

        :param image_path: Path to the image file.
        :param svg_data: JSON-encoded SVG data containing the annotation points.
        :param image_name: Name of the image file.
        :param metadata: Dictionary of metadata extracted from the XML file.
        :return: An Image object with the original image, its mask, and metadata.
        """
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Failed to read image: {image_name}")
                return None
            height, wight = image.shape[:2]
            shapes = json.loads(svg_data)
            mask = np.zeros((height, wight), dtype=np.uint8)

            for shape in shapes:
                points = [(point["x"], point["y"]) for point in shape["points"]]
                polygon = np.array(points, dtype=np.int32)
                cv2.fillPoly(mask, [polygon], color=(255, 255, 255))

            return Image(image_name, image, mask, copy.deepcopy(metadata))

        except json.JSONDecodeError:
            print(f" Error with parsing svg in {image_name}")
            return None
        except cv2.error as ex:
            print(f"Error OpenCv with {image_name}: {ex}")
            return None

    @staticmethod
    def _check_directory(xml_dir: str, images_dir: str) -> None:
        """
        Verifies the existence of the XML and image directories.

        :param xml_dir: Path to the XML directory.
        :param images_dir: Path to the image directory.
        :raises FileNotFoundError: If either directory does not exist.
        """
        if not Path(xml_dir).exists():
            raise FileNotFoundError(f"Directory not found. Check xml directory path.")
        if not Path(images_dir).exists():
            raise FileNotFoundError(f"Directory not found. Check images directory path.")
