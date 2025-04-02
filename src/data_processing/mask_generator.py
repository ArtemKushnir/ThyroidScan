import json
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np


class XMLMaskGenerator:
    """
    Processing xml files and creating masks from them.
    """

    def __init__(self, xml_dir: str, images_dir: str, output_mask_dir: str = "masks") -> None:
        """
        :param xml_dir: directory with xml files.
        :param images_dir: directory with images.
        :param output_mask_dir: directory where masks will be loaded.
        """
        self._check_directory(xml_dir, images_dir)

        self.xml_dir = Path(xml_dir)
        self.images_dir = Path(images_dir)
        self.output_mask_dir = Path(output_mask_dir)

        self.output_mask_dir.mkdir(exist_ok=True)

    def create_directory_with_masks(self) -> None:
        """
        Creates a directory with masks using xml files and source images.
        """
        for xml_file in self.xml_dir.glob("*.xml"):
            self._create_masks_by_xml(xml_file)

    def _create_masks_by_xml(self, xml_file: Path) -> None:
        """
        Creates masks for images, information about which is contained in a single xml file.
        :param xml_file: xml file.
        """
        tree = ET.parse(xml_file)
        root = tree.getroot()

        num = root.find("number")
        if num is None or num.text is None:
            raise ValueError("xml must contains <number>.")
        case_number = num.text.strip()

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

            self._create_mask(image_path, svg_data, image_name)

    def _create_mask(self, image_path: Path, svg_data: str, image_name: str) -> None:
        """
        Creates mask using points from <svg> in xml.
        :param image_path: image file path.
        :param svg_data: svg data, that contains in xml.
        :param image_name:name of image file.
        """
        try:
            image = cv2.imread(str(image_path))
            height, wight = image.shape[:2]
            shapes = json.loads(svg_data)
            mask = np.zeros((height, wight), dtype=np.uint8)

            for shape in shapes:
                points = [(point["x"], point["y"]) for point in shape["points"]]
                polygon = np.array(points, dtype=np.int32)
                cv2.fillPoly(mask, [polygon], color=(255, 255, 255))

            mask_name = f"{image_name.split('.')[0]}_mask.png"
            mask_path = self.output_mask_dir / mask_name
            cv2.imwrite(str(mask_path), mask)

        except json.JSONDecodeError:
            print(f" Error with parsing svg in {image_name}")
        except cv2.error as ex:
            print(f"Error OpenCv with {image_name}: {ex}")

    @staticmethod
    def _check_directory(xml_dir: str, images_dir: str) -> None:
        """
         Verifies the correctness of the directory path.
        :param xml_dir: directory with xml files.
        :param images_dir: directory with images.
        """
        if not Path(xml_dir).exists():
            raise FileNotFoundError(f"Directory not found. Check xml directory path.")
        if not Path(images_dir).exists():
            raise FileNotFoundError(f"Directory not found. Check images directory path.")


if __name__ == "__main__":
    xml_dir = "xml_files"
    images_dir = "images"
    output_mask_dir = "masks"

    mask_gen = XMLMaskGenerator(xml_dir, images_dir, output_mask_dir)
    mask_gen.create_directory_with_masks()
