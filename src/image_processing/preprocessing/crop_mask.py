from copy import deepcopy
from pathlib import Path

import cv2

from src.image_processing.my_image.my_image import Image


def crop_mask(image_inp: Image) -> Image:
    """
    Crops the mask according to the size of the image and divides the mask into two if the image consists of two.

    :param image_inp: object of Image.
    """
    image = deepcopy(image_inp)
    if not isinstance(image, Image):
        raise TypeError("Input image must have type Image.")

    mask_path = "../images/masks/" + image.name + "_mask.png"
    if not Path(mask_path).exists():
        raise FileNotFoundError(f"File not found. Check mask path.")

    mask = cv2.imread(mask_path)
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    cropped_mask = gray_mask[image.crop_points[0] : image.crop_points[1], image.crop_points[2] : image.crop_points[3]]

    if len(image.cropped) == 1:
        image.true_mask = [cropped_mask]

    else:
        split_column = image.cropped[0].shape[1]
        mask_left = cropped_mask[:, :split_column]
        mask_right = cropped_mask[:, split_column:]
        image.true_mask = [mask_left, mask_right]

    return image
