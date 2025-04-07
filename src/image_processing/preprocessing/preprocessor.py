from copy import deepcopy

import cv2
import numpy as np
from sklearn.exceptions import NotFittedError

from src.image_processing.my_image.my_image import Image
from src.image_processing.preprocessing.crop_mask import crop_mask


class Preprocessor:
    """Implementation of image preprocessing."""

    def __init__(self, input_image: Image) -> None:
        if not isinstance(input_image, Image):
            raise TypeError("Image must been object of Image.")

        self.image = deepcopy(input_image)

    def crop_black_frame(self, cropping_radius: int = 10) -> Image:
        """
        Crops out the black area with any white characters around the image
        and divides the image into two if it consists of two parts.

        :param cropping_radius: offset in pixels from the intended border.
        """
        gray_image = cv2.cvtColor(self.image.org_image, cv2.COLOR_BGR2GRAY)
        denoised_image = cv2.medianBlur(gray_image, 3)

        threshold_value, binary_image = cv2.threshold(denoised_image, 5, 255, cv2.THRESH_BINARY)

        contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 10]
        max_contour = max(filtered_contours, key=cv2.contourArea)

        x, y, wight, height = cv2.boundingRect(max_contour)
        crop_points = np.array(
            [y + cropping_radius, y + height - cropping_radius, x + cropping_radius + 5, x + wight - cropping_radius]
        )
        cropped_image = self.image.org_image[crop_points[0] : crop_points[1], crop_points[2] : crop_points[3]]
        self.image.crop_points = crop_points
        self.image.cropped = [cropped_image]

        self._check_for_double()

        return self.image

    def _check_for_double(self) -> None:
        """
        Checks whether the image consists of two parts.
        """
        gray = cv2.cvtColor(self.image.cropped[0], cv2.COLOR_BGR2GRAY)

        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_x = np.absolute(sobel_x)
        sobel_x = np.uint8(255 * sobel_x / np.max(sobel_x))

        column_edges = np.sum(sobel_x, axis=0)
        peak_arg = np.argmax(column_edges)
        peak_value = column_edges[peak_arg]
        median = np.median(column_edges)

        if peak_value < 2 * median:
            return
        left_part = self.image.cropped[0][:, :peak_arg]
        right_part = self.image.cropped[0][:, peak_arg:]
        self.image.cropped_image = [left_part, right_part]

    def write_cropped_to_file(self) -> None:
        """
        Write cropped image into file.
        """
        if self.image.cropped_image is None:
            raise NotFittedError("")
        if len(self.image.cropped_image) == 1:
            cropped_image_path = "cropped_" + self.image.name + ".png"
            cv2.imwrite(cropped_image_path, self.image.cropped[0])
        else:
            cropped_path_1 = "cropped_" + self.image.name + "_1.png"
            cropped_path_2 = "cropped_" + self.image.name + "_2.png"
            cv2.imwrite(cropped_path_1, self.image.cropped[0])
            cv2.imwrite(cropped_path_2, self.image.cropped[1])


if __name__ == "__main__":
    image_path = "../images/images/115_1.png"
    my_image = Image(image_path)

    prepr = Preprocessor(my_image)
    prep_image = prepr.crop_black_frame()
    prepr.write_cropped_to_file()

    image_with_masks = crop_mask(prep_image)
