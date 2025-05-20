from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

from src.image_data.image_data import Image


class Splitter:
    def __init__(self, images: list[Image]):
        self.images = images

    def get_split_data(
        self, test_size: float = 0.2, stratify: bool = True, random_state: int = 42
    ) -> tuple[list[Image], list[Image]]:
        """
        Split the images into training and testing sets.
        """

        if len(self.images) == 0:
            return [], []

        if stratify:
            stratify_labels: Any = []
            for img in self.images:
                if img.metadata and "tirads" in img.metadata:
                    stratify_labels.append(img.metadata["tirads"])
                else:
                    stratify = False
                    break

            if stratify:
                stratify_labels = np.array(stratify_labels)
            else:
                stratify_labels = None
        else:
            stratify_labels = None

        indices = np.arange(len(self.images))
        train_indices, test_indices = train_test_split(
            indices, test_size=test_size, random_state=random_state, stratify=stratify_labels
        )

        train_images = [self.images[i] for i in train_indices]
        test_images = [self.images[i] for i in test_indices]

        return train_images, test_images
