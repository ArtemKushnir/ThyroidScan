from typing import Any, Callable, Optional

import torch
from torch.utils.data import Dataset

from src.image_data.image_data import Image


class ImageDataset(Dataset):
    def __init__(self, images: list[Image], transform: Optional[Callable] = None) -> None:
        self.images = images
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        img = self.images[idx]

        image_tensor = torch.from_numpy(img.org_image).float()
        if self.transform is not None:
            image_tensor = self.transform(image_tensor)
        if img.features is None:
            raise ValueError("Missing features")
        features = torch.tensor(list(img.features.values())).float()

        if img.metadata is None:
            raise ValueError("Missing metadata")
        label = torch.tensor(img.metadata["tirads"])

        return {"pixels": image_tensor, "features": features, "label": label}
