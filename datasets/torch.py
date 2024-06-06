from typing import Callable, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

from datasets.base import BaseImageClassificationDataset


class TorchImageClassificationDataset(BaseImageClassificationDataset, Dataset[Tuple[torch.Tensor, int]]):
    def __init__(self, transform: Optional[Callable] = None):
        super(TorchImageClassificationDataset, self).__init__()
        self.transform = transform
        if transform is None:
            self.transform = lambda x: x

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        image, label, _ = super(TorchImageClassificationDataset, self).__getitem__(idx)
        image = self.transform(Image.fromarray(image))
        return image, label

    def __len__(self) -> int:
        return len(self.images)
