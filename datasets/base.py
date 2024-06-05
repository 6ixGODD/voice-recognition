from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from utils.images import padding_resize

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


class BaseImageDataset:
    def __init__(self):
        self.images: List[np.ndarray] = []
        self.labels: List[int] = []
        self.categories: Dict[int, str] = {}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.images[idx], self.labels[idx], self.categories[self.labels[idx]]
        elif isinstance(idx, slice):
            return [
                (self.images[i], self.labels[i], self.categories[self.labels[i]]) for i in range(*idx.indices(len(self)))
            ]
        elif isinstance(idx, list) or isinstance(idx, np.ndarray):
            return [
                (self.images[i], self.labels[i], self.categories[self.labels[i]]) for i in idx
            ]
        else:
            raise ValueError(f"Invalid index type {type(idx)}")

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __str__(self):
        return (
            f'{self.__class__.__name__}('
            f'size={len(self)}, '
            f'num_classes={len(self.categories)}, '
            f'categories={self.categories})'
        )

    def __repr__(self):
        return self.__str__()

    def shuffle(self):
        indices = np.arange(len(self))
        np.random.shuffle(indices)
        self.images = [self.images[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]

    def resize(self, size: Tuple[int, int], padding: bool = False, **kwargs):
        for i, image in enumerate(self.images):
            if padding:
                self.images[i], _ = padding_resize(image, size=size, **kwargs)
            else:
                self.images[i] = cv2.resize(image, size)

    def append(self, image: np.ndarray, label: int, category: str = '', **kwargs):
        if category != '' and label not in self.categories:
            if category in self.categories.values():
                raise ValueError(f"Category {category} not match with label {label}")
            self.categories[label] = category
        self.images.append(image)
        self.labels.append(label)

    def merge(self, dataset: 'BaseImageDataset'):
        if self.categories != dataset.categories:
            raise ValueError(f"Categories mismatch: {self.categories} != {dataset.categories}")
        self.images += dataset.images
        self.labels += dataset.labels

    def save_images(self, output_dir: str, fmt: str = 'jpg'):
        output_dir = Path(output_dir)
        for i, (image, label, category, *_) in enumerate(self):
            if not Path(output_dir / category).exists():
                Path(output_dir / category).mkdir(parents=True, exist_ok=True)
            print(f"-- Saving image {i + 1}/{len(self)}")
            cv2.imwrite(
                str(output_dir / category / f"{label}-{i}.{fmt}"),
                image
            )

    def load_images(self, root: str, categories: list, limit: int = 0):
        print("== Loading Image")
        num_classes = len(categories)
        for i, c in enumerate(categories):
            path = Path(root) / c
            for j, f in enumerate(path.glob('*')):
                if f.suffix in ['.jpg', '.png', '.jpeg']:
                    print(f"-- Processing Image {f} with label {i} in {c}")
                    self.append(
                        image=cv2.imread(str(f)),
                        label=i,
                        category=c
                    )
                    if limit != 0 and j >= limit / num_classes:
                        break
                else:
                    print(f"-- Skipping File {f} with invalid extension")


if __name__ == '__main__':
    dataset_test = BaseImageDataset()
    dataset_test.load_images(root='../data/A', categories=['busy', 'free'], limit=10)
    print(dataset_test)
    dataset_test.save_images(output_dir='output')
