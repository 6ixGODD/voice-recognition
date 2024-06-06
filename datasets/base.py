from pathlib import Path
from typing import Dict, Generator, Iterable, List, Tuple, Union, Sized

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

from utils.images import padding_resize

plt.rcParams['font.family'] = 'Times New Roman'
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png']


class BaseImageClassificationDataset(Iterable, Sized):
    def __init__(self, **kwargs):
        self.images: List[np.ndarray] = []
        self.labels: List[int] = []
        self.categories: Dict[int, str] = {}

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx) -> Union[Tuple[np.ndarray, int, str], List[Tuple[np.ndarray, int, str]]]:
        if isinstance(idx, int):
            return self.images[idx], self.labels[idx], self.categories[self.labels[idx]]
        elif isinstance(idx, slice):
            return [
                (self.images[i], self.labels[i], self.categories[self.labels[i]]) for i in
                range(*idx.indices(len(self)))
            ]
        elif isinstance(idx, list) or isinstance(idx, np.ndarray):
            return [
                (self.images[i], self.labels[i], self.categories[self.labels[i]]) for i in idx
            ]
        else:
            raise ValueError(f"Invalid index type {type(idx)}")

    def __iter__(self) -> Generator[Tuple[np.ndarray, int, str], None, None]:
        for i in range(len(self)):
            yield self[i]

    def __add__(self, other: 'BaseImageClassificationDataset') -> 'BaseImageClassificationDataset':
        if self.categories != other.categories:
            raise ValueError(f"Categories mismatch: {self.categories} != {other.categories}")
        new_dataset = BaseImageClassificationDataset()
        new_dataset.images = self.images + other.images
        new_dataset.labels = self.labels + other.labels
        new_dataset.categories = self.categories
        return new_dataset

    def __str__(self) -> str:
        return (
            f'{self.__class__.__name__}('
            f'size={len(self)}, '
            f'num_classes={len(self.categories)}, '
            f'categories={self.categories})'
        )

    def __repr__(self) -> str:
        return self.__str__()

    def overview(self):
        figure = plt.figure(figsize=(10, 6.5), dpi=300)
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

        ax0 = figure.add_subplot(gs[0])
        labels, counts = np.unique(self.labels, return_counts=True)
        categories = [self.categories[lb] for lb in labels]
        ax0.bar(categories, counts)
        ax0.set_xticks(categories)
        plt.xticks(rotation=45, fontstyle='italic', fontsize=10)
        ax0.set_xlabel('Class')
        ax0.set_ylabel('Count')
        ax0.spines['right'].set_visible(False)
        ax0.spines['top'].set_visible(False)
        ax0.set_title('Image Distribution')

        ax1 = figure.add_subplot(gs[1])
        inner_gs = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs[1], wspace=0.1, hspace=0.1)
        for i in range(9):
            ax = figure.add_subplot(inner_gs[i])
            img_index = np.random.randint(len(self.images))
            ax.imshow(self.images[img_index])
            ax.set_title(f'Class: {self.categories[self.labels[img_index]]}', fontsize=10)
            ax.axis('off')
        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        ax1.axis('off')

        plt.tight_layout()
        plt.show()

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

    def save_images(self, output_dir: str, fmt: str = 'jpg'):
        if f'.{fmt}' not in IMAGE_EXTENSIONS:
            raise ValueError(f"Invalid image format {fmt}")
        output_dir = Path(output_dir)
        for i, (image, label, category, *_) in enumerate(self):
            if not Path(output_dir / category).exists():
                Path(output_dir / category).mkdir(parents=True, exist_ok=True)
            print(f"-- Saving image {i + 1}/{len(self)}")
            cv2.imwrite(
                str(output_dir / category / f"{label}-{i}.{fmt}"),
                image
            )

    def load_images(self, root: str, limit: int = 0):
        categories = [f.name for f in Path(root).iterdir() if f.is_dir()]
        num_classes = len(categories)
        for i, c in enumerate(categories):
            path = Path(root) / c
            for j, f in enumerate(path.glob('*')):
                if f.suffix in IMAGE_EXTENSIONS:
                    self.append(
                        image=cv2.imread(str(f)),
                        label=i,
                        category=c
                    )
                    if limit != 0 and j >= limit / num_classes:
                        break


if __name__ == '__main__':
    dataset_test = BaseImageClassificationDataset()
    dataset_test.load_images(root='../data/SpectrogramImages')
    print(dataset_test)
    dataset_test.overview()
