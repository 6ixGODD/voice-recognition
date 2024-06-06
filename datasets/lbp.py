import warnings
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
from matplotlib import gridspec

plt.rcParams['font.family'] = 'Times New Roman'

import cv2
import numpy as np

from datasets.base import BaseImageClassifierDataset

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


class LocalBinaryPatternsImageClassifierDataset(BaseImageClassifierDataset):
    def __init__(self):
        super(LocalBinaryPatternsImageClassifierDataset, self).__init__()
        self.with_channel_flatten = False
        self.lbp_images: List[np.ndarray] = []
        self.lbp_vectors: List[np.ndarray] = []

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return (
                self.images[idx],
                self.labels[idx],
                self.categories[self.labels[idx]],
                self.lbp_images[idx],
                self.lbp_vectors[idx]
            )
        elif isinstance(idx, slice):
            return [
                (
                    self.images[i],
                    self.labels[i],
                    self.categories[self.labels[i]],
                    self.lbp_images[i],
                    self.lbp_vectors[i]
                ) for i in range(*idx.indices(len(self)))
            ]
        elif isinstance(idx, list) or isinstance(idx, np.ndarray):
            return [
                (
                    self.images[i],
                    self.labels[i],
                    self.categories[self.labels[i]],
                    self.lbp_images[i],
                    self.lbp_vectors[i]
                ) for i in idx
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
            f'categories={self.categories}'
            f'with_channel_flatten={self.with_channel_flatten})'
        )

    def __repr__(self):
        return self.__str__()

    def from_base_dataset(self, dataset: BaseImageClassifierDataset):
        self.images = dataset.images
        self.labels = dataset.labels
        self.categories = dataset.categories
        for image in self.images:
            lbp_image = self.__faster_calculate_lbp(image)
            lbp_vector = self.__calculate_lbp_vector(lbp_image)
            self.lbp_images.append(lbp_image)
            self.lbp_vectors.append(lbp_vector)

    def shuffle(self):
        indices = np.arange(len(self))
        np.random.shuffle(indices)
        self.images = [self.images[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]
        self.lbp_images = [self.lbp_images[i] for i in indices]
        self.lbp_vectors = [self.lbp_vectors[i] for i in indices]

    def append(self, image: np.ndarray, label: int, category: str = '', **kwargs):
        super(LocalBinaryPatternsImageClassifierDataset, self).append(image, label, category, **kwargs)
        self.lbp_images.append(kwargs['lbp_image'])
        self.lbp_vectors.append(kwargs['lbp_vector'])

    def merge(self, dataset: 'LocalBinaryPatternsImageClassifierDataset'):
        if not self.with_channel_flatten and dataset.with_channel_flatten:
            raise ValueError("Cannot merge dataset with channel flatten and dataset without channel flatten.")
        super(LocalBinaryPatternsImageClassifierDataset, self).merge(dataset)
        self.lbp_images += dataset.lbp_images
        self.lbp_vectors += dataset.lbp_vectors

    def save_lbp_images(self, output_dir: str, fmt: str = 'jpg'):
        output_dir = Path(output_dir)
        for i, (image, label, category, lbp_image, lbp_vector) in enumerate(self):
            if not Path(output_dir / category).exists():
                Path(output_dir / category).mkdir(parents=True, exist_ok=True)
            print(f"-- Saving LBP image {i + 1}/{len(self)}")
            cv2.imwrite(
                str(output_dir / category / f"{label}-{i}.lbp.{fmt}"),
                lbp_image
            )

    def load_images(self, root: str, limit: int = 0, channel_flatten: bool = False):
        print("== Loading Image")
        self.with_channel_flatten = channel_flatten
        categories = [d.name for d in Path(root).iterdir() if d.is_dir()]
        num_classes = len(categories)
        for i, c in enumerate(categories):
            path = Path(root) / c
            for j, f in enumerate(path.glob('*')):
                if f.suffix in ['.jpg', '.png', '.jpeg']:
                    # print(f"-- Processing Image {f} with label {i} in {c}")
                    img = cv2.imread(str(f))
                    if channel_flatten:
                        # Flatten the channel by stacking the RGB channel horizontally into an image
                        # with 3 times the width of the original image
                        image = cv2.imread(str(f))
                        R, G, B = cv2.split(image)
                        gray_img = np.hstack((R, G, B))
                    else:
                        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    lbp_img = self.__faster_calculate_lbp(gray_img)
                    v = self.__calculate_lbp_vector(lbp_img)
                    self.append(image=img, label=i, category=c, lbp_image=lbp_img, lbp_vector=v)
                    if limit != 0 and j >= limit / num_classes:
                        break
                else:
                    print(f"-- Skipping {f} due to unsupported format")

    def overview(self):
        figure = plt.figure(figsize=(10, 12), dpi=300)
        gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, .8])

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

        ax2 = figure.add_subplot(gs[2])
        ax3 = figure.add_subplot(gs[3])
        # Display 9 random LBP images and their histogram
        inner_gs_ax2 = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs[2], wspace=0.1, hspace=0.1)
        inner_gs_ax3 = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs[3], wspace=0.1, hspace=0.1)
        for i in range(9):
            ax_ax2 = figure.add_subplot(inner_gs_ax2[i])
            ax_ax3 = figure.add_subplot(inner_gs_ax3[i])
            img_index = np.random.randint(len(self.lbp_images))

            ax_ax2.imshow(self.lbp_images[img_index], cmap='gray')
            ax_ax2.set_title(f'Class: {self.categories[self.labels[img_index]]}', fontsize=10)
            ax_ax2.axis('off')

            ax_ax3.bar(np.arange(256), self.lbp_vectors[img_index], color='black')
            ax_ax3.set_title(f'Class: {self.categories[self.labels[img_index]]}', fontsize=10)
            ax_ax3.set_xticks([])
            ax_ax3.set_yticks([])

        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        ax2.axis('off')

        ax3.spines['right'].set_visible(False)
        ax3.spines['top'].set_visible(False)
        ax3.spines['left'].set_visible(False)
        ax3.spines['bottom'].set_visible(False)
        ax3.axis('off')

        plt.tight_layout()
        plt.show()

    @staticmethod
    def __calculate_lbp(image: np.ndarray) -> np.ndarray:
        warnings.warn("Deprecated. Use `faster_calculate_lbp` instead.", DeprecationWarning)
        lbp_image = np.zeros_like(image)
        for i in range(1, lbp_image.shape[0] - 1):
            for j in range(1, lbp_image.shape[1] - 1):
                bin_str = ''
                center_value = image[i, j]
                bin_str += '1' if image[i - 1, j - 1] >= center_value else '0'
                bin_str += '1' if image[i - 1, j] >= center_value else '0'
                bin_str += '1' if image[i - 1, j + 1] >= center_value else '0'
                bin_str += '1' if image[i, j + 1] >= center_value else '0'
                bin_str += '1' if image[i + 1, j + 1] >= center_value else '0'
                bin_str += '1' if image[i + 1, j] >= center_value else '0'
                bin_str += '1' if image[i + 1, j - 1] >= center_value else '0'
                bin_str += '1' if image[i, j - 1] >= center_value else '0'

                lbp_image[i, j] = int(bin_str, 2)

        return lbp_image

    @staticmethod
    def __faster_calculate_lbp(gray_image: np.ndarray) -> np.ndarray:
        padded_image = np.pad(gray_image, pad_width=1, mode='edge')
        lbp_image = np.zeros_like(gray_image, dtype=np.int32)

        # Define the offsets for the 8 neighbors
        offsets = [(i, j) for i in range(-1, 2) for j in range(-1, 2) if (i, j) != (0, 0)]

        for idx, (di, dj) in enumerate(offsets):
            # Shift the padded image using the offsets
            shifted_image = padded_image[1 + di: 1 + di + gray_image.shape[0], 1 + dj: 1 + dj + gray_image.shape[1]]
            # Update the binary representation of the LBP image
            lbp_image += (shifted_image >= padded_image[1:-1, 1:-1]) << idx

        return lbp_image.astype(np.uint8)

    @staticmethod
    def __calculate_lbp_vector(lbp_image: np.ndarray) -> np.ndarray:
        hist, _ = np.histogram(lbp_image.ravel(), bins=np.arange(0, 256 + 1), range=(0, 256))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-06)
        return hist

    def export_csv(
            self,
            output_dir: str,
            train_test_split: bool = False,
            train_ratio: float = 0.8
    ):
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
        if train_test_split:
            train_csv = output_dir / "train.csv"
            train_csv.unlink() if train_csv.exists() else None
            test_csv = output_dir / "test.csv"
            test_csv.unlink() if test_csv.exists() else None
            train_size = int(len(self) * train_ratio)
            train_indices = np.random.choice(len(self), train_size, replace=False)
            test_indices = np.setdiff1d(np.arange(len(self)), train_indices)
            self.__export_csv(self[train_indices], train_csv)
            self.__export_csv(self[test_indices], test_csv)

        else:
            output_csv = output_dir / "dataset.csv"
            output_csv.unlink() if output_csv.exists() else None
            self.__export_csv(self, output_csv)

    @staticmethod
    def __export_csv(dataset, output_csv: Path):
        with open(output_csv, 'w') as f:
            for image, label, category, lbp_image, lbp_vector in dataset:
                f.write(f"{label},")
                f.write(','.join(map(str, lbp_vector)))
                f.write("\n")


class LocalBinaryPatternsDataset:
    def __init__(self):
        self.lbp_vectors = []
        self.labels = []

    def from_lbp_image_dataset(self, dataset: LocalBinaryPatternsImageClassifierDataset):
        self.lbp_vectors = dataset.lbp_vectors
        self.labels = dataset.labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.lbp_vectors[idx], self.labels[idx]
        elif isinstance(idx, slice):
            return [(self.lbp_vectors[i], self.labels[i]) for i in range(*idx.indices(len(self)))]
        elif isinstance(idx, list) or isinstance(idx, np.ndarray):
            return [(self.lbp_vectors[i], self.labels[i]) for i in idx]
        else:
            raise ValueError(f"Invalid index type {type(idx)}")

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def shuffle(self):
        indices = np.arange(len(self))
        np.random.shuffle(indices)
        self.lbp_vectors = [self.lbp_vectors[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]

    def export_csv(self, output_csv: str):
        with open(output_csv, 'w') as f:
            for v, label in self:
                f.write(f"{label},")
                f.write(','.join(map(str, v)))
                f.write("\n")

    def load_csv(self, input_csv: str):
        with open(input_csv, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                self.labels.append(int(parts[0]))
                self.lbp_vectors.append(list(map(float, parts[1:])))


if __name__ == '__main__':
    dataset_test = LocalBinaryPatternsImageClassifierDataset()
    dataset_test.load_images(root='../data/prev/A', limit=10, channel_flatten=True)
    print(dataset_test)
    dataset_test.save_images(output_dir='output')
    dataset_test.save_lbp_images(output_dir='output-ldp')
    dataset_test.export_csv(output_dir='output', train_test_split=True, train_ratio=0.8)
    dataset_test.export_csv(output_dir='output', train_test_split=False)
    dataset_test.overview()
    print("Done")
