import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

from datasets.base import BaseImageClassifierDataset

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class ImageAugmentationDataset(BaseImageClassifierDataset):
    def __init__(self):
        super(ImageAugmentationDataset, self).__init__()
        self.augmented_images = []
        self.augmented_labels = []

    def __str__(self):
        return (
            f'{self.__class__.__name__}('
            f'size={len(self)}, '
            f'num_classes={len(self.categories)}, '
            f'categories={self.categories}, '
            f'augmented_size={len(self.augmented_images)})'
        )

    def __repr__(self):
        return self.__str__()

    def overview(self):
        figure = plt.figure(figsize=(10, 12), dpi=300)
        gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])

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
        labels, counts = np.unique(self.augmented_labels, return_counts=True)
        categories = [self.categories[lb] for lb in labels]
        ax2.bar(categories, counts)
        ax2.set_xticks(categories)
        plt.xticks(rotation=45, fontstyle='italic', fontsize=10)
        ax2.set_xlabel('Class')
        ax2.set_ylabel('Count')
        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.set_title('Augmented Image Distribution')

        ax3 = figure.add_subplot(gs[3])
        inner_gs = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs[3], wspace=0.1, hspace=0.1)
        for i in range(9):
            ax = figure.add_subplot(inner_gs[i])
            img_index = np.random.randint(len(self.augmented_images))
            ax.imshow(self.augmented_images[img_index].astype(np.uint8))
            ax.set_title(f'Class: {self.categories[self.augmented_labels[img_index]]}', fontsize=10)
            ax.axis('off')
        ax3.spines['right'].set_visible(False)
        ax3.spines['top'].set_visible(False)
        ax3.spines['left'].set_visible(False)
        ax3.spines['bottom'].set_visible(False)
        ax3.axis('off')

        plt.tight_layout()
        plt.show()

    def from_base_dataset(self, dataset: BaseImageClassifierDataset):
        self.images = dataset.images
        self.labels = dataset.labels
        self.categories = dataset.categories

    def apply_augmentation(
            self, aug_ratio: float = 0.5, gaussian_noise: bool = False, mix_patch: bool = False, **kwargs
    ):
        if 0 < aug_ratio < 1:
            num_augmented = int(len(self) * aug_ratio)
            indices = np.arange(len(self))
            np.random.shuffle(indices)
            images = [self.images[i] for i in indices]
            images = images[:num_augmented]
            labels = [self.labels[i] for i in indices]
            labels = labels[:num_augmented]
            augmented_images, augmented_labels = self.__apply_augmentation(
                images, labels, gaussian_noise, mix_patch
            )
            self.augmented_images += augmented_images
            self.augmented_labels += augmented_labels
            self.images += augmented_images
            self.labels += augmented_labels

        elif aug_ratio >= 1:
            augmented_images = []
            augmented_labels = []
            while len(augmented_images) < aug_ratio * len(self):
                # Apply augmentation to whole dataset if number of augmented images is not enough
                if len(augmented_images) + len(self) < aug_ratio * len(self):
                    _augmented_images, _augmented_labels = self.__apply_augmentation(
                        self.images, self.labels, gaussian_noise, mix_patch, **kwargs
                    )
                    augmented_images += _augmented_images
                    augmented_labels += _augmented_labels
                else:
                    num_augmented = int(aug_ratio * len(self) - len(augmented_images))
                    indices = np.arange(len(self))
                    np.random.shuffle(indices)
                    images = [self.images[i] for i in indices]
                    images = images[:num_augmented]
                    labels = [self.labels[i] for i in indices]
                    labels = labels[:num_augmented]
                    _augmented_images, _augmented_labels = self.__apply_augmentation(
                        images, labels, gaussian_noise, mix_patch, **kwargs
                    )
                    augmented_images += _augmented_images
                    augmented_labels += _augmented_labels
            self.augmented_images += augmented_images
            self.augmented_labels += augmented_labels
            self.images += augmented_images
            self.labels += augmented_labels

        else:
            raise ValueError("Augmentation ratio should be greater than 0.")

    def __apply_augmentation(
            self, images: List[np.ndarray], labels: List[int], gaussian_noise: bool, mix_patch: bool, **kwargs
    ) -> Tuple[List[np.ndarray], List[int]]:
        if gaussian_noise and mix_patch:
            images_for_gaussian_noise = images[:len(images) // 2]
            labels_for_gaussian_noise = labels[:len(images) // 2]
            images_for_mix_patch = images[len(images) // 2:]
            labels_for_mix_patch = labels[len(images) // 2:]
            _mean, _std = kwargs.get('mean', 0), kwargs.get('std', 1)  # Default mean and std for Gaussian noise
            _patch_num = kwargs.get('patch_num', 5)  # Default patch number for mix patch
            augmented_images = [self.__gaussian_noise(image, _mean, _std) for image in images_for_gaussian_noise]
            augmented_labels = labels_for_gaussian_noise
            grouped_images = {}
            for i, label in enumerate(labels_for_mix_patch):
                if label not in grouped_images:
                    grouped_images[label] = []
                grouped_images[label].append(images_for_mix_patch[i])
            for label, images in grouped_images.items():
                augmented_images += self.__mix_patch(images, _patch_num)
                augmented_labels += [label] * len(images)
            return augmented_images, augmented_labels

        elif gaussian_noise:
            _mean, _std = kwargs.get('mean', 0), kwargs.get('std', 1)
            augmented_images = [self.__gaussian_noise(image, _mean, _std) for image in images]
            return augmented_images, labels
        elif mix_patch:
            _patch_num = kwargs.get('patch_num', 5)
            grouped_images = {}
            for i, label in enumerate(labels):
                if label not in grouped_images:
                    grouped_images[label] = []
                grouped_images[label].append(images[i])
            augmented_images = []
            augmented_labels = []
            for label, images in grouped_images.items():
                augmented_images += self.__mix_patch(images, _patch_num)
                augmented_labels += [label] * len(images)
            return augmented_images, augmented_labels
        else:
            raise ValueError("At least one augmentation method should be selected.")

    @staticmethod
    def __gaussian_noise(image: np.ndarray, mean: float = 0, std: float = 1) -> np.ndarray:
        """
        Add Gaussian noise to the image.

        Args:
            image (np.ndarray): Input image.
            mean (float): Mean of the Gaussian noise.
            std (float): Standard deviation of the Gaussian noise.

        Returns:
            np.ndarray: Image with Gaussian noise.
        """
        h, w, c = image.shape
        noise = np.random.normal(mean, std, (h, w, c))
        noisy_image = image + noise
        return np.clip(noisy_image, 0, 255)

    @staticmethod
    def __mix_patch(images: List[np.ndarray], patch_num: int) -> List[np.ndarray]:
        """
        Mix patches of images.

        Args:
            images (List[np.ndarray]): List of images.
            patch_num (int): Number of patches to mix.

        Returns:
            List[np.ndarray]: List of images after mixing patches.
        """
        patch_images = [
            patch for image in images for patch in ImageAugmentationDataset.__split_image(image, patch_num)
        ]
        augmented_images = []
        random.shuffle(patch_images)
        for i, image in enumerate(images):
            image = np.hstack(patch_images[i * patch_num: (i + 1) * patch_num])
            cv2.resize(image, images[0].shape[:2])
            augmented_images.append(image)
        return augmented_images

    @staticmethod
    def __split_image(image: np.ndarray, patch_num):
        h, w, c = image.shape
        patch_width = w // patch_num
        return [
            image[:, i * patch_width: (i + 1) * patch_width, :] for i in range(patch_num)
        ]

    def save_augmented_images(self, output_dir: str, fmt: str = 'jpg'):
        output_dir = Path(output_dir)
        for i, (image, label) in enumerate(zip(self.augmented_images, self.augmented_labels)):
            if not Path(output_dir / self.categories[label]).exists():
                Path(output_dir / self.categories[label]).mkdir(parents=True, exist_ok=True)
            print(f"-- Saving augmented image {i + 1}/{len(self.augmented_images)}")
            cv2.imwrite(
                str(output_dir / self.categories[label] / f"{label}-{i}.augmented.{fmt}"),
                image
            )


if __name__ == '__main__':
    # dataset_test = BaseImageDataset()
    # dataset_test.load_images(root='../data/A', categories=['busy', 'free'], limit=10)
    # print(dataset_test)
    # dataset_test.save_images(output_dir='output')
    dataset_test = ImageAugmentationDataset()
    dataset_test.load_images(root='../data/prev/A', limit=20)
    print(dataset_test)
    dataset_test.apply_augmentation(aug_ratio=2, gaussian_noise=True, mix_patch=True)
    print(dataset_test)
    dataset_test.save_augmented_images(output_dir='output-augmented')
    dataset_test.overview()
    # dataset_test.save_images(output_dir='output')
    # print("Done")
