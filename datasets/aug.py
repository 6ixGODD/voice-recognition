import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from datasets.base import BaseImageDataset

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class AugmentationDataset(BaseImageDataset):
    def __init__(self):
        super(AugmentationDataset, self).__init__()
        self.augmented_images = []
        self.augmented_labels = []

    def from_base_dataset(self, dataset: BaseImageDataset):
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
            patch for image in images for patch in AugmentationDataset.__split_image(image, patch_num)
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
    dataset_test = AugmentationDataset()
    dataset_test.load_images(root='../data/A', categories=['busy', 'free'], limit=20)
    print(dataset_test)
    dataset_test.apply_augmentation(aug_ratio=2, gaussian_noise=True, mix_patch=True)
    print(dataset_test)
    dataset_test.save_augmented_images(output_dir='output-augmented')
    # dataset_test.save_images(output_dir='output')
    # print("Done")
