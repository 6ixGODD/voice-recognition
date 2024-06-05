import os
import random
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np

from dataset import ImageData

RANDOM_SEED = 42
random.seed(RANDOM_SEED)


@dataclass
class ImageWithLabel:
    image: np.ndarray
    label: str


def gaussian_blur(image_list: List[ImageWithLabel], kernel_size: int = 5) -> List[ImageWithLabel]:
    """
    Apply Gaussian blur to the image list.

    Args:
        image_list (List[ImageData]): List of ImageData objects.
        kernel_size (int): Kernel size for the Gaussian blur.

    Returns:
        List[ImageWithLabel]: List of ImageData objects after applying Gaussian blur.
    """
    return [
        ImageWithLabel(
            image=cv2.GaussianBlur(image_data.image, (kernel_size, kernel_size), 0),
            label=image_data.label
        )
        for image_data in image_list
    ]


def mix_patch(image_list: List[ImageWithLabel], patch_num: int = 3) -> List[ImageWithLabel]:
    """
    Mix patches of the image list.

    Args:
        image_list (List[ImageWithLabel]): List of ImageData objects.
        patch_num (int): Number of patches to mix.

    Returns:
        List[ImageWithLabel]: List of ImageData objects after mixing patches.

    """
    grouped_images = {}
    for image_data in image_list:
        if image_data.label not in grouped_images:
            grouped_images[image_data.label] = []
        grouped_images[image_data.label].append(image_data.image)

    aug_image = []
    for label, images in grouped_images.items():
        print(f"Mixing patches for label {label}")
        patch_images = [
            patch for image in images for patch in __split_image(image, patch_num)
        ]
        random.shuffle(patch_images)
        for i, image in enumerate(images):
            image = np.hstack(patch_images[i * patch_num: (i + 1) * patch_num])
            cv2.resize(image, images[0].shape[:2])
            aug_image.append(ImageWithLabel(image=image, label=label))

    return aug_image

def __split_image(image: np.ndarray, patch_num):
    _, y, _ = image.shape
    patch_width = y // patch_num
    return [
        image[:, i * patch_width: (i + 1) * patch_width, :]
        for i in range(patch_num)
    ]


def augment_images(
        image_list: List[ImageWithLabel],
        augment_ratio: float = 0.5,
        patch_num: int = 5,
        kernel_size: int = 5
) -> List[ImageWithLabel]:
    """
    Augment images in the image list.

    Args:
        image_list (List[ImageWithLabel]): List of ImageWithLabel objects.
        augment_ratio (float): Ratio of augmentation.
        patch_num (int): Size of the patch to mix.
        kernel_size (int): Kernel size for the Gaussian blur.

    Returns:
        List[ImageWithLabel]: List of ImageWithLabel objects after augmenting images.
    """
    assert 0 < augment_ratio < 1, "Augment ratio should be in the range (0, 1]."
    random.shuffle(image_list)
    image_list = image_list[:int(len(image_list) * augment_ratio)]
    aug_image = mix_patch(
        image_list[len(image_list) // 2:],
        patch_num
    )
    aug_image += gaussian_blur(
        image_list[:len(image_list) // 2],
        kernel_size
    )
    return aug_image


def __load_images(
        root: str,
        category: List[str],
        limit: int = 0,
) -> List[ImageWithLabel]:
    data_list = []
    print("== Loading Image")
    num_class = len(category)

    for i, c in enumerate(category):
        path = os.path.join(root, c)
        for j, f in enumerate(os.listdir(path)):
            if f.endswith(".jpg") or f.endswith(".png"):
                print(f"-- Processing Image {f} with label {i} in {c}")
                data_list.append(
                    process_image(
                        file_path=os.path.join(path, f),
                        label=c,
                    )
                )
                if limit != 0 and j >= limit / num_class:
                    break

    return data_list


def process_image(file_path: str, label: str) -> ImageWithLabel:
    image = cv2.imread(file_path)
    return ImageWithLabel(image=image, label=label)


def __save_images(image_list: List[ImageWithLabel], output_dir: str):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, image_data in enumerate(image_list):
        if not os.path.exists(os.path.join(output_dir, str(image_data.label))):
            os.makedirs(os.path.join(output_dir, str(image_data.label)))
        print(f"Saving image {i + 1}/{len(image_list)}")
        cv2.imwrite(
            os.path.join(output_dir, str(image_data.label), f"aug.{i}.jpg"),
            image_data.image
        )


if __name__ == '__main__':
    _image_list = __load_images(
        "data/SpectrogramImages",
        ["baohui", "bochen", "lai", "mengyang", "peiyu", "tian", "xiang", "xinyu", "yaobing", "yaoyi",
         "yaoyuan", "yongqing", "zhaoyu"],
        # ['busy', 'free'],
        # limit=10,
    )
    _aug_image_list = augment_images(_image_list)
    __save_images(_aug_image_list, "data/augmentedSpectrogramImages")


