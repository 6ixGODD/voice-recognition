import os
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)


@dataclass
class ImageData:
    image: np.ndarray
    label: int  # 0 / 1
    lbp: np.ndarray
    lbp_vector: np.ndarray


def calculate_lbp(image: np.ndarray) -> np.ndarray:
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


def calculate_lbp_hist(lbp_image: np.ndarray) -> np.ndarray:
    hist, _ = np.histogram(lbp_image.ravel(), bins=np.arange(0, 256 + 1), range=(0, 256))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-06)
    return hist


def load_images(root: str, category: List[str], limit: int = 0) -> List[ImageData]:
    data_list = []
    print("== Loading Image")
    num_class = len(category)
    for i, c in enumerate(category):
        path = os.path.join(root, c)
        for j, f in enumerate(os.listdir(path)):
            if f.endswith(".jpg"):
                print(f"-- Processing Image {f} with label {i}")
                image = cv2.imread(os.path.join(path, f), cv2.IMREAD_GRAYSCALE)
                lbp_image = calculate_lbp(image=image)
                hist = calculate_lbp_hist(lbp_image=lbp_image)
                data_list.append(ImageData(image=image, label=i, lbp=lbp_image, lbp_vector=hist))
            if limit != 0 and j >= limit / num_class:
                break

    return data_list


def export(
        data_list: List[ImageData],
        limit: int = 0,
        save_dir: str = './output',
        split: float = 0.5
):
    print("== Export Data")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.random.shuffle(np.array(data_list))
    if limit != 0:
        data_list = data_list[:limit + 1]
    split_index = int(split * len(data_list))
    train_set = data_list[:split_index]
    test_set = data_list[split_index:]
    print("-- Processing train set")
    __save_csv(train_set, "train.csv", save_dir)
    print("-- Processing test set")
    __save_csv(test_set, "test.csv", save_dir)


def __save_csv(data_list: List[ImageData], filename, save_dir: str = './output'):
    with open(os.path.join(save_dir, filename), "w") as f:
        for data in data_list:
            f.write(f"{data.label},{','.join(map(str, data.lbp_vector))}\n")


def run(
        category: List[str],
        limit: int = 0,
        save_dir: str = './output',
):
    data_a = load_images("dataset/A", category=category, limit=limit)
    data_b = load_images("dataset/B", category=category, limit=limit)
    dataset = np.append(np.array(data_a), np.array(data_b))
    export(data_list=dataset, limit=limit, save_dir=save_dir)


if __name__ == "__main__":
    run(["busy", "free"], limit=200, save_dir='./output')
