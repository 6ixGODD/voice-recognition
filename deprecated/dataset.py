import os
import random
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np

from lbp import calculate_lbp_hist, faster_calculate_lbp

RANDOM_SEED = 42
random.seed(RANDOM_SEED)


@dataclass
class Row:
    label: int
    lbp_vector: np.ndarray


@dataclass
class ImageData:
    image: np.ndarray
    label: int  # 0 / 1
    lbp: np.ndarray
    lbp_vector: np.ndarray


def load_dataset(data_csv: str) -> List[Row]:
    data = np.genfromtxt(data_csv, delimiter=",")
    row_list = []
    for row in data:
        row_list.append(Row(label=int(row[0]), lbp_vector=row[1:]))
    return row_list


def load_images(
        root: str,
        category: List[str],
        limit: int = 0,
        flatten_channel: bool = False,
        augment: bool = False,
        augment_ratio: int = 0.2
) -> List[ImageData]:
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
                        label=i,
                        flatten_channel=flatten_channel
                    )
                )
                if limit != 0 and j >= limit / num_class:
                    break

    return data_list


def process_image(file_path: str, label: int, flatten_channel: bool) -> ImageData:
    if flatten_channel:
        image = cv2.imread(file_path)
        R, G, B = cv2.split(image)
        image = np.hstack((R, G, B))
    else:
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    lbp_image = faster_calculate_lbp(image=image)
    hist = calculate_lbp_hist(lbp_image=lbp_image)
    return ImageData(image=image, label=label, lbp=lbp_image, lbp_vector=hist)


def export(
        data_list: List[ImageData],
        limit: int = 0,
        save_dir: str = './output',
        split: float = 0.5,
):
    print("== Export Data")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    random.shuffle(data_list)
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
        image_dirs: List[str],
        category: List[str],
        limit: int = 0,
        save_dir: str = './output',
        flatten_channel: bool = False
):
    data_list = np.array([])
    for image_dir in image_dirs:
        data_list = np.append(
            data_list,
            np.array(load_images(
                image_dir,
                category,
                limit,
                flatten_channel))
        )
    export(data_list, limit, save_dir)


if __name__ == "__main__":
    import time

    start = time.time()
    run(
        image_dirs=["./data/SpectrogramImages-640"],
        category=["baohui", "bochen", "lai", "mengyang", "peiyu", "tian", "xiang", "xinyu", "yaobing", "yaoyi",
                  "yaoyuan", "yongqing", "zhaoyu"],
        # limit=0,
        save_dir='./output-spectrogram-flatten-640',
        flatten_channel=True
    )
    print(f"Time elapsed: {time.time() - start:.2f}s")

