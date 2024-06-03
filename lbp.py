from dataclasses import dataclass

import numpy as np

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


def calculate_euclid_distance(v1, v2):
    return np.linalg.norm(v1 - v2)


def calculate_manhattan_distance(v1, v2):
    return np.sum(np.abs(v2 - v1))


def calculate_cosine_similarity(v1, v2):
    return 1 - (np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def calculate_jaccard_similarity(v1, v2):
    return 1 - (np.sum(np.minimum(v1, v2)) / np.sum(np.maximum(v1, v2)))


def calculate_dice_similarity(v1, v2):
    return 1 - (2 * np.sum(np.minimum(v1, v2)) / (np.sum(v1) + np.sum(v2)))


def calculate_hamming_distance(v1, v2):
    return np.sum(v1 != v2)
