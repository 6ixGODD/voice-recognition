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


def faster_calculate_lbp(image: np.ndarray) -> np.ndarray:
    padded_image = np.pad(image, pad_width=1, mode='edge')
    lbp_image = np.zeros_like(image, dtype=np.int32)

    # Define the offsets for the 8 neighbors
    offsets = [(i, j) for i in range(-1, 2) for j in range(-1, 2) if (i, j) != (0, 0)]

    for idx, (di, dj) in enumerate(offsets):
        # Shift the padded image using the offsets
        shifted_image = padded_image[1 + di: 1 + di + image.shape[0], 1 + dj: 1 + dj + image.shape[1]]
        # Update the binary representation of the LBP image
        lbp_image += (shifted_image >= padded_image[1:-1, 1:-1]) << idx

    return lbp_image.astype(np.uint8)


def calculate_lbp_hist(lbp_image: np.ndarray) -> np.ndarray:
    hist, _ = np.histogram(lbp_image.ravel(), bins=np.arange(0, 256 + 1), range=(0, 256))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-06)
    return hist


def calculate_euclid_distance(v1, v2):
    return np.linalg.norm(v1 - v2)


def calculate_manhattan_distance(v1, v2):
    return np.sum(np.abs(v2 - v1))


def calculate_cosine_similarity(v1, v2) -> float:
    result = 1 - (np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    return result

def calculate_minkowski_distance(v1, v2, p=3):
    return np.sum(np.abs(v1 - v2) ** p) ** (1 / p)


def calculate_chebyshev_distance(v1, v2):
    return np.max(np.abs(v1 - v2))


def calculate_bray_curtis_distance(v1, v2):
    return np.sum(np.abs(v1 - v2)) / np.sum(np.abs(v1 + v2))
