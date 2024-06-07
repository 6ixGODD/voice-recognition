from typing import Tuple

import cv2
import numpy as np


def padding_resize(
        image: np.ndarray,
        size: Tuple[int, int] = (640, 640),
        stride: int = 32,
        full_padding: bool = True,
        color: Tuple[int, int, int] = (114, 114, 114)
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Resize and pad image to size with padding color.

    Args:
        image (np.ndarray): Image to resize.
        size (tuple): New size to resize to.
        stride (int): Stride.
        full_padding (bool): Padding.
        color (tuple): Padding color.

    Returns:
        np.ndarray: Resized image.

    """
    h, w = image.shape[:2]
    scale = min(size[0] / w, size[1] / h)  # scale to resize
    new_w = int(w * scale)
    new_h = int(h * scale)
    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)  # resized, no border
    dw = size[0] - new_w if full_padding else (stride - new_w % stride) % stride  # width padding
    dh = size[1] - new_h if full_padding else (stride - new_h % stride) % stride  # height padding
    top = dh // 2
    bottom = dh - top
    left = dw // 2
    right = dw - left
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return image, (dw, dh)


def faster_calculate_lbp(gray_image: np.ndarray) -> np.ndarray:
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


def calculate_lbp_vector(lbp_image: np.ndarray) -> np.ndarray:
    hist, _ = np.histogram(lbp_image.ravel(), bins=np.arange(0, 256 + 1), range=(0, 256))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-06)
    return hist
