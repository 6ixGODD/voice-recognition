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
    assert size[0] % stride == 0 and size[1] % stride == 0, f"size {size} should be divisible by stride {stride}"
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


if __name__ == '__main__':
    # data_split
    # -- train
    #    -- 0
    #       -- 0.jpg
    #       -- 1.jpg
    #    -- 1
    #       -- 0.jpg
    #       -- 1.jpg
    # ...
    # -- test
    #    -- 0
    #       -- 0.jpg
    #       -- 1.jpg
    #    -- 1
    #       -- 0.jpg
    #       -- 1.jpg
    # ...
    # -- val
    #    -- 0
    #       -- 0.jpg
    #       -- 1.jpg
    #    -- 1
    #       -- 0.jpg
    #       -- 1.jpg
    # ...

    # Split it into train, test, and validation set and resize them all
    from pathlib import Path
    import shutil

    source = Path('../data/SpectrogramImages')
    target = Path('../data_split')
    target.mkdir(parents=True, exist_ok=True)

    for label, c in enumerate(source.iterdir()):
        if c.is_dir():
            for i, f in enumerate(c.iterdir()):
                (target / "train" / str(label)).mkdir(parents=True, exist_ok=True)
                (target / "test" / str(label)).mkdir(parents=True, exist_ok=True)
                (target / "val" / str(label)).mkdir(parents=True, exist_ok=True)
                if i < int(0.7 * len(list(c.iterdir()))):
                    shutil.copy(f, target / "train" / str(label) / f.name)
                elif i < int(0.8 * len(list(c.iterdir()))):
                    shutil.copy(f, target / "test" / str(label) / f.name)
                else:
                    shutil.copy(f, target / "val" / str(label) / f.name)

                im = cv2.imread(str(f))
                im, _ = padding_resize(im, size=(224, 224), color=(0, 0, 0))
                cv2.imwrite(str(f), im)
                print(f"Save image to {f}")
    print("Done!")
