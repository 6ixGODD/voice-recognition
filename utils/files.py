import hashlib
import uuid
from pathlib import Path
import shutil


def increment_path(path: str, separator: str = "-") -> Path:
    """
    Automatically increment path, i.e. weights/exp -> weights/exp{sep}2, weights/exp{sep}3, ...

    Args:
        path (str): path to be incremented
        separator (str): separator between path and number

    Returns:
        Path: incremented path

    """
    path = Path(path)
    if path.exists():
        path, suffix = (
            (path.with_suffix(""), path.suffix) if path.is_file() else (path, "")
        )
        for n in range(2, 9999):
            p = f"{path}{separator}{n}{suffix}"
            if not Path(p).exists():
                path = Path(p)
                break
        path.mkdir(parents=True, exist_ok=True)  # make directory
    else:
        path.mkdir(parents=True, exist_ok=True)
    return path


def rename(target_dir: str, mode: str):
    """
    Rename the files in the directory.

    Args:
        target_dir (str): Path to the directory.
        mode (str): Mode to rename the files.

    """
    target_dir = Path(target_dir)
    if mode == "uuid":
        for i, f in enumerate(target_dir.iterdir()):
            new_name = f"{uuid.uuid4().hex}{f.suffix}"
            f.rename(target_dir / new_name)
    elif mode == "hash":
        for i, f in enumerate(target_dir.iterdir()):
            new_name = f"{hashlib.md5(f.name.encode()).hexdigest()}{f.suffix}"
            f.rename(target_dir / new_name)
    elif mode == "increment":
        for i, f in enumerate(target_dir.iterdir()):
            new_name = f"{i}{f.suffix}"
            f.rename(target_dir / new_name)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def ttv_split(train: float, test: float, source: str, target: str):
    """
    Split the dataset into train, test, and validation set.

    Args:
        train (float): Percentage of the training set.
        test (float): Percentage of the test set.
        source (str): Path to the source directory.
        target (str): Path to the target directory.

    """
    source = Path(source)
    target = Path(target)
    assert source.exists(), f"Source {source} does not exist!"
    target.mkdir(parents=True, exist_ok=True)
    for label, c in enumerate(source.iterdir()):
        if c.is_dir():
            for i, f in enumerate(c.iterdir()):
                (target / "train" / str(label)).mkdir(parents=True, exist_ok=True)
                (target / "test" / str(label)).mkdir(parents=True, exist_ok=True)
                (target / "val" / str(label)).mkdir(parents=True, exist_ok=True)
                if i < int(train * len(list(c.iterdir()))):
                    shutil.copy(f, target / "train" / str(label) / f.name)
                elif i < int((train + test) * len(list(c.iterdir()))):
                    shutil.copy(f, target / "test" / str(label) / f.name)
                else:
                    shutil.copy(f, target / "val" / str(label) / f.name)


if __name__ == '__main__':
    ttv_split(0.7, 0, '../data/SpectrogramImages', '../data_split')
