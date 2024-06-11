import sys
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent.parents[1]
sys.path.append(str(ROOT))

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Prepare dataset',
        add_help=True,
        usage='python tools/dataset.py [--help] --from-images <path> --from-audios <path> --save-dir <path> '
              '[--save-waveforms] [--channel-flatten] [--save-lbp-images] [--save-lbp-vectors] [--split] '
              '[--split-ratio TRAIN VAL TEST] [--shuffle] [--resize WIDTH HEIGHT] [--padding-color R G B] '
              '[--augment] [--augment-ratio RATIO] [--gaussian-noise] [--mix-patch] [--format {jpg,jpeg,png}]',
    )

    parser.add_argument(
        '--from-images', '-fi',
        type=str,
        default=None,
        help='Path to images directory. If provided, `from-audios` should be `None`'
    )

    parser.add_argument(
        '--from-audios', '-fa',
        type=str,
        default=None,
        help='Path to audios directory. If provided, `from-images` should be `None`'
    )

    parser.add_argument(
        '--save-dir', '-sd',
        type=str,
        default='./output/dataset',
        help='Directory to save output'
    )

    parser.add_argument(
        '--save-waveforms', '-sw',
        action='store_true',
        help='Save audio waveforms'
    )

    parser.add_argument(
        '--channel-flatten', '-cf',
        action='store_true',
        help='Enable channel flatten when generating LBP features'
    )

    parser.add_argument(
        '--save-lbp-images', '-sli',
        action='store_true',
        help='Save LBP images'
    )

    parser.add_argument(
        '--save-lbp-vectors', '-slv',
        action='store_true',
        help='Save LBP feature vectors as CSV file'
    )

    parser.add_argument(
        '--split',
        action='store_true',
        help='Split the dataset into train, validation, and test sets'
    )

    parser.add_argument(
        '--split-ratio',
        type=float,
        nargs=3,
        default=[0.7, 0.2, 0.1],
        metavar=('TRAIN', 'VAL', 'TEST'),
        help='Split ratio for train, validation, and test sets'
    )

    parser.add_argument(
        '--shuffle',
        action='store_true',
        help='Shuffle the dataset before splitting'
    )

    parser.add_argument(
        '--resize',
        type=int,
        nargs=2,
        default=None,
        metavar=('WIDTH', 'HEIGHT'),
        help='Resize the image to the specified width and height'
    )

    parser.add_argument(
        '--padding-color',
        type=int,
        nargs=3,
        default=None,
        metavar=('R', 'G', 'B'),
        help='Padding color for the image in RGB format for resizing'
    )

    parser.add_argument(
        '--augment', '-a',
        action='store_true',
        help='Augment the dataset'
    )

    parser.add_argument(
        '--augment-ratio', '-ar',
        type=float,
        default=0.5,
        help='Augmentation ratio. number of augmented images = `augment_ratio` * number of original images'
    )

    parser.add_argument(
        "--gaussian-noise",
        action="store_true",
        help="Enable Gaussian noise augmentation"
    )

    parser.add_argument(
        "--mix-patch",
        action="store_true",
        help="Enable mix patch augmentation"
    )

    parser.add_argument(
        '--format', '-f',
        type=str,
        default='jpg',
        choices=['jpg', 'jpeg', 'png'],
        help='Format to save the image(s), default is `jpg`, include `jpg`, `jpeg`, `png`'
    )

    return parser.parse_args()


def run(
        from_images: Optional[str],
        from_audios: Optional[str],
        save_dir: str,
        save_waveforms: bool,
        channel_flatten: bool,
        save_lbp_images: bool,
        save_lbp_vectors: bool,
        split: bool,
        split_ratio: list,
        shuffle: bool,
        resize: Optional[List[int]],
        padding_color: Optional[List[int]],
        augment: bool,
        augment_ratio: float,
        gaussian_noise: bool,
        mix_patch: bool,
        fmt: str
):
    # Validate arguments
    assert from_images or from_audios, 'Either `from_images` or `from_audios` should be provided'
    assert not (from_images and from_audios), 'Only one of `from_images` or `from_audios` should be provided'
    assert from_images is None or Path(from_images).is_dir(), f"Images directory `{from_images}` not found"
    assert from_audios is None or Path(from_audios).is_dir(), f"Audios directory `{from_audios}` not found"
    assert not resize or all(i > 0 for i in resize), 'All values in `resize` should be greater than 0'
    assert (
            not padding_color or all(0 <= i <= 255 for i in padding_color)
    ), 'All values in `padding_color` should be in range [0, 255]'
    assert not split or all(0 <= i <= 1 for i in split_ratio), 'All values in `split_ratio` should be in range [0, 1]'
    assert not split or sum(split_ratio) == 1.0, f'Sum of `split_ratio` {sum(split_ratio)} should be equal to 1'
    assert 0 < augment_ratio, 'Value of `augment_ratio` should be greater than 0'
    assert not augment or gaussian_noise or mix_patch, 'At least one augmentation method should be enabled'

    from utils.files import increment_path

    save_dir = increment_path(save_dir)

    if from_images:
        from datasets.base import BaseImageClassificationDataset

        dataset = BaseImageClassificationDataset()
        print(f"Loading images from \033[92m{from_images}\033[0m")
        dataset.load_images(from_images)
        dataset.overview()

        dataset.shuffle() if shuffle else None

        dataset.resize(
            (resize[0], resize[1]),
            padding_color is not None,
            color=padding_color
        ) if resize else None

        if augment:
            from datasets.augment import ImageAugmentationDataset

            print(f"Augmenting dataset with ratio \033[92m{augment_ratio}\033[0m")
            augment_dataset = ImageAugmentationDataset()
            augment_dataset.from_base_dataset(dataset)
            augment_dataset.apply_augmentation(augment_ratio, gaussian_noise, mix_patch)
            augment_dataset.overview()
            dataset = augment_dataset
        (save_dir / 'images').mkdir(parents=True, exist_ok=True)
        print(f"Saving images to \033[92m{save_dir / 'images'}\033[0m")
        dataset.save_images(
            str(save_dir / 'images'),
            fmt=fmt,
            split=split,
            split_ratio=(split_ratio[0], split_ratio[1], split_ratio[2])
        )

        if save_lbp_images:
            from datasets.lbp import LocalBinaryPatternsImageClassificationDataset

            print(f"Saving LBP images to \033[92m{save_dir / 'lbp'}\033[0m")
            (save_dir / 'lbp').mkdir(parents=True, exist_ok=True)
            lbp_dataset = LocalBinaryPatternsImageClassificationDataset()
            lbp_dataset.from_base_dataset(dataset, channel_flatten)
            lbp_dataset.overview()
            lbp_dataset.save_lbp_images(str(save_dir / 'lbp'), fmt=fmt)
            (save_dir / 'lbp_vector').mkdir(parents=True, exist_ok=True) if save_lbp_vectors else None

            print(
                f"Saving LBP feature vectors to \033[92m{save_dir / 'lbp_vector'}\033[0m"
                ) if save_lbp_vectors else None
            lbp_dataset.export_csv(
                str(save_dir / 'lbp_vector'),
                train_test_split=split,
                train_ratio=split_ratio[0]
            ) if save_lbp_vectors else None

    elif from_audios:
        from utils.audios import transform_audio, plot_spectrogram, plot_wave

        print(f"Converting audios from \033[92m{from_audios}\033[0m to \033[92m{save_dir}\033[0m")
        for category in Path(from_audios).iterdir():
            if not category.is_dir():
                continue
            for i, audio in enumerate(category.iterdir()):
                if audio.suffix not in ['.wav', '.mp3', '.m4a']:
                    continue
                print(f"Processing \033[92m{audio}\033[0m")
                (save_dir / 'audio' / category.name).mkdir(parents=True, exist_ok=True)
                output_audio = save_dir / 'audio' / category.name / f'{i}.wav'
                transform_audio(str(audio), str(output_audio))
                (save_dir / 'spectrogram' / category.name).mkdir(parents=True, exist_ok=True)
                plot_spectrogram(
                    str(output_audio), str(
                        save_dir / 'spectrogram' / category.name / f'{i}.{fmt}'
                    )
                )
                (save_dir / 'waveform' / category.name).mkdir(parents=True, exist_ok=True) if save_waveforms else None
                plot_wave(
                    str(output_audio), str(
                        save_dir / 'waveform' / category.name / f'{i}.{fmt}'
                    )
                ) if save_waveforms else None

    print(f"Data saved to \033[92m{save_dir}\033[0m")


if __name__ == '__main__':
    args = parse_args()
    run(
        args.from_images,
        args.from_audios,
        args.save_dir,
        args.save_waveforms,
        args.channel_flatten,
        args.save_lbp_images,
        args.save_lbp_vectors,
        args.split,
        args.split_ratio,
        args.shuffle,
        args.resize,
        args.padding_color,
        args.augment,
        args.augment_ratio,
        args.gaussian_noise,
        args.mix_patch,
        args.format,
    )
