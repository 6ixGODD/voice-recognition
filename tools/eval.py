import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parents[1]
sys.path.append(str(ROOT))

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Evaluate a model for classification',
        add_help=True,
        usage='python tools/eval.py [--help] --source <path> [--method METHOD] [--save-dir <path>] [--save-metrics] '
              '[--weights <path>] [--device DEVICE]',
    )

    parser.add_argument(
        '--source', '-s',
        type=str,
        required=True,
        help='Source of test data, should be a directory. When using `cnn` method, this should be a directory '
             'containing subdirectories of classes and images. When using `lbp` method, this should be a directory '
             'containing `train.csv` and `test.csv`'
    )

    parser.add_argument(
        '--method', '-m',
        type=str,
        default='lbp',
        choices=['lbp', 'cnn'],
        help='Method to use, default is `lbp`'
    )

    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Model to use, should be a model name in `pytorch/vision:v0.6.0`. Only required for `cnn` method'
    )

    parser.add_argument(
        '--save-dir', '-sd',
        type=str,
        default='./output/eval',
        help='Directory to save output'
    )

    parser.add_argument(
        '--save-metrics', '-sm',
        action='store_true',
        help='Save metrics to CSV file'
    )

    parser.add_argument(
        '--weights', '-w',
        type=str,
        default=None,
        help='Path to model weights, should be PyTorch model. Only required for `cnn` method'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        choices=['cpu', 'cuda', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'],
        help='Device to run inference on, default is `"cuda:0"`. Only required for `cnn` method'
    )

    return parser.parse_args()


def run(
        source: str,
        method: str,
        model: str,
        save_dir: str,
        save_metrics: bool,
        weights: str,
        device: str
):
    assert Path(source).is_dir(), f"Source directory `{source}` not found"
    assert method != 'lbp' or (Path(source) / 'train.csv').is_file(), f"Train data `train.csv` not found"
    assert method != 'lbp' or (Path(source) / 'test.csv').is_file(), f"Test data `test.csv` not found"
    assert method != 'cnn' or Path(weights).is_file(), f"Model weights `{weights}` not found"
    assert (
            method != 'cnn' or device in ['cpu', 'cuda', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']
    ), f"Device `{device}` not found"
    assert method != 'cnn' or model is not None, f"Model name is required for `cnn` method"

    from utils.files import increment_path

    save_dir = increment_path(save_dir)
    source = Path(source)
    if method == 'lbp':
        from datasets.lbp import LocalBinaryPatternsDataset
        from models.lbp import LocalBinaryPatternsClassifierBackend

        train_data = LocalBinaryPatternsDataset()
        test_data = LocalBinaryPatternsDataset()
        train_data.load_csv(str(source / 'train.csv'))
        test_data.load_csv(str(source / 'test.csv'))

        classifier = LocalBinaryPatternsClassifierBackend()
        classifier.train(train_data)
        metric = classifier.test(test_data)
        metric.to_csv(str(save_dir / 'metrics.csv'), index=False) if save_metrics else None

    elif method == 'cnn':
        from torchvision import transforms

        from datasets.torch import TorchImageClassificationDataset
        from models.cnn import ConvolutionNeuralNetworkClassifierBackend

        tfs = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ]
        )
        test_data = TorchImageClassificationDataset(transform=tfs)
        test_data.load_images(root=str(source))

        classifier = ConvolutionNeuralNetworkClassifierBackend(
            model_name=model,
            weight_path=weights,
            device=device
        )
        metric = classifier.test(test_data, 1)
        metric.to_csv(str(save_dir / 'metrics.csv'), index=False) if save_metrics else None


if __name__ == '__main__':
    args = parse_args()
    run(
        args.source,
        args.method,
        args.model,
        args.save_dir,
        args.save_metrics,
        args.weights,
        args.device
    )
