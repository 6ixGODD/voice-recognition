import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parents[1]
sys.path.append(str(ROOT))

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train a CNN-based model for classification',
        add_help=True,
        usage='python tools/train.py [--help] --source <path> --model <name> [--weights <path>] [--pretrained] '
              '[--save-dir <path>] [--save-jit] [--save-onnx] [--input-size WIDTH HEIGHT] [--device DEVICE] '
              '[--batch-size SIZE] [--epochs EPOCHS] [--learning-rate LR] [--early-stopping] [--patience PATIENCE] '
              '[--delta DELTA] [--decay DECAY] [--ckpt-interval INTERVAL]',
    )

    parser.add_argument(
        '--source', '-s',
        type=str,
        required=True,
        help='Source of train images, should be a directory'
    )

    parser.add_argument(
        '--weights', '-w',
        type=str,
        default=None,
        help='Path to model weights, should be PyTorch model.'
    )

    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help='Model to use, should be a model name in `pytorch/vision:v0.6.0`'
    )

    parser.add_argument(
        '--pretrained',
        action='store_true',
        help='Use pre-trained model weights, default is `False`. If `True`, `--weights` will be ignored'
    )

    parser.add_argument(
        '--save-dir', '-sd',
        type=str,
        default='./output/train',
        help='Directory to save output'
    )

    parser.add_argument(
        '--save-jit',
        action='store_true',
        help='Save model as JIT model (`TorchScript`)'
    )

    parser.add_argument(
        '--save-onnx',
        action='store_true',
        help='Save model as ONNX model'
    )

    parser.add_argument(
        '--input-size',
        type=int,
        nargs=2,
        default=[224, 224],
        metavar=('WIDTH', 'HEIGHT'),
        help='Input size for the model, default is `224 224`'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        choices=['cpu', 'cuda', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'],
        help='Device to run inference on, default is `"cuda:0"`'
    )

    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=32,
        help='Batch size for inference, default is `32`'
    )

    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=10,
        help='Number of epochs to train the model, default is `100`'
    )

    parser.add_argument(
        '--learning-rate', '-lr',
        type=float,
        default=0.001,
        help='Learning rate for training the model, default is `0.001`'
    )

    parser.add_argument(
        '--early-stopping',
        action='store_true',
        help='Enable early stopping'
    )

    parser.add_argument(
        '--patience',
        type=int,
        default=5,
        help='Patience for early stopping, default is `5`'
    )

    parser.add_argument(
        '--delta',
        type=float,
        default=0.0,
        help='Delta for early stopping, default is `0.0`'
    )

    parser.add_argument(
        '--decay',
        type=float,
        default=0.0,
        help='Decay for learning rate scheduler, default is `0.0`'
    )

    parser.add_argument(
        '--ckpt-interval',
        type=int,
        default=10,
        help='Save checkpoint interval, default is `10`'
    )

    return parser.parse_args()


def run(
        source: str,
        weights: str,
        model: str,
        pretrained: bool,
        save_dir: str,
        save_jit: bool,
        save_onnx: bool,
        input_size: list,
        device: str,
        batch_size: int,
        epochs: int,
        learning_rate: float,
        early_stopping: bool,
        patience: int,
        delta: float,
        decay: float,
        ckpt_interval: int
):
    assert Path(source).is_dir(), f"Source directory `{source}` not found"
    assert not weights or Path(weights).exists() or pretrained, f"Model weights `{weights}` not found"

    from torchvision import transforms
    from datasets.torch import TorchImageClassificationDataset
    from models.cnn import ConvolutionNeuralNetworkClassifierBackend
    from utils.files import increment_path

    save_dir = increment_path(save_dir)
    source = Path(source)

    tfs = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )
    classifier = ConvolutionNeuralNetworkClassifierBackend(
        model_name=model,
        pretrained=pretrained,
        weight_path=weights,
        input_size=(input_size[0], input_size[1]),
        device=device,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        early_stopping=early_stopping,
        patience=patience,
        delta=delta,
        decay=decay,
        ckpt_interval=ckpt_interval,
        save_dir=str(save_dir),
        save_jit=save_jit,
        save_onnx=save_onnx,
    )

    train_dataset = TorchImageClassificationDataset(transform=tfs)
    train_dataset.load_images(root=str(source / 'train'))
    val_dataset = TorchImageClassificationDataset(transform=tfs)
    val_dataset.load_images(root=str(source / 'val'))

    classifier.train(train_dataset, val_dataset)


if __name__ == '__main__':
    args = parse_args()
    run(
        args.source,
        args.weights,
        args.model,
        args.pretrained,
        args.save_dir,
        args.save_jit,
        args.save_onnx,
        args.input_size,
        args.device,
        args.batch_size,
        args.epochs,
        args.learning_rate,
        args.early_stopping,
        args.patience,
        args.delta,
        args.decay,
        args.ckpt_interval
    )
