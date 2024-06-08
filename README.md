# Voice Classification

## Prerequisites

* Python 3.x
* Pytorch 2.x
* FFmpeg

### Install dependencies

```shell
pip install -r requirements.txt
```

### Install FFmpeg

```shell
pip install ffmpeg-downloader

ffdl install --add-path  
```

## Project layout

```shell
.
├── _              # Temporary scripts
├── applications   # Analyser & Collector applications
├── data           # Data directory
│   ├── audios     # Audio files
│   └── images     # Image files
├── datasets       # Dataset classes
├── models         # Model classes
├── tools          # Command line tools
└── utils          # Utility functions
```

## Usage

### Prepare dataset

```shell
python python tools/dataset.py [--help] --from-images <path> --from-audios <path> --save-dir <path>
       [--save-waveforms] [--channel-flatten] [--save-lbp-images] [--save-lbp-vectors] [--split]
       [--split-ratio TRAIN VAL TEST] [--shuffle] [--resize WIDTH HEIGHT] [--padding-color R G B]
       [--augment] [--augment-ratio RATIO] [--gaussian-noise] [--mix-patch] [--format {jpg,jpeg,png}]
```

* `--from-images`: Path to images directory. If provided, `from-audios` should be `None`.
* `--from-audios`: Path to audios directory. If provided, `from-images` should be `None`.
* `--save-dir`: Directory to save output.
* `--save-waveforms`: Save audio waveforms.
* `--channel-flatten`: Enable channel flatten when generating LBP features.
* `--save-lbp-images`: Save LBP images.
* `--save-lbp-vectors`: Save LBP feature vectors as CSV file.
* `--split`: Split the dataset into train, validation, and test sets.
* `--split-ratio`: Split ratio for train, validation, and test sets.
* `--shuffle`: Shuffle the dataset before splitting.
* `--resize`: Resize the image to the specified width and height.
* `--padding-color`: Padding color for the image in RGB format for resizing.
* `--augment`: Augment the dataset.
* `--augment-ratio`: Augmentation ratio. number of augmented images = `augment_ratio` * number of original images.
* `--gaussian-noise`: Enable Gaussian noise augmentation.
* `--mix-patch`: Enable mix patch augmentation.
* `--format`: Format to save the image(s), default is `jpg`, include `jpg`, `jpeg`, `png`.
* `--help`: Show help message.

e.g.
from images

```shell
python tools/dataset.py --from-images data/images --save-dir ./output/dataset --channel-flatten --save-lbp-images --save-lbp-vectors --split --split-ratio 0.8 0.1 0.1 --shuffle --resize 224 224 --padding-color 0 0 0 --augment --augment-ratio 1 --gaussian-noise --mix-patch --format jpg
```

from audios

```shell
python tools/dataset.py --from-audios data/audios --save-dir ./output/images --save-waveforms --format jpg
```

### Train model

```shell
python tools/train.py [--help] --source <path> --model <name> [--weights <path>] [--pretrained] 
       [--save-dir <path>] [--save-jit] [--save-onnx] [--input-size WIDTH HEIGHT] [--device DEVICE] 
       [--batch-size SIZE] [--epochs EPOCHS] [--learning-rate LR] [--early-stopping] [--patience PATIENCE] 
       [--delta DELTA] [--decay DECAY] [--ckpt-interval INTERVAL]
```

* `--source`: Source of train images, should be a directory.
* `--model`: Model to use, should be a model name in `pytorch/vision:v0.6.0`.
* `--weights`: Path to model weights, should be PyTorch model.
* `--pretrained`: Use pre-trained model weights, default is `False`. If `True`, `--weights` will be ignored.
* `--save-dir`: Directory to save output.
* `--save-jit`: Save model as JIT model (`TorchScript`).
* `--save-onnx`: Save model as ONNX model.
* `--input-size`: Input size for the model, default is `224 224`.
* `--device`: Device to run inference on, default is `"cuda:0".
* `--batch-size`: Batch size for inference, default is `32`.
* `--epochs`: Number of epochs to train the model, default is `100`.
* `--learning-rate`: Learning rate for training the model, default is `0.001`.
* `--early-stopping`: Enable early stopping.
* `--patience`: Patience for early stopping, default is `5`.
* `--delta`: Delta for early stopping, default is `0.0`.
* `--decay`: Decay for learning rate scheduler, default is `0.0`.
* `--ckpt-interval`: Save checkpoint interval, default is `10`.
* `--help`: Show help message.

e.g.

```shell
python tools/train.py --source output/dataset/images --model resnet18 --save-dir output/resnet18 --save-jit --save-onnx --input-size 224 224 --device cuda --batch-size 32 --epochs 100 --learning-rate 0.001 --early-stopping --patience 5 --delta 0.0 --decay 0.0 --ckpt-interval 10
```

### Evaluate model

```shell
python [--help] --source <path> [--method METHOD] [--save-dir <path>] [--save-metrics] 
       [--weights <path>] [--device DEVICE]
```

* `--source`: Source of test data, should be a directory. When using `cnn` method, this should be a directory containing
  subdirectories of classes and images. When using `lbp` method, this should be a directory containing `train.csv` and `
  test.csv.
* `--method`: Method to use, default is `lbp`, can be `cnn` or `lbp`.
* `--save-dir`: Directory to save output.
* `--save-metrics`: Save metrics to CSV file.
* `--weights`: Path to model weights, should be PyTorch model. Only required for `cnn` method.
* `--device`: Device to run inference on, default is `"cuda:0"`. Only required for `cnn` method.
* `--help`: Show help message.

e.g.
lbp

```shell
python tools/eval.py --source output/dataset/lbp_vector --method lbp --save-dir output/lbp-metrics --save-metrics
```

cnn

```shell
python tools/eval.py --source output/dataset/images/test --method cnn --model resnet18 --save-dir output/cnn-metrics --save-metrics --weights output/resnet18/best_resnet18.pth --device cuda
```

### Experiment

see [exp.ipynb](exp.ipynb).

