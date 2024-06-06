from pathlib import Path

from torch.utils.data import DataLoader
import torch
from torchvision import datasets, transforms
from datasets.base import BaseImageClassifierDataset
from models.common import ClassifierBackend


class ConvolutionNeuralNetworkClassifierBackend(ClassifierBackend):
    def __init__(
            self,
            model_name: str = "resnet18",
            device: str = "cuda:0",
            batch_size: int = 32,
            epochs: int = 100,
            learning_rate: float = 0.001,
            save_dir: str = "./output",
    ):
        self.model_name = model_name
        self.device = torch.device("cuda") if device == 'cuda' and torch.cuda.is_available() else torch.device("cpu")
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.model = None

    def __str__(self):
        return f"{self.model_name} (device: {self.device})"

    def init_model(self, num_classes: int) -> torch.nn.Module:
        model = torch.hub.load(
            "pytorch/vision:v0.6.0",
            self.model_name,
            weights=True
        )
        if self.model_name in ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]:
            num_features = model.fc.in_features
            model.fc = torch.nn.Linear(num_features, num_classes)
        elif self.model_name in ["alexnet", "vgg11", "vgg13", "vgg16", "vgg19"]:
            num_features = model.classifier[6].in_features
            model.classifier[6] = torch.nn.Linear(num_features, num_classes)
        elif self.model_name in ["squeezenet1_0", "squeezenet1_1"]:
            model.classifier[1] = torch.nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
            model.num_classes = num_classes
        elif self.model_name in ["densenet121", "densenet169", "densenet161", "densenet201"]:
            num_features = model.classifier.in_features
            model.classifier = torch.nn.Linear(num_features, num_classes)
        elif self.model_name in ["inception_v3"]:
            num_features = model.fc.in_features
            model.fc = torch.nn.Linear(num_features, num_classes)
        else:
            raise ValueError(f"Not support this model yet: {self.model_name}")
        return model.to(self.device)

    def train(self, train_data: BaseImageClassifierDataset, **kwargs):
        pass

    def predict(self, data, **kwargs):
        pass

    def evaluate(self, test_data, **kwargs):
        pass
