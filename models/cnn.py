import signal
import sys
import warnings
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchsummary
from matplotlib import gridspec
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from datasets.torch import TorchImageClassificationDataset
from models.common import ClassifierBackend

warnings.filterwarnings("ignore")  # Ignore deprecated parameter 'pretrained' warnings because it's annoying :<
plt.rcParams['font.sans-serif'] = ['Time New Roman']


class ConvolutionNeuralNetworkClassifierBackend(ClassifierBackend):
    def __init__(
            self,
            model: Optional[torch.nn.Module] = None,
            weight_path: Optional[str] = None,  # Should be a pytorch model file, not torchscript or onnx or sth else
            pretrained: bool = True,  # Load pretrained model from PyTorch Hub
            model_name: str = "resnet18",  # Default model name
            save_dir: str = "./output",
            save_jit: bool = False,  # Save model as TorchScript
            save_onnx: bool = False,  # Save model as ONNX

            input_size: Optional[Tuple[int, int]] = None,
            device: str = "cuda:0",  # Device to run inference on
            batch_size: int = 32,
            epochs: int = 100,
            learning_rate: float = 0.001,
            loss_fn: torch.nn.Module = None,
            optimizer: torch.optim.Optimizer = None,

            early_stopping: bool = False,  # Enable early stopping
            patience: int = 10,  # Number of epochs with no improvement after which training will be stopped
            delta: float = 0.0,  # Minimum change in monitored quantity to qualify as an improvement
            decay: float = 0.0,  # Decay learning rate by calculating: lr = lr * (1 / (1 + decay * epoch))
            decay_epoch: int = 20,  # Decay learning rate every `decay_epoch` epochs
            ckpt_interval: int = 10  # Save model every `ckpt_interval` epochs
    ):
        self.model = model
        self.weight_path = weight_path
        self.model_name = model_name
        device = 'cpu' if 'cuda' in device and not torch.cuda.is_available() else device  # Fallback to CPU if no CUDA
        self.device = torch.device(device)
        self.input_size = input_size or (224, 224)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.__pretrained = pretrained
        self.__save_jit = save_jit
        self.__save_onnx = save_onnx

        # Hyperparameters
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn or torch.nn.CrossEntropyLoss()
        self.optimizer = optimizer or torch.optim.Adam

        # Training options
        self.__early_stopping = early_stopping
        self.patience = patience
        self.delta = delta
        self.decay = decay
        self.decay_epoch = decay_epoch
        self.ckpt_interval = ckpt_interval

        # Metrics
        self._train_losses = []
        self._val_losses = []
        self._train_accuracies = []
        self._val_accuracies = []
        self._best_accuracy = 0.0
        self._best_loss = float("inf")

    def __str__(self) -> str:
        return f"{self.model_name} (device: {self.device})"

    def init_model(self, num_classes: int) -> torch.nn.Module:
        if self.model is not None:
            self.__init_model(num_classes)
            if self.weight_path:
                print(f">> Load model from {self.weight_path}")
                self.model.load_state_dict(torch.load(self.weight_path))
            return self.model.to(self.device)
        elif self.weight_path:
            print(f">> Load model from {self.weight_path}")
            self.model = torch.hub.load(
                "pytorch/vision:v0.6.0",
                self.model_name,
            )
            self.__init_model(num_classes)
            self.model.load_state_dict(torch.load(self.weight_path))
        else:
            print(f">> Load {self.model_name} model")
            self.model = torch.hub.load(
                "pytorch/vision:v0.6.0",
                self.model_name,
                pretrained=self.__pretrained
            )
            self.__init_model(num_classes)
        return self.model.to(self.device)

    def __init_model(self, num_classes: int):
        # Modify the last layer to match the number of classes
        if self.model_name in ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]:
            num_features = self.model.fc.in_features
            self.model.fc = torch.nn.Linear(num_features, num_classes)
        elif self.model_name in ["alexnet", "vgg11", "vgg13", "vgg16", "vgg19"]:
            num_features = self.model.classifier[6].in_features
            self.model.classifier[6] = torch.nn.Linear(num_features, num_classes)
        elif self.model_name in ["squeezenet1_0", "squeezenet1_1"]:
            self.model.classifier[1] = torch.nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
            self.model.num_classes = num_classes
        elif self.model_name in ["densenet121", "densenet169", "densenet161", "densenet201"]:
            num_features = self.model.classifier.in_features
            self.model.classifier = torch.nn.Linear(num_features, num_classes)
        elif self.model_name in ["inception_v3"]:
            num_features = self.model.fc.in_features
            self.model.fc = torch.nn.Linear(num_features, num_classes)
        else:
            raise ValueError(f"Not support this model yet: {self.model_name}")

    def train(self, train_data: TorchImageClassificationDataset, val_data: TorchImageClassificationDataset, **kwargs):
        # When Ctrl+C is pressed, save the model before exit
        signal.signal(signal.SIGINT, self.__train_interrupt_handler)
        print(f">> Training {self.model_name} model")
        print("=" * 50)
        train_data_loader = DataLoader(
            train_data,
            batch_size=self.batch_size,
            shuffle=True
        )
        val_data_loader = DataLoader(
            val_data,
            batch_size=self.batch_size,
            shuffle=False
        )
        print(f"> Load {len(train_data)} training images")
        print(f"> Load {len(val_data)} validation images")
        print(f"> Number of classes: {len(train_data.categories)}")
        print(f"> {'Classes index:'.ljust(len('Number of classes:'))} {train_data.categories}")
        print(f"> {'Input size:'.ljust(len('Number of classes:'))} {self.input_size}")
        print(f"> {'Batch size:'.ljust(len('Number of classes:'))} {self.batch_size}")
        print(f"> {'Epochs:'.ljust(len('Number of classes:'))} {self.epochs}")
        print(f"> {'Learning rate:'.ljust(len('Number of classes:'))} {self.learning_rate}")
        print(f"> {'Loss function:'.ljust(len('Number of classes:'))} {self.loss_fn}")
        print(f"> {'Optimizer:'.ljust(len('Number of classes:'))} {self.optimizer}")
        print("-" * 50)
        self.model = self.init_model(num_classes=len(train_data.categories))
        print("> Model Summary:")
        torchsummary.summary(self.model, input_size=train_data_loader.dataset[0][0].shape)

        self.optimizer = self.optimizer(self.model.parameters(), lr=self.learning_rate)
        print("Start training...\n" + "-" * 50)

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0  # Reset training loss
            train_y_true, train_y_pred = [], []
            for i, (images, labels) in enumerate(train_data_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                train_y_true.extend(labels.cpu().numpy())
                train_y_pred.extend(predicted.cpu().numpy())
                if (i + 1) % 5 == 0:
                    print(
                        f"> Epoch [{epoch + 1}/{self.epochs}], "
                        f"Step [{i + 1}/{len(train_data_loader)}], "
                        f"Loss = {loss.item():.4f}"
                    )

            train_accuracy = accuracy_score(train_y_true, train_y_pred) * 100
            train_precision = precision_score(train_y_true, train_y_pred, average='macro')
            train_recall = recall_score(train_y_true, train_y_pred, average='macro')
            train_f1 = f1_score(train_y_true, train_y_pred, average='macro')
            print(
                f"==>> Finished {epoch + 1}/{self.epochs} epoch(s), Loss = {train_loss / len(train_data_loader):.4f}\n"
                + "-" * 50
            )
            print(f">> Training Metrics:")
            print(f"{'Accuracy'.ljust(len('Precision'))} = {train_accuracy:.2f}%")
            print(f"Precision = {train_precision:.2f}")
            print(f"{f'Recall'.ljust(len('Precision'))} = {train_recall:.2f}")
            print(f"{f'F1 Score'.ljust(len('Precision'))} = {train_f1:.2f}")
            print("-" * 50)

            self._train_accuracies.append(train_accuracy)

            if (epoch + 1) % self.decay_epoch == 0:
                self.learning_rate *= 1 / (1 + self.decay * epoch)
                self.__update_lr(self.optimizer, self.learning_rate)
                print(f"=> Learning rate decayed to {self.learning_rate}")

            self._train_losses.append(train_loss / len(train_data_loader))

            self.model.eval()
            with torch.no_grad():
                val_loss = 0.0
                val_y_true, val_y_pred = [], []
                for images, labels in val_data_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = self.loss_fn(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    val_y_true.extend(labels.cpu().numpy())
                    val_y_pred.extend(predicted.cpu().numpy())

                val_accuracy = accuracy_score(val_y_true, val_y_pred) * 100
                val_precision = precision_score(val_y_true, val_y_pred, average='macro')
                val_recall = recall_score(val_y_true, val_y_pred, average='macro')
                val_f1 = f1_score(val_y_true, val_y_pred, average='macro')
                print(f"==>> Validation Loss = {val_loss / len(val_data_loader):.4f}")
                print(f">> Validation Metrics:")
                print(f"{'Accuracy'.ljust(len('Precision'))} = {val_accuracy:.2f}%")
                print(f"Precision = {val_precision:.2f}")
                print(f"{f'Recall'.ljust(len('Precision'))} = {val_recall:.2f}")
                print(f"{f'F1 Score'.ljust(len('Precision'))} = {val_f1:.2f}")
                print("-" * 50)
                self._val_accuracies.append(val_accuracy)
                self._val_losses.append(val_loss / len(val_data_loader))

            if val_accuracy > self._best_accuracy:
                self._best_accuracy = val_accuracy
                self.__save_model("best")

            if val_loss < self._best_loss:
                self._best_loss = val_loss

            if self.__early_stopping and self.__check_early_stopping():
                print(f"Early stopping at epoch {epoch + 1}...")
                print(f"Training finished! Best accuracy: {self._best_accuracy:.2f}%")
                self.__save_model(f"final_e{epoch + 1}")
                self.__plot_metrics()
                return

            if (epoch + 1) % self.ckpt_interval == 0:
                self.__save_model(f"e{epoch + 1}")

        print(f"Training finished! Best accuracy: {self._best_accuracy:.2f}%")
        self.__save_model(f"final")
        self.__plot_metrics()

    def __check_early_stopping(self) -> bool:
        if len(self._val_losses) < self.patience:
            return False
        last_losses = self._val_losses[-self.patience:]
        if all([last_losses[i] - last_losses[i + 1] < self.delta for i in range(self.patience - 1)]):
            return True
        return False

    def __plot_metrics(self):
        # For now plot training loss, validation loss, training accuracy, validation accuracy curves
        # Maybe add more later, e.g. confusion matrix, feature maps, grad-cam
        figure = plt.figure(figsize=(10, 10), dpi=300)
        gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])

        # Training loss
        ax0 = figure.add_subplot(gs[0])
        ax0.plot(range(1, len(self._train_losses) + 1), self._train_losses)
        ax0.set_xlabel("Epoch")
        ax0.set_ylabel("Loss")
        ax0.set_title("Training Loss")

        # Validation loss
        ax1 = figure.add_subplot(gs[1])
        ax1.plot(range(1, len(self._val_losses) + 1), self._val_losses)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Validation Loss")

        # Training accuracy
        ax2 = figure.add_subplot(gs[2])
        ax2.plot(range(1, len(self._train_accuracies) + 1), self._train_accuracies)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.set_title("Training Accuracy")

        # Validation accuracy
        ax3 = figure.add_subplot(gs[3])
        ax3.plot(range(1, len(self._val_accuracies) + 1), self._val_accuracies)
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Accuracy (%)")
        ax3.set_title("Validation Accuracy")

        plt.tight_layout()
        plt.show()

    @staticmethod
    def __update_lr(optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def predict(self, data: torch.Tensor, **kwargs) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            data = data.to(self.device)
            outputs = self.model(data)
            _, predicted = torch.max(outputs, 1)
            return predicted.cpu().numpy()

    def test(self, test_data: TorchImageClassificationDataset, batch_size: int = 1, **kwargs) -> pd.DataFrame:
        print(f">> Testing {self.model_name} model")
        print("=" * 50)
        test_data_loader = DataLoader(
            test_data,
            batch_size=self.batch_size,
            shuffle=False
        )
        print(f"> Load {len(test_data)} testing images")
        print(f"> Number of classes: {len(test_data.categories)}")
        print(f"> {'Classes index:'.ljust(len('Number of classes:'))} {test_data.categories}")
        print(f"> {'Input size:'.ljust(len('Number of classes:'))} {self.input_size}")
        print(f"> {'Batch size:'.ljust(len('Number of classes:'))} {batch_size}")
        print("-" * 50)
        self.init_model(num_classes=len(test_data.categories))
        with torch.no_grad():
            self.model.eval()
            test_loss = 0.0
            test_y_true, test_y_pred = [], []
            for images, labels in test_data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                test_y_true.extend(labels.cpu().numpy())
                test_y_pred.extend(predicted.cpu().numpy())

            test_accuracy = accuracy_score(test_y_true, test_y_pred) * 100
            test_confusion_matrix = confusion_matrix(test_y_true, test_y_pred)
            test_precision = precision_score(test_y_true, test_y_pred, average='macro')
            test_recall = recall_score(test_y_true, test_y_pred, average='macro')
            test_f1 = f1_score(test_y_true, test_y_pred, average='macro')
            print(f">> Test Loss = {test_loss / len(test_data_loader):.4f}")
            print(f">> Test Metrics:")
            print(f"{'Accuracy'.ljust(len('Precision'))} = {test_accuracy:.2f}%")
            print(f"Precision = {test_precision:.2f}")
            print(f"{f'Recall'.ljust(len('Precision'))} = {test_recall:.2f}")
            print(f"{f'F1 Score'.ljust(len('Precision'))} = {test_f1:.2f}")
            print("-" * 50)
            return pd.DataFrame(
                columns=['Test Loss', 'confusion_matrix', 'Accuracy', 'Precision', 'Recall', 'F1 Score'],
                data=[[
                    test_loss / len(test_data_loader),
                    test_confusion_matrix, test_accuracy, test_precision, test_recall, test_f1
                ]]
            )

    def __save_model(self, prefix: str = ""):
        (self.save_dir / f"{prefix}_{self.model_name}.pth").unlink(missing_ok=True)  # Remove old model
        torch.save(self.model.state_dict(), str(self.save_dir / f"{prefix}_{self.model_name}.pth"))
        ex = torch.randn(1, 3, *self.input_size).to(self.device)
        torch.onnx.export(
            self.model,
            ex,
            str(self.save_dir / f"{prefix}_{self.model_name}.onnx")
        ) if self.__save_onnx else None

        torch.jit.save(
            torch.jit.trace(self.model, torch.randn(1, 3, *self.input_size).to(self.device)),
            str(self.save_dir / f"{prefix}_{self.model_name}.torchscript")
        ) if self.__save_jit else None
        print(f"* Model saved to {self.save_dir}\n" + "-" * 50)

    def __train_interrupt_handler(self, _, __):
        warnings.warn("Model training interrupted! Saving model...")
        self.__save_model("interrupted")
        print(f"Best accuracy: {self._best_accuracy:.2f}%")
        sys.exit(0)
