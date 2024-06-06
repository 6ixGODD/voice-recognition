# > Classes index:
# {'0': 0, '1': 1, '10': 2, '11': 3, '12': 4, '2': 5, '3': 6, '4': 7, '5': 8, '6': 9, '7': 10, '8': 11, '9': 12}
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import datetime
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchsummary
# import wandb  # TODO: Add wandb logging support
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

from datasets.torch import TorchImageClassificationDataset

# config
# resnet18, resnet34, resnet50, resnet101, resnet152, alexnet, vgg11, vgg13, vgg16, vgg19,
# squeezenet1_0, squeezenet1_1, densenet121, densenet169, densenet161, densenet201, inception_v3
MODEL = 'vgg11'
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
NUM_CLASSES = 13
DATA_DIR = '../data/data_split'
SAVE_DIR = './output'

if not Path(SAVE_DIR).exists():
    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)


# wandb.init(
#     project="tongue-image-recognition",
#     name=f"{MODEL}",
#     config={
#         "epochs": EPOCHS,
#         "batch_size": BATCH_SIZE,
#         "learning_rate": LEARNING_RATE,
#         "model": MODEL,
#         "num_classes": NUM_CLASSES
#     }
# )


def get_logger(name, save_dir, enable_ch=True):
    if not Path(save_dir).exists():
        Path(save_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s")
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    ch.flush()
    fh = logging.FileHandler(Path(save_dir, f"{name}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.log"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    if enable_ch:
        logger.addHandler(ch)
    logger.addHandler(fh)
    logger.info(
        "# log file: {}".format(
            Path(save_dir, f"{name}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.log")
        )
    )
    return logger


def increment_path(path, sep="-"):
    """ Automatically increment path, i.e. weights/exp -> weights/exp{sep}2, weights/exp{sep}3, ..."""
    path = Path(path)
    if path.exists():
        path, suffix = (
            (path.with_suffix(""), path.suffix) if path.is_file() else (path, "")
        )
        for n in range(2, 9999):
            p = f"{path}{sep}{n}{suffix}"
            if not Path(p).exists():
                path = Path(p)
                break
        path.mkdir(parents=True, exist_ok=True)  # make directory
    else:
        path.mkdir(parents=True, exist_ok=True)
    return path


SAVE_DIR = increment_path(Path(SAVE_DIR) / 'exp')

LOGGER = get_logger(f"{MODEL}", SAVE_DIR / 'log')
LOGGER.info(f">> Start training {MODEL} model")
LOGGER.info("=" * 66)
LOGGER.info(f"> Batch size: {BATCH_SIZE}")
LOGGER.info(f"> Epochs: {EPOCHS}")
LOGGER.info(f"> Learning rate: {LEARNING_RATE}")
LOGGER.info(f"> Num classes: {NUM_CLASSES}")
LOGGER.info("=" * 66)

# device setting ----------------------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
LOGGER.info('> Using {} device'.format(device))
LOGGER.info("=" * 66)

# preprocessing ----------------------------------------------
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
)

# load dataset
# train_dataset = datasets.ImageFolder(root=f'{DATA_DIR}/train', transform=transform)
# val_dataset = datasets.ImageFolder(root=f'{DATA_DIR}/val', transform=transform)
# test_dataset = datasets.ImageFolder(root=f'{DATA_DIR}/test', transform=transform)

train_dataset = TorchImageClassificationDataset(transform=transform)
train_dataset.load_images(root=f'{DATA_DIR}/train')
val_dataset = TorchImageClassificationDataset(transform=transform)
val_dataset.load_images(root=f'{DATA_DIR}/val')
test_dataset = TorchImageClassificationDataset(transform=transform)
test_dataset.load_images(root=f'{DATA_DIR}/test')

train_loader = DataLoader(
    train_dataset,
    # batch_size=wandb.config.batch_size if wandb.config.batch_size else BATCH_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)
val_loader = DataLoader(
    val_dataset,
    # batch_size=wandb.config.batch_size if wandb.config.batch_size else BATCH_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)
test_loader = DataLoader(
    test_dataset,
    # batch_size=wandb.config.batch_size if wandb.config.batch_size else BATCH_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

LOGGER.info(f'> Train dataset has {len(train_dataset)} samples, {len(train_loader)} batches')
LOGGER.info(f'> Validation dataset has {len(val_dataset)} samples, {len(val_loader)} batches')
# LOGGER.info(f'> Test dataset has {len(test_dataset)} samples, {len(test_loader)} batches')
LOGGER.info(f"> Classes index: {train_dataset.categories}")
LOGGER.info("=" * 66)

model = torch.hub.load(
    'pytorch/vision:v0.6.0',
    # wandb.config.model if wandb.config.model else MODEL,
    MODEL,
    pretrained=True
)

# modify fc layer ----------------------------------------------
if MODEL in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, NUM_CLASSES)
elif MODEL in ['alexnet', 'vgg11', 'vgg13', 'vgg16', 'vgg19']:
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, NUM_CLASSES)
elif MODEL in ['squeezenet1_0', 'squeezenet1_1']:
    model.classifier[1] = nn.Conv2d(512, NUM_CLASSES, kernel_size=(1, 1), stride=(1, 1))
    model.num_classes = NUM_CLASSES
elif MODEL in ['densenet121', 'densenet169', 'densenet161', 'densenet201']:
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, NUM_CLASSES)
elif MODEL in ['inception_v3']:
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, NUM_CLASSES)
else:
    raise ValueError('>> Not support this model yet!')

# to gpu/cpu
model.to(device)
torchsummary.summary(model, (3, 224, 224))

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    # lr=wandb.config.learning_rate if wandb.config.learning_rate else LEARNING_RATE
    lr=LEARNING_RATE
)


# for updating learning rate
def update_lr(opt, lr):
    for param_group in opt.param_groups:
        param_group['lr'] = lr


# train model ----------------------------------------------
LOGGER.info("Start training...")
LOGGER.info("=" * 66)
total_step = len(train_loader)
total_step_val = len(val_loader)
# curr_lr = wandb.config.learning_rate if wandb.config.learning_rate else LEARNING_RATE
curr_lr = LEARNING_RATE
best_acc = 0.0
train_losses = []
val_acc = []
val_losses = []
# for epoch in range(wandb.config.epochs if wandb.config.epochs else EPOCHS):
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if (i + 1) % 5 == 0:
            LOGGER.info(f"> Epoch [{epoch + 1}/{EPOCHS}], Step [{i + 1}/{total_step}], Loss: {loss.item():.4f}")

    # decay learning rate
    if (epoch + 1) % 20 == 0:
        curr_lr /= 3
        update_lr(optimizer, curr_lr)
        LOGGER.info(f'=> Learning rate decay: lr={curr_lr}')

    LOGGER.info(f'==>> Finish {epoch + 1} epoch, Loss: {train_loss / total_step:.6f}')
    train_losses.append(train_loss / total_step)

    # validation
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        val_loss = 0.0
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            # forward
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            val_loss += criterion(outputs, labels).item()

        LOGGER.info(f'==>> Validation Accuracy: {correct / total * 100:.2f} %')
        if correct / total > best_acc:
            best_acc = correct / total
            torch.save(model.state_dict(), f'{SAVE_DIR}/best_{MODEL}.pth')
            ex = torch.randn(1, 3, 224, 224).to(device)
            torch.jit.save(torch.jit.trace(model, ex), f'{SAVE_DIR}/best_{MODEL}.torchscript')
        val_acc.append(correct / total * 100)
        val_losses.append(val_loss / total_step_val)

        # wandb.log({"running_loss": running_loss / total_step, "val_acc": correct / total * 100})

# test model ----------------------------------------------
LOGGER.info("=" * 66)
LOGGER.info("Start testing...")
LOGGER.info("=" * 66)
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        # forward
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    LOGGER.info('>> Test Accuracy: {:.2f} %'.format(correct / total * 100))

# save model ----------------------------------------------
torch.save(model.state_dict(), f'{SAVE_DIR}/final_{MODEL}.pth')
ex = torch.randn(1, 3, 224, 224).to(device)
torch.jit.save(torch.jit.trace(model, ex), f'{SAVE_DIR}/final_{MODEL}.torchscript')
LOGGER.info(f">> Weights saved to {SAVE_DIR}/final_{MODEL}.pth")
LOGGER.info(f">> Torchscript saved to {SAVE_DIR}/final_{MODEL}.torchscript")
LOGGER.info('>> Best accuracy: {:.2f} %'.format(best_acc * 100))

# plot ----------------------------------------------
if not Path(SAVE_DIR, 'plot').exists():
    Path(SAVE_DIR, 'plot').mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(10, 5))
plt.title("Loss")
plt.plot(train_losses, label="train loss")
plt.plot(val_losses, label="validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig(Path(SAVE_DIR, 'plot', f'loss_{MODEL}.png'))
plt.show()

plt.figure(figsize=(10, 5))
plt.title("Accuracy")
plt.plot(val_acc, label="validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig(Path(SAVE_DIR, 'plot', f'acc_{MODEL}.png'))
plt.show()
