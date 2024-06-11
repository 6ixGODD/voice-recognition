from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
# import torchsummary
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights

from datasets.torch import TorchImageClassificationDataset

# from transformers import ViTForImageClassification

DATA_DIR = Path('output/dataset-aug-0.2/images/')
SAVE_DIR = Path('output/vit')
SAVE_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32

LEARNING_RATE = 0.001

NUM_CLASSES = 13

EPOCHS = 1000

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
)
train_dataset = TorchImageClassificationDataset(transform=transform)
train_dataset.load_images(root=str(DATA_DIR / 'train'))
val_dataset = TorchImageClassificationDataset(transform=transform)
val_dataset.load_images(root=str(DATA_DIR / 'val'))
test_dataset = TorchImageClassificationDataset(transform=transform)
test_dataset.load_images(root=str(DATA_DIR / 'test'))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model = ViTForImageClassification.from_pretrained(
#     'google/vit-base-patch16-224-in21k',
#     num_labels=NUM_CLASSES, ignore_mismatched_sizes=True
# ).to(device)
model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1).to(device)

# Modify the model to fit the number of classes
in_features = model.hidden_dim
model.heads = nn.Linear(in_features, NUM_CLASSES).to(device)
# torchsummary.summary(model, (3, 224, 224))

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

def update_lr(opt, lr):
    for param_group in opt.param_groups:
        param_group['lr'] = lr


best_val_acc = 0
curr_lr = LEARNING_RATE
train_losses, val_losses = [], []
train_acc, val_acc = [], []
# training loop
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # decay learning rate
    if (epoch + 1) % 20 == 0:
        curr_lr /= 3
        update_lr(optimizer, curr_lr)

    print(f'Epoch {epoch + 1}/{EPOCHS}, Loss: {train_loss / len(train_loader)}, Accuracy: {100 * correct / total}')
    train_losses.append(train_loss / len(train_loader))
    train_acc.append(100 * correct / total)

    # validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Epoch {epoch + 1}/{EPOCHS}, Validation Loss: {val_loss / len(val_loader)}, Validation Accuracy: {100 * correct / total}')
    val_losses.append(val_loss / len(val_loader))
    val_acc.append(100 * correct / total)

    if 100 * correct / total > best_val_acc:
        best_val_acc = 100 * correct / total
        torch.save(model.state_dict(), str(SAVE_DIR / 'best_model.pth'))

# Save the model
torch.save(model.state_dict(), str(SAVE_DIR / 'final_model.pth'))
