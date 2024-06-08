from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import ViTForImageClassification, ViTFeatureExtractor

from datasets.torch import TorchImageClassificationDataset

DATA_DIR = Path('D:\WorkSpace\ESIGELEC.Course\ArtificialIntelligenceForSmartSys\local-binary-patterns\data\data_split')

BATCH_SIZE = 32

LEARNING_RATE = 0.001

NUM_CLASSES = 13

EPOCHS = 100

feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
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

model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224', num_labels=NUM_CLASSES, ignore_mismatched_sizes=True
).to(device)
# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# training loop
for epoch in range(EPOCHS):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{EPOCHS}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item()}')

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print(f'Epoch [{epoch + 1}/{EPOCHS}], Val Accuracy: {accuracy}')

# Save the model
torch.save(model.state_dict(), 'vit_model.pth')
