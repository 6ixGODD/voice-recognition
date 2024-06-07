import cv2

from models.cnn import ConvolutionNeuralNetworkClassifierBackend
from torchvision import transforms
from PIL import Image


backend = ConvolutionNeuralNetworkClassifierBackend(
    weight_path="../best_resnet18_tongue.pth",
    model_name="resnet18",
)

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
)
backend.init_model(num_classes=13)

im = Image.open("0.png")
im = transform(im)
im = im[None]
print(backend.predict(im))
