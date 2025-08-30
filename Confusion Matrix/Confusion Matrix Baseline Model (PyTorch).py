from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.models import mobilenet_v2
import torch.nn as nn
import numpy as np

# Load CIFAR-10 test set
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
])
testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

# Recreate model architecture
model = mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, 10)  # CIFAR-10 has 10 classes

# Load weights
model.load_state_dict(torch.load("../mobilenetv2_cifar10_baseline.pth"))
model.eval()

# Inference
y_true, y_pred = [], []
with torch.no_grad():
    for images, labels in testloader:
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.numpy())
        y_pred.extend(preds.numpy())

# Plot
disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=testset.classes, cmap=plt.cm.Blues)
plt.title("Confusion Matrix – Baseline Model")
plt.tight_layout()
plt.show()
