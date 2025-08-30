import onnxruntime as ort
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm

# === CIFAR-10 Loaders ===
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(root='../../data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='../../data', train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

# === Load ONNX Model ===
session = ort.InferenceSession("../../mobilenetv2_unified.onnx", providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name

def get_accuracy(dataloader):
    correct = 0
    total = 0
    for images, labels in tqdm(dataloader):
        images_np = images.numpy()
        outputs = session.run(None, {input_name: images_np})
        preds = np.argmax(outputs[0], axis=1)
        correct += (preds == labels.numpy()).sum()
        total += labels.size(0)
    return correct / total

# === Evaluate ===
train_acc = get_accuracy(trainloader)
test_acc = get_accuracy(testloader)

print(f"\nUnified ONNX Model - Train Accuracy: {train_acc * 100:.2f}%")
print(f"Unified ONNX Model - Test Accuracy: {test_acc * 100:.2f}%")
