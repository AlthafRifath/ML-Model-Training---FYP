import torch
import torchvision
import torchvision.transforms as transforms
import onnxruntime as ort
import numpy as np
from tqdm import tqdm

# === Config ===
BATCH_SIZE = 32
IMAGE_SIZE = 224
MODEL_PATH = "../../mobilenetv2_unified_finetuned.pth.onnx"

# === Preprocessing ===
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# === CIFAR-10 Data ===
trainset = torchvision.datasets.CIFAR10(root='../../data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='../../data', train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

# === Load ONNX Model ===
session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name

def predict_onnx(batch_tensor):
    batch_np = batch_tensor.numpy()
    ort_inputs = {input_name: batch_np}
    ort_outs = session.run(None, ort_inputs)
    return np.argmax(ort_outs[0], axis=1)

def evaluate_accuracy(dataloader, split_name):
    correct, total = 0, 0
    for images, labels in tqdm(dataloader, desc=f"Evaluating {split_name} Accuracy"):
        preds = predict_onnx(images)
        correct += (preds == labels.numpy()).sum()
        total += labels.size(0)
    return 100.0 * correct / total

# === Evaluate ===
train_acc = evaluate_accuracy(trainloader, "Train")
test_acc = evaluate_accuracy(testloader, "Test")

print(f"\nUnified Fine-Tuned ONNX Model - Train Accuracy: {train_acc:.2f}%")
print(f"Unified Fine-Tuned ONNX Model - Test Accuracy: {test_acc:.2f}%")
