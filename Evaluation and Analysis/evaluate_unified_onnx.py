import onnxruntime as ort
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch
import time
from tqdm import tqdm

# === Load CIFAR-10 test data ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

# === Load ONNX unified model ===
ort_session = ort.InferenceSession("../mobilenetv2_unified.onnx", providers=['CPUExecutionProvider'])

# === Evaluate ONNX model ===
correct = 0
total = 0
inference_times = []

for inputs, labels in tqdm(testloader):
    inputs = inputs.numpy()
    start = time.time()
    outputs = ort_session.run(None, {"input": inputs})
    end = time.time()
    inference_times.append(end - start)

    pred = np.argmax(outputs[0])
    correct += (pred == labels.item())
    total += 1

accuracy = 100 * correct / total
avg_time = np.mean(inference_times) * 1000

print(f"\nUnified Model Accuracy: {accuracy:.2f}%")
print(f"Inference Time (CPU): {avg_time:.2f} ms")
