import onnxruntime as ort
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch
from tqdm import tqdm
import time

# === 1. Load CIFAR-10 Test Set ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Match MobileNetV2 input
    transforms.ToTensor()
])

testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

# === 2. Load ONNX Model ===
ort_session = ort.InferenceSession("../mobilenetv2_unified_finetuned.pth.onnx", providers=['CPUExecutionProvider'])

# === 3. Inference Loop ===
y_true = []
y_pred = []
start_time = time.time()

for images, labels in tqdm(testloader):
    ort_inputs = {ort_session.get_inputs()[0].name: images.numpy()}
    ort_outs = ort_session.run(None, ort_inputs)
    preds = np.argmax(ort_outs[0], axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(preds)

inference_time = (time.time() - start_time) * 1000 / len(testset)  # ms/sample
accuracy = np.mean(np.array(y_true) == np.array(y_pred)) * 100

# === 4. Output ===
print(f"\nUnified Fine-Tuned ONNX Model Accuracy: {accuracy:.2f}%")
print(f"Inference Time (CPU): {inference_time:.2f} ms/sample")
