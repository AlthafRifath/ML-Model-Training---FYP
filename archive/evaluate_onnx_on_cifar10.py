import onnxruntime as ort
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from tqdm import tqdm

# Load ONNX model with CPU provider
ort_session = ort.InferenceSession("mobilenetv2_cifar10_pruned_finetuned.onnx", providers=['CPUExecutionProvider'])

# Preprocessing (same as during training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# CIFAR-10 test dataset
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

# Inference loop
correct = 0
total = 0

for images, labels in tqdm(testloader):
    # ONNX requires numpy input
    images_np = images.numpy()

    for i in range(images_np.shape[0]):
        input_tensor = np.expand_dims(images_np[i], axis=0).astype(np.float32)
        outputs = ort_session.run(None, {"input": input_tensor})
        predicted = np.argmax(outputs[0])

        if predicted == labels[i].item():
            correct += 1
        total += 1

# Report accuracy
accuracy = 100 * correct / total
print(f"Accuracy on CIFAR-10 test set (pruned ONNX model): {accuracy:.2f}%")
