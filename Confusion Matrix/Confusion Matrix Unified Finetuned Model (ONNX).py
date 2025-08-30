import onnxruntime as ort
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tqdm import tqdm

# === 1. Load CIFAR-10 Test Set ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)
class_names = testset.classes

# === 2. Load Unified Fine-Tuned ONNX Model ===
ort_session = ort.InferenceSession("../mobilenetv2_unified_finetuned.pth.onnx", providers=['CPUExecutionProvider'])

# === 3. Prediction Function ===
def predict_onnx(batch):
    batch_np = batch.numpy()
    ort_inputs = {ort_session.get_inputs()[0].name: batch_np}
    ort_outs = ort_session.run(None, ort_inputs)
    preds = np.argmax(ort_outs[0], axis=1)
    return preds

# === 4. Evaluate & Build Confusion Matrix ===
y_true, y_pred = [], []

for images, labels in tqdm(testloader, desc="Evaluating"):
    preds = predict_onnx(images)
    y_pred.extend(preds)
    y_true.extend(labels.numpy())

# === 5. Plot Confusion Matrix ===
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Confusion Matrix – Unified Fine-Tuned ONNX Model")
plt.tight_layout()
plt.show()
