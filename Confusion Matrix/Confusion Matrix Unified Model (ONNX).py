import onnxruntime as ort
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tqdm import tqdm

# CIFAR-10 test set preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

# Load ONNX model
ort_session = ort.InferenceSession("../archive/mobilenetv2_cifar10_pruned_finetuned.onnx", providers=['CPUExecutionProvider'])

# Prediction function for a batch
def predict_onnx(images):
    images = images.numpy()
    ort_inputs = {ort_session.get_inputs()[0].name: images}
    ort_outs = ort_session.run(None, ort_inputs)
    preds = np.argmax(ort_outs[0], axis=1)
    return preds

# Inference loop
y_true = []
y_pred = []

for images, labels in tqdm(testloader):
    preds = predict_onnx(images)
    y_true.extend(labels.numpy())
    y_pred.extend(preds)

# Plot confusion matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=testset.classes)
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Confusion Matrix – Unified Pruned ONNX Model")
plt.tight_layout()
plt.show()
