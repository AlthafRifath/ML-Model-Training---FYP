import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision.models import mobilenet_v2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# === Load CIFAR-10 ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(root='../../data', train=True, download=True, transform=transform)
testset  = torchvision.datasets.CIFAR10(root='../../data', train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=False)
testloader  = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# === Load trained model ===
model = mobilenet_v2(num_classes=10)
model.load_state_dict(torch.load("../../mobilenetv2_taskaware_pruned_finetuned.pth", map_location=device))
model.to(device)
model.eval()

def evaluate(loader, name):
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    print(f"{name} Accuracy: {acc:.2f}%")

evaluate(trainloader, "Train")
evaluate(testloader,  "Test")
