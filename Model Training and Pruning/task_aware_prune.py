# === 1. Imports ===
import torch
import torch.nn.utils.prune as prune
import torchvision
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2
import torch.nn as nn

# === 2. Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === 3. Load CIFAR-10 test set ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# === 4. Load baseline model ===
model = mobilenet_v2(num_classes=10)
model.load_state_dict(torch.load("../mobilenetv2_cifar10_baseline.pth"))
model.to(device)
model.eval()

# === 5. Apply L1 structured pruning to Conv2d layers only ===
print("Applying L1-norm structured pruning (30%)...")
for name, module in model.features.named_modules():
    if isinstance(module, nn.Conv2d):
        prune.ln_structured(module, name='weight', amount=0.3, n=1, dim=0)

# === 6. Remove pruning reparam to make it permanent ===
for name, module in model.features.named_modules():
    if isinstance(module, nn.Conv2d) and hasattr(module, 'weight_orig'):
        prune.remove(module, 'weight')

# === 7. Evaluate pruned model ===
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Accuracy after task-aware pruning (no fine-tuning): {accuracy:.2f}%")

# === 8. Save model ===
torch.save(model.state_dict(), "../mobilenetv2_taskaware_pruned.pth")
print("Pruned model saved as mobilenetv2_taskaware_pruned.pth")
