import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import mobilenet_v2
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load CIFAR-10 again
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=0)

# Load trained model
model = mobilenet_v2(pretrained=False)
model.classifier[1] = nn.Linear(model.last_channel, 10)
model.load_state_dict(torch.load("mobilenetv2_cifar10.pth"))
model = model.to(device)
model.eval()

# Apply L1-norm structured pruning to all conv2d layers
def apply_l1_pruning(model, amount=0.3):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.ln_structured(module, name="weight", amount=amount, n=1, dim=0)
            prune.remove(module, 'weight')  # remove pruning reparameterization
    return model

# Evaluate function
def evaluate(model):
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    print(f"Accuracy after pruning: {acc:.2f}%")

if __name__ == "__main__":
    # Prune and evaluate
    pruned_model = apply_l1_pruning(model, amount=0.3)
    evaluate(pruned_model)

    # Save pruned model
    torch.save(pruned_model.state_dict(), "mobilenetv2_cifar10_pruned.pth")
    print("Pruned model saved as mobilenetv2_cifar10_pruned.pth")

