import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
import os

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")

# CIFAR-10 preprocessing
transform = transforms.Compose([
    transforms.Resize(224),  # MobileNet expects 224x224
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=4)

# Load pretrained MobileNetV2
model = mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(model.last_channel, 10)  # Adjust for CIFAR-10
model = model.to(device)

# Print model summary
summary(model, input_size=(3, 224, 224))

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train
def train_model(epochs=3):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"[Epoch {epoch+1}] Loss: {running_loss / len(trainloader):.4f}")

# Evaluate
def evaluate_model():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy on test set: {100 * correct / total:.2f}%")

# Main execution
if __name__ == "__main__":
    train_model(epochs=3)
    evaluate_model()
    torch.save(model.state_dict(), "mobilenetv2_cifar10.pth")
    print("Model saved as mobilenetv2_cifar10.pth")
