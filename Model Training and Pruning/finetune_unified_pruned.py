# === finetune_unified_pruned.py ===
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.models import mobilenet_v2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# Load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Load pruned model (weights only)
model = mobilenet_v2(pretrained=False)
model.classifier[1] = nn.Linear(model.last_channel, 10)
model.load_state_dict(torch.load("../mobilenetv2_taskaware_pruned_finetuned.pth", map_location=device))
model.to(device)

# Optimizer and loss
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# === Fine-tune for 10 epochs ===
print("\n🔁 Fine-tuning unified model for 10 epochs...\n")
for epoch in range(10):
    model.train()
    running_loss = 0.0
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"[Epoch {epoch+1}] Loss: {running_loss/len(trainloader):.4f}")

# Save fine-tuned unified model
torch.save(model.state_dict(), "../mobilenetv2_unified_finetuned.pth")
print("\nUnified fine-tuned model saved.")
