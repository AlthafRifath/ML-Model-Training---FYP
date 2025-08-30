import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Recreate model structure
model = mobilenet_v2(pretrained=False)
model.classifier[1] = nn.Linear(model.last_channel, 10)

# Load fine-tuned pruned weights
model.load_state_dict(torch.load("mobilenetv2_cifar10_pruned_finetuned.pth"))
model.eval()

# Create dummy input for tracing
dummy_input = torch.randn(1, 3, 224, 224)

# Convert to TorchScript
traced_model = torch.jit.trace(model, dummy_input)

# Save TorchScript model
traced_model.save("mobilenetv2_cifar10_pruned_finetuned.pt")
print("TorchScript model saved as mobilenetv2_cifar10_pruned_finetuned.pt")
