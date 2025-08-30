# import torch
# import torch.nn as nn
# from torchvision.models import mobilenet_v2
#
# # Load model
# model = mobilenet_v2(pretrained=False)
# model.classifier[1] = nn.Linear(model.last_channel, 10)
# model.load_state_dict(torch.load("mobilenetv2_cifar10_pruned_finetuned.pth"))
# model.eval()
#
# # Dummy input
# dummy_input = torch.randn(1, 3, 224, 224)
#
# # Export to ONNX
# torch.onnx.export(
#     model,
#     dummy_input,
#     "mobilenetv2_cifar10_pruned_finetuned.onnx",
#     input_names=["input"],
#     output_names=["output"],
#     dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
#     opset_version=11
# )
#
# print("Model exported to mobilenetv2_cifar10_pruned_finetuned.onnx")

import torch
import torchvision
from torchvision.models import mobilenet_v2

# === 1. Load trained baseline model ===
model = mobilenet_v2(num_classes=10)
model.load_state_dict(torch.load("../mobilenetv2_cifar10_baseline.pth"))
model.eval()

# === 2. Dummy input for ONNX export ===
dummy_input = torch.randn(1, 3, 224, 224)

# === 3. Export to ONNX ===
torch.onnx.export(
    model,
    dummy_input,
    "../mobilenetv2_hardware_onnx.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=11
)

print("✅ Exported to mobilenetv2_hardware_onnx.onnx")