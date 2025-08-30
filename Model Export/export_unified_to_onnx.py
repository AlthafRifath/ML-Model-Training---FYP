import torch
from torchvision.models import mobilenet_v2

# === Load the fine-tuned, task-aware pruned model ===
model = mobilenet_v2(num_classes=10)
model.load_state_dict(torch.load("../mobilenetv2_unified_finetuned.pth"))
model.eval()

# === Dummy input for ONNX export ===
dummy_input = torch.randn(1, 3, 224, 224)

# === Export to ONNX ===
torch.onnx.export(
    model,
    dummy_input,
    "../mobilenetv2_unified_finetuned.pth.onnx",
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
)

print("Exported: mobilenetv2_unified_finetuned.onnx")
