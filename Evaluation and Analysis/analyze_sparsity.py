import torch
import torch.nn.utils.prune as prune
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np

# Load the pruned model
model = models.mobilenet_v2(num_classes=10)
model.load_state_dict(torch.load("../mobilenetv2_taskaware_pruned_finetuned.pth"))
model.eval()

# Function to compute sparsity
sparsity_data = []

for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        weight = module.weight.detach().cpu().numpy()
        total_params = weight.size
        nonzero_params = np.count_nonzero(weight)
        sparsity = 100 * (1 - (nonzero_params / total_params))
        sparsity_data.append((name, round(sparsity, 2)))

# Sort by sparsity descending
sparsity_data.sort(key=lambda x: x[1], reverse=True)

# Plot
layers = [x[0] for x in sparsity_data]
sparsity_values = [x[1] for x in sparsity_data]

plt.figure(figsize=(10, 6))
plt.barh(layers, sparsity_values, color='slateblue')
plt.xlabel("Sparsity (%)")
plt.title("Structured Pruning Sparsity per Conv Layer")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
