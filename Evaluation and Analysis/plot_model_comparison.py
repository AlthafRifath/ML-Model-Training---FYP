import matplotlib.pyplot as plt

# Model names and metrics
models = [
    "Baseline (Unpruned)",
    "Task-Aware + Fine-Tuned",
    "Hardware-Aware (ONNX)",
    "Unified Pruned (Final)"
]

accuracies = [88.04, 84.67, 84.68, 84.67]
times = [90, 3.25, 3.29, 3.25]  # in ms

# Create subplots
fig, ax1 = plt.subplots()

# Accuracy bar (left Y-axis)
color = 'tab:blue'
ax1.set_xlabel("Model Type")
ax1.set_ylabel("Accuracy (%)", color=color)
ax1.bar(models, accuracies, color=color, alpha=0.6)
ax1.tick_params(axis='y', labelcolor=color)
plt.xticks(rotation=10)

# Inference time line (right Y-axis)
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel("Inference Time (ms)", color=color)
ax2.plot(models, times, color=color, marker='o', linewidth=2)
ax2.tick_params(axis='y', labelcolor=color)

# Title and layout
plt.title("Model Accuracy vs. Inference Time (CPU)")
fig.tight_layout()
plt.grid(True)
plt.show()
