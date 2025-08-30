import matplotlib.pyplot as plt

models = ['Baseline', 'Task-Aware\nFine-Tuned', 'Hardware-Aware\n(ONNX)', 'Unified ONNX\n(Fine-Tuned)']
accuracy = [88.04, 84.67, 84.68, 84.86]
inference_time = [90, 3.42, 3.29, 3.25]
model_size = [13.5, 9.6, 4.7, 4.6]

fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('Model')
ax1.set_ylabel('Accuracy (%)', color=color)
ax1.bar(models, accuracy, color=color, alpha=0.6, label='Accuracy')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim([80, 90])

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Inference Time (ms)', color=color)
ax2.plot(models, inference_time, color=color, marker='o', linewidth=2, label='Inference Time')
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim([0, 100])

plt.title('Model Accuracy vs Inference Time (CPU)')
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.show()
