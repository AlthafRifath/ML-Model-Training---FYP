import onnxruntime as ort
import numpy as np
import time

# Force ONNX Runtime to use CPU only
providers = ['CPUExecutionProvider']
ort_session = ort.InferenceSession("mobilenetv2_cifar10_pruned_finetuned.onnx", providers=providers)

# Create dummy input (same shape as 224x224 RGB image)
dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)

# Run inference and time it
start_time = time.time()
outputs = ort_session.run(None, {"input": dummy_input})
end_time = time.time()

# Output results
output = outputs[0]
predicted_class = np.argmax(output)

print(f"Predicted class: {predicted_class}")
print(f"Inference time (CPU): {(end_time - start_time) * 1000:.2f} ms")
