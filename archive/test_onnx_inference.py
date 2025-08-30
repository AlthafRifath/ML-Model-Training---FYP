import onnxruntime as ort
import numpy as np
import time

# Load the ONNX model
ort_session = ort.InferenceSession("mobilenetv2_cifar10_pruned_finetuned.onnx")

# Create dummy input (like a CIFAR-10 image)
dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)

# Run inference
start = time.time()
outputs = ort_session.run(None, {"input": dummy_input})
end = time.time()

# Post-process
output = outputs[0]
predicted_class = np.argmax(output)

print(f"Predicted class: {predicted_class}")
print(f"Inference time: {(end - start) * 1000:.2f} ms")
