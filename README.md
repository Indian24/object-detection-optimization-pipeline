# Efficient Object Detection + Robustness Analysis

End-to-end YOLOv8 pipeline built on a Pascal VOC subset demonstrating training, optimization, and deployment readiness.

## Key Results
- mAP@0.5: ~0.84
- Precision: ~0.86
- Recall: ~0.75
- ~69% latency reduction via input scaling

## Highlights
- Transfer learning using YOLOv8
- Overfitting analysis (train vs val curves)
- Latency optimization experiments
- ONNX export for deployment
- INT8 quantization for edge inference
- Failure case analysis

## Repository Structure
- **models/** → trained weights (PyTorch + ONNX)
- **notebook/** → full pipeline
- **results/** → predictions and failure cases

This project demonstrates a real-world balance between accuracy, efficiency, and deployability in computer vision systems.
