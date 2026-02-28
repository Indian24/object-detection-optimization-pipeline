# Efficient Object Detection + Robustness Analysis

End-to-end YOLOv8 pipeline on Pascal VOC subset.

## Results
- mAP@0.5: ~0.84
- Precision: ~0.86
- Recall: ~0.75
- 69% latency improvement

## Structure
- models/ → trained weights
- notebook/ → full pipeline
- results/ → predictions + failure analysis

Includes ONNX export and INT8 quantization for deployment.