# Perceptra Detector

Production-ready object detection and segmentation framework with unified interface for YOLO, DETR, RT-DETR, and custom models.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- 🎯 **Unified Interface**: Single API for multiple detection backends (YOLO, DETR, RT-DETR)
- 🔌 **Pluggable Backends**: Easy to add custom models and new architectures
- 🚀 **Production Ready**: FastAPI server, Docker support, comprehensive error handling
- 📊 **Rich Output**: Standardized detection results with visualization utilities
- 🎨 **Multiple Interfaces**: Python SDK, CLI, and REST API
- ⚡ **Performance**: GPU acceleration, batch processing, model warmup
- 🔧 **Extensible**: Reserved architecture for future training capabilities

## Installation

### Basic Installation

```bash
pip install perceptra-detector
```

### With Specific Backends

```bash
# YOLO support
pip install perceptra-detector[yolo]

# DETR support
pip install perceptra-detector[detr]

# API server
pip install perceptra-detector[api]

# Everything
pip install perceptra-detector[all]
```

### From Source

```bash
git clone https://github.com/tannousgeagea/perceptra-detector.git
cd perceptra-detector
pip install -e .
```

## Quick Start

### Python API

```python
from perceptra_detector import Detector

# Initialize detector (auto-detects backend from file extension)
detector = Detector("yolov8n.pt")

# Run detection on an image
result = detector.detect("image.jpg")

# Print results
print(f"Found {len(result)} objects")
for detection in result:
    print(f"{detection.class_name}: {detection.confidence:.2f}")

# Visualize results
from perceptra_detector.utils.visualization import draw_detections
from perceptra_detector.utils.image import load_image, save_image

image = load_image("image.jpg")
annotated = draw_detections(image, result)
save_image(annotated, "output.jpg")
```

### Batch Processing

```python
# Process multiple images
images = ["img1.jpg", "img2.jpg", "img3.jpg"]
batch_result = detector.detect_batch(images)

# Process entire directory
batch_result = detector.detect_directory(
    "images/",
    recursive=True,
    confidence_threshold=0.5
)
```

### Video Processing

```python
# Process video and save annotated output
results = detector.detect_video(
    video_path="input.mp4",
    output_path="output.mp4",
    skip_frames=2  # Process every 3rd frame
)
```

### CLI Usage

```bash
# Single image detection
perceptra-detector detect yolov8n.pt image.jpg -o output.jpg

# Batch processing
perceptra-detector batch yolov8n.pt images/ -o results/ --recursive

# Video processing
perceptra-detector video yolov8n.pt video.mp4 -o output.mp4

# Start API server
perceptra-detector serve -m yolo:yolov8n.pt -m detr:detr-model.pth --port 8000

# List available backends
perceptra-detector list-backends
```

## API Server

### Starting the Server

```bash
# Start with models
perceptra-detector serve \
    -m yolo:models/yolov8n.pt \
    -m detr:models/detr-resnet-50.pth \
    --host 0.0.0.0 \
    --port 8000
```

### API Endpoints

- `GET /health` - Health check
- `GET /models` - List available models
- `GET /models/{model_name}` - Get model info
- `POST /detect` - Detect objects in image
- `POST /detect/batch` - Batch detection
- `POST /detect/url` - Detect from image URL

### Using the Python SDK Client

```python
from perceptra_detector.client import DetectorClient

# Connect to API
client = DetectorClient("http://localhost:8000")

# Check health
health = client.health_check()
print(health)

# List models
models = client.list_models()
print(f"Available models: {models['models']}")

# Run detection
result = client.detect("image.jpg", model_name="yolo")
print(f"Found {result['num_detections']} objects")

# Batch detection
results = client.detect_batch(
    ["img1.jpg", "img2.jpg"],
    confidence_threshold=0.5
)
```

### cURL Examples

```bash
# Health check
curl http://localhost:8000/health

# List models
curl http://localhost:8000/models

# Detect objects
curl -X POST \
  -F "file=@image.jpg" \
  -F "confidence_threshold=0.5" \
  http://localhost:8000/detect

# Detect from URL
curl -X POST \
  "http://localhost:8000/detect/url?url=https://example.com/image.jpg"
```

## Docker Deployment

### Build Image

```bash
docker build -t perceptra-detector .
```

### Run Container

```bash
# CPU
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  perceptra-detector

# GPU (requires nvidia-docker)
docker run --gpus all -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  perceptra-detector
```

### Docker Compose

```yaml
version: '3.8'

services:
  detector:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
    environment:
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## Supported Backends

### YOLO (YOLOv8, YOLOv9, YOLOv11)

```python
detector = Detector(
    "yolov8n.pt",
    backend="yolo",
    device="cuda",
    imgsz=640
)
```

**Supported formats**: `.pt`, `.onnx`

### DETR

```python
detector = Detector(
    "facebook/detr-resnet-50",  # HuggingFace model
    backend="detr",
    use_transformers=True
)
```

**Supported formats**: `.pth`, `.pt`

### RT-DETR

```python
detector = Detector(
    "rtdetr-l.pt",
    backend="rt-detr"
)
```

**Supported formats**: `.pt`

### Custom Models

```python
# Register custom backend
from perceptra_detector.core.base import BaseDetector
from perceptra_detector.core.registry import register_backend

@register_backend('custom', ['.custom'])
class CustomDetector(BaseDetector):
    def load_model(self):
        # Load your model
        pass
    
    def preprocess(self, image):
        # Preprocess image
        pass
    
    def predict(self, preprocessed_input):
        # Run inference
        pass
    
    def postprocess(self, predictions, original_shape):
        # Convert to DetectionResult
        pass

# Use custom backend
detector = Detector("model.custom", backend="custom")
```

## Advanced Usage

### Custom Confidence and IoU Thresholds

```python
result = detector.detect(
    "image.jpg",
    confidence_threshold=0.7,
    iou_threshold=0.5
)
```

### Filter Results

```python
# Filter by confidence
filtered = result.filter_by_confidence(0.8)

# Filter by class
filtered = result.filter_by_class(["person", "car"])

# Get class counts
counts = result.get_class_counts()
```

### Model Warmup

```python
# Warmup GPU
detector.warmup(num_iterations=5)
```

### Get Model Information

```python
info = detector.model_info
print(f"Device: {info['device']}")
print(f"Classes: {info['class_names']}")
```

### Export Results

```python
# To dictionary
result_dict = result.to_dict()

# To JSON
json_str = result.to_json()

# Save to file
import json
with open("results.json", "w") as f:
    json.dump(result.to_dict(), f, indent=2)
```

## Configuration

Create a `config.yaml` file:

```yaml
detector:
  model_path: "models/yolov8n.pt"
  backend: "yolo"
  device: "cuda"
  confidence_threshold: 0.25
  iou_threshold: 0.45
  auto_warmup: true

api:
  host: "0.0.0.0"
  port: 8000
  enable_cors: true
  models:
    yolo: "models/yolov8n.pt"
    detr: "models/detr-resnet-50.pth"
```

Load configuration:

```python
from perceptra_detector.utils.config import load_config

config = load_config("config.yaml")
detector = Detector(**config['detector'])
```

## Development

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/tannousgeagea/perceptra-detector.git
cd perceptra-detector

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev,all]"
```

### Run Tests

```bash
pytest tests/ -v --cov=perceptra_detector
```

### Code Formatting

```bash
# Format code
black perceptra_detector/
isort perceptra_detector/

# Check style
flake8 perceptra_detector/
mypy perceptra_detector/
```

## Roadmap

- [x] Core detection interface
- [x] YOLO backend
- [x] DETR backend
- [x] RT-DETR backend
- [x] FastAPI server
- [x] Python SDK client
- [x] CLI interface
- [x] Docker support
- [ ] Model training module
- [ ] Fine-tuning utilities
- [ ] Model quantization
- [ ] ONNX export/optimization
- [ ] Tracking support
- [ ] 3D detection support
- [ ] AutoML model selection

## Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use Perceptra Detector in your research, please cite:

```bibtex
@software{perceptra_detector,
  title={Perceptra Detector: Production-Ready Object Detection Framework},
  author={Perceptra Team},
  year={2024},
  url={https://github.com/tannousgeagea/perceptra-detector}
}
```

## Acknowledgments

- Built on top of [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLO support
- Uses [Transformers](https://github.com/huggingface/transformers) for DETR models
- Inspired by modern MLOps practices

## Support

- 📖 [Documentation](https://perceptra-detector.readthedocs.io)
- 💬 [Discussions](https://github.com/tannousgeagea/perceptra-detector/discussions)
- 🐛 [Issue Tracker](https://github.com/tannousgeagea/perceptra-detector/issues)
- 📧 Email: support@perceptra.ai