# Perceptra Detector - Quick Start Guide

Get started with Perceptra Detector in 5 minutes!

## 🚀 Installation

### Option 1: pip (Recommended)

```bash
# Basic installation
pip install perceptra-detector

# With YOLO support
pip install perceptra-detector[yolo]

# With DETR support
pip install perceptra-detector[detr]

# With API server
pip install perceptra-detector[api]

# Everything
pip install perceptra-detector[all]
```

### Option 2: From Source

```bash
git clone https://github.com/perceptra/perceptra-detector.git
cd perceptra-detector
pip install -e ".[all]"
```

### Option 3: Docker

```bash
docker pull perceptra/detector:latest
docker run -p 8000:8000 -v $(pwd)/models:/app/models perceptra/detector
```

## 📦 Download Models

```bash
# Download YOLOv8 models (requires ultralytics)
pip install ultralytics
yolo export model=yolov8n.pt format=pt

# Or download from Ultralytics directly
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

## 🎯 Basic Usage

### 1. Python API - Single Image

```python
from perceptra_detector import Detector

# Initialize detector
detector = Detector("yolov8n.pt")

# Run detection
result = detector.detect("image.jpg")

# Print results
print(f"Found {len(result)} objects")
for det in result:
    print(f"{det.class_name}: {det.confidence:.2f}")
```

### 2. Command Line - Single Image

```bash
perceptra-detector detect yolov8n.pt image.jpg -o output.jpg --confidence 0.5
```

### 3. Batch Processing

```python
# Process multiple images
detector = Detector("yolov8n.pt")
results = detector.detect_batch(["img1.jpg", "img2.jpg", "img3.jpg"])

# Or process entire directory
results = detector.detect_directory("images/", recursive=True)
```

### 4. Video Processing

```python
detector = Detector("yolov8n.pt")
results = detector.detect_video(
    video_path="input.mp4",
    output_path="output.mp4",
    skip_frames=2  # Process every 3rd frame
)
```

### 5. Start API Server

```bash
# Start server with models
perceptra-detector serve \
    -m yolo:yolov8n.pt \
    -m detr:detr-model.pth \
    --port 8000

# Test the server
curl http://localhost:8000/health
```

### 6. Use API Client

```python
from perceptra_detector.client import DetectorClient

client = DetectorClient("http://localhost:8000")

# Detect from image file
result = client.detect("image.jpg")

# Detect from URL
result = client.detect_url("https://example.com/image.jpg")

# Batch detection
results = client.detect_batch(["img1.jpg", "img2.jpg"])
```

## 🎨 Visualization

```python
from perceptra_detector import Detector
from perceptra_detector.utils.visualization import draw_detections
from perceptra_detector.utils.image import load_image, save_image

# Detect and visualize
detector = Detector("yolov8n.pt")
image = load_image("image.jpg")
result = detector.detect(image)

# Draw detections
annotated = draw_detections(image, result)
save_image(annotated, "output.jpg")
```

## 🔧 Configuration

Create a `config.yaml`:

```yaml
detector:
  model_path: "yolov8n.pt"
  device: "cuda"
  confidence_threshold: 0.25
  iou_threshold: 0.45

api:
  host: "0.0.0.0"
  port: 8000
  models:
    yolo: "models/yolov8n.pt"
```

Load and use:

```python
from perceptra_detector.utils.config import Config

config = Config.from_file("config.yaml")
detector = Detector(**config.detector)
```

## 🐳 Docker Quick Start

### Using Docker

```bash
# Build
docker build -t perceptra-detector .

# Run
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  perceptra-detector

# With GPU
docker run --gpus all -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  perceptra-detector
```

### Using Docker Compose

```bash
# Create docker-compose.yml (see repository)
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

## 📊 Complete Example

```python
"""Complete detection pipeline example."""

from perceptra_detector import Detector
from perceptra_detector.utils.visualization import draw_detections, draw_stats
from perceptra_detector.utils.image import load_image, save_image
import json

# 1. Initialize detector
detector = Detector(
    model_path="yolov8n.pt",
    confidence_threshold=0.5,
    device="cuda",
    auto_warmup=True
)

# 2. Single image detection
image = load_image("test.jpg")
result = detector.detect(image)

print(f"✓ Found {len(result)} objects in {result.inference_time*1000:.1f}ms")

# 3. Print detections
for i, det in enumerate(result.detections, 1):
    print(f"{i}. {det.class_name}: {det.confidence:.3f} at "
          f"({det.bbox.x1:.0f}, {det.bbox.y1:.0f})")

# 4. Visualize
annotated = draw_detections(image, result, show_confidence=True)
annotated = draw_stats(annotated, result)
save_image(annotated, "output.jpg")

# 5. Export results
with open("results.json", "w") as f:
    f.write(result.to_json())

# 6. Filter results
high_conf = result.filter_by_confidence(0.8)
persons = result.filter_by_class(["person"])

print(f"\nHigh confidence: {len(high_conf)} detections")
print(f"Persons only: {len(persons)} detections")

# 7. Class distribution
print("\nClass distribution:")
for class_name, count in result.get_class_counts().items():
    print(f"  {class_name}: {count}")
```

## 🔌 Supported Backends

- **YOLO** (v8, v9, v11): `.pt`, `.onnx`
- **DETR**: `.pth`, `.pt`
- **RT-DETR**: `.pt`
- **Custom**: Implement `BaseDetector`

## 📚 Common Use Cases

### Object Detection in Production

```python
# High-performance setup
detector = Detector(
    "yolov8n.pt",
    device="cuda",
    auto_warmup=True
)

# Process with custom thresholds
result = detector.detect(
    image,
    confidence_threshold=0.7,
    iou_threshold=0.5
)
```

### Batch Processing Pipeline

```python
from pathlib import Path
from tqdm import tqdm

detector = Detector("yolov8n.pt", auto_warmup=True)
output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

for img_path in tqdm(list(Path("images").glob("*.jpg"))):
    result = detector.detect(str(img_path))
    
    # Save results
    with open(output_dir / f"{img_path.stem}.json", "w") as f:
        f.write(result.to_json())
```

### Real-time Video Analysis

```python
import cv2

detector = Detector("yolov8n.pt", device="cuda")
cap = cv2.VideoCapture(0)  # Webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect
    result = detector.detect(frame_rgb)
    
    # Draw and display
    from perceptra_detector.utils.visualization import draw_detections
    annotated = draw_detections(frame_rgb, result)
    annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
    
    cv2.imshow('Detection', annotated_bgr)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## 🆘 Troubleshooting

### Issue: Model not found

```bash
# Ensure model file exists
ls -la yolov8n.pt

# Or use absolute path
detector = Detector("/absolute/path/to/yolov8n.pt")
```

### Issue: CUDA out of memory

```python
# Use CPU instead
detector = Detector("yolov8n.pt", device="cpu")

# Or reduce batch size
results = detector.detect_batch(images[:10])  # Process in smaller batches
```

### Issue: Import error

```bash
# Install missing dependencies
pip install perceptra-detector[all]

# Or specific backend
pip install ultralytics  # For YOLO
pip install transformers  # For DETR
```

## 📖 Next Steps

- Read the [full documentation](https://perceptra-detector.readthedocs.io)
- Check out [examples directory](examples/)
- Join the [community discussions](https://github.com/perceptra/perceptra-detector/discussions)
- Report [issues](https://github.com/perceptra/perceptra-detector/issues)

## 💡 Tips

1. **Use GPU**: Always use `device="cuda"` for faster inference
2. **Warmup**: Enable `auto_warmup=True` for consistent performance
3. **Batch Processing**: Use `detect_batch()` for multiple images
4. **Filtering**: Use built-in filters to reduce false positives
5. **Configuration**: Use YAML config for production deployments

## 🎓 Learning Resources

- [API Documentation](https://perceptra-detector.readthedocs.io/api)
- [Video Tutorials](https://youtube.com/@perceptra)
- [Blog Posts](https://blog.perceptra.ai)
- [Example Notebooks](examples/notebooks/)

Happy detecting! 🎯