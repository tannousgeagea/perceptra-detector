"""
Basic tests for Perceptra Detector core functionality.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import json

from perceptra_detector import Detector, DetectionResult, BoundingBox, Detection
from perceptra_detector.core.registry import DetectorRegistry
from perceptra_detector.core.schemas import TaskType


class TestDetectorRegistry:
    """Test detector registry system."""
    
    def test_list_backends(self):
        """Test listing available backends."""
        backends = DetectorRegistry.list_backends()
        assert isinstance(backends, list)
        assert len(backends) > 0
        # Check for expected backends
        assert 'yolo' in backends or 'detr' in backends or 'rt-detr' in backends
    
    def test_get_supported_extensions(self):
        """Test getting supported extensions."""
        extensions = DetectorRegistry.get_supported_extensions()
        assert isinstance(extensions, dict)
        # Check for common extensions
        assert '.pt' in extensions or '.pth' in extensions
    
    def test_get_backend(self):
        """Test getting a specific backend."""
        backends = DetectorRegistry.list_backends()
        if 'yolo' in backends:
            backend_class = DetectorRegistry.get_backend('yolo')
            assert backend_class is not None
    
    def test_get_backend_not_found(self):
        """Test getting non-existent backend raises error."""
        with pytest.raises(KeyError):
            DetectorRegistry.get_backend('nonexistent_backend')
    
    def test_detect_backend_from_file(self):
        """Test auto-detecting backend from file extension."""
        pt_file = Path("model.pt")
        backend = DetectorRegistry.detect_backend(pt_file)
        assert backend is not None
        
        pth_file = Path("model.pth")
        backend = DetectorRegistry.detect_backend(pth_file)
        assert backend is not None


class TestBoundingBox:
    """Test BoundingBox class."""
    
    def test_bbox_creation(self):
        """Test creating a bounding box."""
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=200)
        assert bbox.x1 == 10
        assert bbox.y1 == 20
        assert bbox.x2 == 100
        assert bbox.y2 == 200
    
    def test_bbox_properties(self):
        """Test bounding box computed properties."""
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=200)
        
        assert bbox.width == 90
        assert bbox.height == 180
        assert bbox.center == (55, 110)
        assert bbox.area == 16200
    
    def test_bbox_conversions(self):
        """Test bounding box format conversions."""
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=200)
        
        # Test xyxy format
        xyxy = bbox.to_xyxy()
        assert xyxy == [10, 20, 100, 200]
        
        # Test xywh format
        xywh = bbox.to_xywh()
        assert xywh == [10, 20, 90, 180]
        
        # Test cxcywh format
        cxcywh = bbox.to_cxcywh()
        assert cxcywh == [55, 110, 90, 180]
    
    def test_bbox_to_dict(self):
        """Test bounding box to dictionary conversion."""
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=200)
        bbox_dict = bbox.to_dict()
        
        assert isinstance(bbox_dict, dict)
        assert bbox_dict['x1'] == 10
        assert bbox_dict['y1'] == 20
        assert bbox_dict['x2'] == 100
        assert bbox_dict['y2'] == 200
        assert bbox_dict['width'] == 90
        assert bbox_dict['height'] == 180


class TestDetection:
    """Test Detection class."""
    
    def test_detection_creation(self):
        """Test creating a detection."""
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=200)
        detection = Detection(
            bbox=bbox,
            confidence=0.95,
            class_id=0,
            class_name="person"
        )
        
        assert detection.bbox == bbox
        assert detection.confidence == 0.95
        assert detection.class_id == 0
        assert detection.class_name == "person"
        assert detection.mask is None
        assert detection.track_id is None
    
    def test_detection_with_mask(self):
        """Test detection with segmentation mask."""
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=200)
        mask = np.random.rand(180, 90) > 0.5
        
        detection = Detection(
            bbox=bbox,
            confidence=0.95,
            class_id=0,
            class_name="person",
            mask=mask
        )
        
        assert detection.mask is not None
        assert detection.mask.shape == (180, 90)
    
    def test_detection_to_dict(self):
        """Test detection to dictionary conversion."""
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=200)
        detection = Detection(
            bbox=bbox,
            confidence=0.95,
            class_id=0,
            class_name="person"
        )
        
        det_dict = detection.to_dict()
        
        assert isinstance(det_dict, dict)
        assert det_dict['confidence'] == 0.95
        assert det_dict['class_name'] == "person"
        assert det_dict['class_id'] == 0
        assert 'bbox' in det_dict
        assert isinstance(det_dict['bbox'], dict)
    
    def test_detection_with_metadata(self):
        """Test detection with custom metadata."""
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=200)
        metadata = {"custom_field": "value", "score": 0.99}
        
        detection = Detection(
            bbox=bbox,
            confidence=0.95,
            class_id=0,
            class_name="person",
            metadata=metadata
        )
        
        assert detection.metadata == metadata
        det_dict = detection.to_dict()
        assert 'metadata' in det_dict


class TestDetectionResult:
    """Test DetectionResult class."""
    
    @pytest.fixture
    def sample_detections(self):
        """Create sample detections for testing."""
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=200)
        return [
            Detection(bbox=bbox, confidence=0.9, class_id=0, class_name="person"),
            Detection(bbox=bbox, confidence=0.8, class_id=1, class_name="car"),
            Detection(bbox=bbox, confidence=0.7, class_id=0, class_name="person"),
            Detection(bbox=bbox, confidence=0.6, class_id=2, class_name="dog"),
        ]
    
    def test_detection_result_creation(self, sample_detections):
        """Test creating a detection result."""
        result = DetectionResult(
            detections=sample_detections,
            image_shape=(640, 640, 3),
            task_type=TaskType.DETECTION,
            model_name="test_model",
            inference_time=0.1
        )
        
        assert len(result) == 4
        assert result.image_shape == (640, 640, 3)
        assert result.task_type == TaskType.DETECTION
        assert result.model_name == "test_model"
        assert result.inference_time == 0.1
    
    def test_detection_result_iteration(self, sample_detections):
        """Test iterating over detections."""
        result = DetectionResult(
            detections=sample_detections,
            image_shape=(640, 640, 3),
            task_type=TaskType.DETECTION,
            model_name="test_model",
            inference_time=0.1
        )
        
        count = 0
        for det in result:
            assert isinstance(det, Detection)
            count += 1
        assert count == 4
    
    def test_detection_result_indexing(self, sample_detections):
        """Test indexing detections."""
        result = DetectionResult(
            detections=sample_detections,
            image_shape=(640, 640, 3),
            task_type=TaskType.DETECTION,
            model_name="test_model",
            inference_time=0.1
        )
        
        assert result[0].class_name == "person"
        assert result[1].class_name == "car"
    
    def test_filter_by_confidence(self, sample_detections):
        """Test filtering detections by confidence."""
        result = DetectionResult(
            detections=sample_detections,
            image_shape=(640, 640, 3),
            task_type=TaskType.DETECTION,
            model_name="test_model",
            inference_time=0.1
        )
        
        filtered = result.filter_by_confidence(0.75)
        assert len(filtered) == 2  # Only 0.9 and 0.8
        assert all(d.confidence >= 0.75 for d in filtered.detections)
    
    def test_filter_by_class(self, sample_detections):
        """Test filtering detections by class."""
        result = DetectionResult(
            detections=sample_detections,
            image_shape=(640, 640, 3),
            task_type=TaskType.DETECTION,
            model_name="test_model",
            inference_time=0.1
        )
        
        filtered = result.filter_by_class(["person"])
        assert len(filtered) == 2
        assert all(d.class_name == "person" for d in filtered.detections)
        
        filtered_multi = result.filter_by_class(["person", "car"])
        assert len(filtered_multi) == 3
    
    def test_get_class_counts(self, sample_detections):
        """Test getting class counts."""
        result = DetectionResult(
            detections=sample_detections,
            image_shape=(640, 640, 3),
            task_type=TaskType.DETECTION,
            model_name="test_model",
            inference_time=0.1
        )
        
        counts = result.get_class_counts()
        assert counts["person"] == 2
        assert counts["car"] == 1
        assert counts["dog"] == 1
    
    def test_detection_result_to_dict(self, sample_detections):
        """Test converting result to dictionary."""
        result = DetectionResult(
            detections=sample_detections,
            image_shape=(640, 640, 3),
            task_type=TaskType.DETECTION,
            model_name="test_model",
            inference_time=0.1
        )
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict['num_detections'] == 4
        assert 'detections' in result_dict
        assert 'image_shape' in result_dict
        assert 'class_counts' in result_dict
        assert result_dict['inference_time'] == 0.1
    
    def test_detection_result_to_json(self, sample_detections):
        """Test converting result to JSON."""
        result = DetectionResult(
            detections=sample_detections,
            image_shape=(640, 640, 3),
            task_type=TaskType.DETECTION,
            model_name="test_model",
            inference_time=0.1
        )
        
        json_str = result.to_json()
        assert isinstance(json_str, str)
        
        # Verify it's valid JSON
        parsed = json.loads(json_str)
        assert parsed['num_detections'] == 4


class TestDetector:
    """Test Detector class."""
    
    def test_list_backends(self):
        """Test listing backends from Detector class."""
        backends = Detector.list_backends()
        assert isinstance(backends, list)
        assert len(backends) > 0
    
    def test_list_supported_extensions(self):
        """Test listing supported extensions."""
        extensions = Detector.list_supported_extensions()
        assert isinstance(extensions, dict)
        assert len(extensions) > 0


class TestUtilities:
    """Test utility functions."""
    
    def test_image_resize(self):
        """Test image resizing utility."""
        from perceptra_detector.utils.image import resize_image
        
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Test resize without keeping aspect ratio
        resized = resize_image(image, (320, 320), keep_aspect_ratio=False)
        assert resized.shape[:2] == (320, 320)
    
    def test_image_padding(self):
        """Test image padding utility."""
        from perceptra_detector.utils.image import pad_image
        
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        padded, (pad_x, pad_y) = pad_image(image, (800, 800))
        assert padded.shape[:2] == (800, 800)
        assert isinstance(pad_x, int)
        assert isinstance(pad_y, int)
    
    def test_image_crop(self):
        """Test image cropping utility."""
        from perceptra_detector.utils.image import crop_image
        
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        bbox = (100, 100, 300, 300)
        
        cropped = crop_image(image, bbox)
        assert cropped.shape[:2] == (200, 200)
    
    def test_image_normalize(self):
        """Test image normalization."""
        from perceptra_detector.utils.image import normalize_image
        
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        normalized = normalize_image(image)
        
        assert normalized.dtype == np.float32
        assert normalized.min() >= -5  # Reasonable range after normalization
        assert normalized.max() <= 5
    
    def test_generate_colors(self):
        """Test color generation for visualization."""
        from perceptra_detector.utils.visualization import generate_colors
        
        colors = generate_colors(10, seed=42)
        assert len(colors) == 10
        assert all(isinstance(c, tuple) and len(c) == 3 for c in colors)
        assert all(0 <= val <= 255 for c in colors for val in c)
        
        # Test reproducibility with same seed
        colors2 = generate_colors(10, seed=42)
        assert colors == colors2
    
    def test_coco_classes(self):
        """Test COCO class names."""
        from perceptra_detector.utils.coco_classes import COCO_CLASSES, COCO_CLASSES_DICT
        
        assert isinstance(COCO_CLASSES, list)
        assert len(COCO_CLASSES) == 80
        assert "person" in COCO_CLASSES
        assert "car" in COCO_CLASSES
        
        assert isinstance(COCO_CLASSES_DICT, dict)
        assert len(COCO_CLASSES_DICT) == 80
        assert COCO_CLASSES_DICT[0] == "person"


class TestConfiguration:
    """Test configuration management."""
    
    def test_default_config(self):
        """Test getting default configuration."""
        from perceptra_detector.utils.config import get_default_config
        
        config = get_default_config()
        assert isinstance(config, dict)
        assert 'detector' in config
        assert 'api' in config
        assert 'logging' in config
    
    def test_config_class(self):
        """Test Config class."""
        from perceptra_detector.utils.config import Config
        
        config = Config.from_default()
        assert config is not None
        assert hasattr(config, 'detector')
        assert hasattr(config, 'api')
    
    def test_config_save_load(self):
        """Test saving and loading configuration."""
        from perceptra_detector.utils.config import Config, save_config, load_config
        
        config_dict = {
            'detector': {
                'model_path': 'test.pt',
                'device': 'cpu'
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save
            save_config(config_dict, temp_path)
            assert Path(temp_path).exists()
            
            # Load
            loaded = load_config(temp_path)
            assert loaded['detector']['model_path'] == 'test.pt'
            assert loaded['detector']['device'] == 'cpu'
        finally:
            Path(temp_path).unlink()


class TestTaskType:
    """Test TaskType enum."""
    
    def test_task_types(self):
        """Test available task types."""
        assert hasattr(TaskType, 'DETECTION')
        assert hasattr(TaskType, 'SEGMENTATION')
        assert hasattr(TaskType, 'INSTANCE_SEGMENTATION')
        
        assert TaskType.DETECTION.value == "detection"
        assert TaskType.SEGMENTATION.value == "segmentation"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])