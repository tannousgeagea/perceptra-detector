"""
Basic tests for Perceptra Detector.
"""

import pytest
import numpy as np
from pathlib import Path

from perceptra_detector import Detector, DetectionResult, BoundingBox, Detection, TaskType
from perceptra_detector.core.registry import DetectorRegistry


class TestDetectorRegistry:
    """Test detector registry."""
    
    def test_list_backends(self):
        """Test listing available backends."""
        backends = DetectorRegistry.list_backends()
        assert isinstance(backends, list)
        assert len(backends) > 0
    
    def test_get_supported_extensions(self):
        """Test getting supported extensions."""
        extensions = DetectorRegistry.get_supported_extensions()
        assert isinstance(extensions, dict)
        assert '.pt' in extensions or '.pth' in extensions


class TestSchemas:
    """Test schema classes."""
    
    def test_bounding_box(self):
        """Test BoundingBox class."""
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=200)
        
        assert bbox.width == 90
        assert bbox.height == 180
        assert bbox.center == (55, 110)
        assert bbox.area == 16200
        
        # Test conversions
        xyxy = bbox.to_xyxy()
        assert xyxy == [10, 20, 100, 200]
        
        xywh = bbox.to_xywh()
        assert xywh == [10, 20, 90, 180]
        
        cxcywh = bbox.to_cxcywh()
        assert cxcywh == [55, 110, 90, 180]
    
    def test_detection(self):
        """Test Detection class."""
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=200)
        detection = Detection(
            bbox=bbox,
            confidence=0.95,
            class_id=0,
            class_name="person"
        )
        
        assert detection.confidence == 0.95
        assert detection.class_name == "person"
        
        # Test to_dict
        det_dict = detection.to_dict()
        assert det_dict['confidence'] == 0.95
        assert det_dict['class_name'] == "person"
        assert 'bbox' in det_dict
    
    def test_detection_result(self):
        """Test DetectionResult class."""
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=200)
        detections = [
            Detection(bbox=bbox, confidence=0.9, class_id=0, class_name="person"),
            Detection(bbox=bbox, confidence=0.8, class_id=1, class_name="car"),
            Detection(bbox=bbox, confidence=0.7, class_id=0, class_name="person"),
        ]
        
        result = DetectionResult(
            detections=detections,
            image_shape=(640, 640, 3),
            task_type=TaskType.DETECTION,
            model_name="test_model",
            inference_time=0.1
        )
        
        assert len(result) == 3
        assert result[0].class_name == "person"
        
        # Test filtering
        filtered = result.filter_by_confidence(0.85)
        assert len(filtered) == 1
        
        filtered_class = result.filter_by_class(["person"])
        assert len(filtered_class) == 2
        
        # Test class counts
        counts = result.get_class_counts()
        assert counts["person"] == 2
        assert counts["car"] == 1
        
        # Test to_dict
        result_dict = result.to_dict()
        assert result_dict['num_detections'] == 3
        assert 'detections' in result_dict


class TestDetector:
    """Test Detector class."""
    
    @pytest.fixture
    def dummy_image(self):
        """Create a dummy image."""
        return np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    def test_detector_initialization(self):
        """Test detector initialization."""
        # This would require an actual model file
        # In real tests, use pytest fixtures with mock models
        pass
    
    def test_list_backends(self):
        """Test listing backends."""
        backends = Detector.list_backends()
        assert isinstance(backends, list)
        assert 'yolo' in backends or 'detr' in backends
    
    def test_list_supported_extensions(self):
        """Test listing supported extensions."""
        extensions = Detector.list_supported_extensions()
        assert isinstance(extensions, dict)


class TestUtils:
    """Test utility functions."""
    
    def test_image_loading(self):
        """Test image loading utilities."""
        from perceptra_detector.utils.image import resize_image, pad_image
        
        # Create test image
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Test resize
        resized = resize_image(image, (320, 320), keep_aspect_ratio=False)
        assert resized.shape[:2] == (320, 320)
        
        # Test pad
        padded, (pad_x, pad_y) = pad_image(image, (800, 800))
        assert padded.shape[:2] == (800, 800)
    
    def test_visualization(self):
        """Test visualization utilities."""
        from perceptra_detector.utils.visualization import generate_colors
        
        colors = generate_colors(10)
        assert len(colors) == 10
        assert all(isinstance(c, tuple) and len(c) == 3 for c in colors)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])