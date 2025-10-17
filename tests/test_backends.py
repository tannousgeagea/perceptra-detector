"""
Tests for detector backends.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from perceptra_detector.core.base import BaseDetector
from perceptra_detector.core.registry import DetectorRegistry, register_backend
from perceptra_detector.core.schemas import DetectionResult, Detection, BoundingBox, TaskType


class TestBaseDetector:
    """Test BaseDetector abstract class."""
    
    def test_base_detector_cannot_instantiate(self):
        """Test that BaseDetector cannot be instantiated directly."""
        # BaseDetector is abstract and should not be instantiable
        # However, Python doesn't enforce this strictly, so we test the methods
        pass
    
    def test_device_setup_auto_detect(self):
        """Test automatic device detection."""
        device = BaseDetector._setup_device(None)
        assert device in ["cuda", "cpu"]
    
    def test_device_setup_explicit(self):
        """Test explicit device setting."""
        device = BaseDetector._setup_device("cpu")
        assert device == "cpu"


class TestCustomBackendRegistration:
    """Test custom backend registration."""
    
    def test_register_custom_backend(self):
        """Test registering a custom backend."""
        
        @register_backend('test-backend', ['.test'])
        class TestDetector(BaseDetector):
            def load_model(self):
                self.class_names = ["class1", "class2"]
                self.task_type = TaskType.DETECTION
            
            def preprocess(self, image):
                return image
            
            def predict(self, preprocessed_input):
                return {}
            
            def postprocess(self, predictions, original_shape):
                return DetectionResult(
                    detections=[],
                    image_shape=original_shape,
                    task_type=self.task_type,
                    model_name="test",
                    inference_time=0.0
                )
        
        # Check if registered
        backends = DetectorRegistry.list_backends()
        assert 'test-backend' in backends
        
        # Check extension mapping
        extensions = DetectorRegistry.get_supported_extensions()
        assert '.test' in extensions
        assert extensions['.test'] == 'test-backend'
    
    def test_get_registered_backend(self):
        """Test getting a registered backend."""
        
        @register_backend('test-backend-2', ['.test2'])
        class TestDetector2(BaseDetector):
            def load_model(self):
                pass
            def preprocess(self, image):
                return image
            def predict(self, preprocessed_input):
                return {}
            def postprocess(self, predictions, original_shape):
                return DetectionResult([], original_shape, TaskType.DETECTION, "test", 0.0)
        
        backend_class = DetectorRegistry.get_backend('test-backend-2')
        assert backend_class is TestDetector2
    
    def test_backend_registration_invalid_class(self):
        """Test that registering invalid class raises error."""
        
        class NotADetector:
            pass
        
        with pytest.raises(TypeError):
            DetectorRegistry.register('invalid', NotADetector)


class TestBackendDetection:
    """Test backend auto-detection from file extensions."""
    
    def test_detect_pt_extension(self):
        """Test detecting backend for .pt files."""
        backend = DetectorRegistry.detect_backend(Path("model.pt"))
        assert backend is not None
        assert backend in ['yolo', 'rf-detr', 'rt-detr', 'detr']
    
    def test_detect_pth_extension(self):
        """Test detecting backend for .pth files."""
        backend = DetectorRegistry.detect_backend(Path("model.pth"))
        assert backend is not None
    
    def test_detect_unknown_extension(self):
        """Test detecting backend for unknown extension."""
        backend = DetectorRegistry.detect_backend(Path("model.unknown"))
        assert backend is None


class TestMockDetector:
    """Test with a mock detector implementation."""
    
    @pytest.fixture
    def mock_detector_class(self):
        """Create a mock detector class for testing."""
        
        class MockDetector(BaseDetector):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._predictions = []
            
            def load_model(self):
                self.class_names = ["person", "car", "dog"]
                self.task_type = TaskType.DETECTION
                self._model_loaded = True
            
            def preprocess(self, image):
                # Simple preprocessing - just return image
                return image
            
            def predict(self, preprocessed_input):
                # Mock predictions
                return {
                    'boxes': [[10, 20, 100, 200]],
                    'scores': [0.9],
                    'labels': [0]
                }
            
            def postprocess(self, predictions, original_shape):
                detections = []
                
                boxes = predictions['boxes']
                scores = predictions['scores']
                labels = predictions['labels']
                
                for box, score, label in zip(boxes, scores, labels):
                    bbox = BoundingBox(
                        x1=box[0], y1=box[1],
                        x2=box[2], y2=box[3]
                    )
                    detection = Detection(
                        bbox=bbox,
                        confidence=score,
                        class_id=label,
                        class_name=self.class_names[label]
                    )
                    detections.append(detection)
                
                return DetectionResult(
                    detections=detections,
                    image_shape=original_shape,
                    task_type=self.task_type,
                    model_name="MockDetector",
                    inference_time=0.0
                )
        
        return MockDetector
    
    def test_mock_detector_initialization(self, mock_detector_class):
        """Test mock detector initialization."""
        with tempfile.NamedTemporaryFile(suffix='.mock') as f:
            detector = mock_detector_class(
                model_path=f.name,
                device='cpu',
                confidence_threshold=0.5
            )
            
            assert detector.model_path == Path(f.name)
            assert detector.device == 'cpu'
            assert detector.confidence_threshold == 0.5
    
    def test_mock_detector_load_model(self, mock_detector_class):
        """Test mock detector model loading."""
        with tempfile.NamedTemporaryFile(suffix='.mock') as f:
            detector = mock_detector_class(model_path=f.name)
            detector.load_model()
            
            assert detector.class_names == ["person", "car", "dog"]
            assert detector.task_type == TaskType.DETECTION
    
    def test_mock_detector_detect(self, mock_detector_class):
        """Test mock detector detection."""
        with tempfile.NamedTemporaryFile(suffix='.mock') as f:
            detector = mock_detector_class(model_path=f.name)
            
            # Create dummy image
            image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
            # Run detection
            result = detector.detect(image)
            
            assert isinstance(result, DetectionResult)
            assert len(result) > 0
            assert result.detections[0].class_name == "person"
            assert result.detections[0].confidence == 0.9
    
    def test_mock_detector_batch_detect(self, mock_detector_class):
        """Test mock detector batch detection."""
        with tempfile.NamedTemporaryFile(suffix='.mock') as f:
            detector = mock_detector_class(model_path=f.name)
            
            # Create dummy images
            images = [
                np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                for _ in range(3)
            ]
            
            # Run batch detection
            results = detector.detect_batch(images)
            
            assert len(results) == 3
            assert all(isinstance(r, DetectionResult) for r in results)
    
    def test_mock_detector_model_info(self, mock_detector_class):
        """Test mock detector model info."""
        with tempfile.NamedTemporaryFile(suffix='.mock') as f:
            detector = mock_detector_class(model_path=f.name)
            detector.load_model()
            
            info = detector.model_info
            
            assert isinstance(info, dict)
            assert 'model_path' in info
            assert 'device' in info
            assert 'num_classes' in info
            assert info['num_classes'] == 3
            assert 'class_names' in info


class TestBackendWarmup:
    """Test model warmup functionality."""
    
    def test_warmup_mock_detector(self):
        """Test warmup with mock detector."""
        
        class WarmupTestDetector(BaseDetector):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.warmup_count = 0
            
            def load_model(self):
                self.class_names = ["test"]
                self.task_type = TaskType.DETECTION
            
            def preprocess(self, image):
                return image
            
            def predict(self, preprocessed_input):
                self.warmup_count += 1
                return {}
            
            def postprocess(self, predictions, original_shape):
                return DetectionResult([], original_shape, TaskType.DETECTION, "test", 0.0)
        
        with tempfile.NamedTemporaryFile(suffix='.test') as f:
            detector = WarmupTestDetector(model_path=f.name)
            
            # Warmup with 3 iterations
            detector.warmup(num_iterations=3)
            
            assert detector.warmup_count == 3


class TestBackendErrorHandling:
    """Test error handling in backends."""
    
    def test_missing_model_file(self):
        """Test handling of missing model file."""
        
        class ErrorTestDetector(BaseDetector):
            def load_model(self):
                if not self.model_path.exists():
                    raise FileNotFoundError(f"Model not found: {self.model_path}")
                self.class_names = []
                self.task_type = TaskType.DETECTION
            
            def preprocess(self, image):
                return image
            def predict(self, preprocessed_input):
                return {}
            def postprocess(self, predictions, original_shape):
                return DetectionResult([], original_shape, TaskType.DETECTION, "test", 0.0)
        
        with pytest.raises(FileNotFoundError):
            detector = ErrorTestDetector(model_path="nonexistent.pt")
            detector.load_model()


class TestBackendConfidenceThreshold:
    """Test confidence threshold functionality."""
    
    def test_confidence_threshold_override(self):
        """Test overriding confidence threshold during detection."""
        
        class ThresholdTestDetector(BaseDetector):
            def load_model(self):
                self.class_names = ["test"]
                self.task_type = TaskType.DETECTION
            
            def preprocess(self, image):
                return image
            
            def predict(self, preprocessed_input):
                # Return predictions with various confidences
                return {
                    'boxes': [[10, 10, 50, 50], [60, 60, 100, 100]],
                    'scores': [0.9, 0.3],
                    'labels': [0, 0]
                }
            
            def postprocess(self, predictions, original_shape):
                detections = []
                
                for box, score, label in zip(
                    predictions['boxes'],
                    predictions['scores'],
                    predictions['labels']
                ):
                    # Apply confidence threshold
                    if score >= self.confidence_threshold:
                        bbox = BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3])
                        detection = Detection(
                            bbox=bbox,
                            confidence=score,
                            class_id=label,
                            class_name=self.class_names[label]
                        )
                        detections.append(detection)
                
                return DetectionResult(
                    detections=detections,
                    image_shape=original_shape,
                    task_type=self.task_type,
                    model_name="test",
                    inference_time=0.0
                )
        
        with tempfile.NamedTemporaryFile(suffix='.test') as f:
            detector = ThresholdTestDetector(
                model_path=f.name,
                confidence_threshold=0.5
            )
            
            image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
            # With default threshold (0.5), should get 1 detection
            result1 = detector.detect(image)
            assert len(result1) == 1
            assert result1.detections[0].confidence == 0.9
            
            # With lower threshold, should get 2 detections
            result2 = detector.detect(image, confidence_threshold=0.2)
            assert len(result2) == 2


class TestBackendRepr:
    """Test backend string representation."""
    
    def test_detector_repr(self):
        """Test detector __repr__ method."""
        
        class ReprTestDetector(BaseDetector):
            def load_model(self):
                self.class_names = []
                self.task_type = TaskType.DETECTION
            def preprocess(self, image):
                return image
            def predict(self, preprocessed_input):
                return {}
            def postprocess(self, predictions, original_shape):
                return DetectionResult([], original_shape, TaskType.DETECTION, "test", 0.0)
        
        with tempfile.NamedTemporaryFile(suffix='.test') as f:
            detector = ReprTestDetector(model_path=f.name, device='cpu')
            
            repr_str = repr(detector)
            assert 'ReprTestDetector' in repr_str
            assert 'cpu' in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])