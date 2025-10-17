"""
Tests for the FastAPI application.
"""

import pytest
from fastapi.testclient import TestClient
import numpy as np
from io import BytesIO
from PIL import Image
import json

from perceptra_detector.api.server import create_app, DetectionService
from perceptra_detector import Detector


@pytest.fixture
def app():
    """Create test FastAPI application."""
    return create_app(enable_cors=True)


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def sample_image_bytes():
    """Create sample image bytes for testing."""
    # Create a simple test image
    image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    img_pil = Image.fromarray(image)
    
    img_byte_arr = BytesIO()
    img_pil.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    
    return img_byte_arr.getvalue()


class TestRootEndpoints:
    """Test root and info endpoints."""
    
    def test_root(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "service" in data
        assert "version" in data
        assert "status" in data
        assert data["status"] == "running"
    
    def test_health(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "models_loaded" in data
        assert "gpu_available" in data
        
        # Status should be degraded if no models loaded
        assert data["status"] in ["healthy", "degraded"]
    
    def test_stats(self, client):
        """Test stats endpoint."""
        response = client.get("/stats")
        assert response.status_code == 200
        
        data = response.json()
        assert "models_loaded" in data
        assert "total_requests" in data


class TestModelEndpoints:
    """Test model management endpoints."""
    
    def test_list_models_empty(self, client):
        """Test listing models when none are loaded."""
        response = client.get("/models")
        # Should return 503 when no models loaded
        assert response.status_code == 503
    
    def test_get_model_not_found(self, client):
        """Test getting info for non-existent model."""
        response = client.get("/models/nonexistent")
        assert response.status_code == 404


class TestDetectionEndpoints:
    """Test detection endpoints."""
    
    def test_detect_no_file(self, client):
        """Test detection without providing a file."""
        response = client.post("/detect")
        assert response.status_code == 422  # Validation error
    
    def test_detect_with_image(self, client, sample_image_bytes):
        """Test detection with image file."""
        # This would require a loaded model
        # In real tests, you'd mock the detector or load a test model
        files = {"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
        response = client.post("/detect", files=files)
        
        # Without a loaded model, should get an error
        assert response.status_code in [404, 500]
    
    def test_detect_url_invalid(self, client):
        """Test detection with invalid URL."""
        response = client.post("/detect/url?url=invalid_url")
        assert response.status_code in [400, 404, 500]
    
    def test_batch_detect_no_files(self, client):
        """Test batch detection without files."""
        response = client.post("/detect/batch")
        assert response.status_code == 422


class TestDetectionService:
    """Test DetectionService class."""
    
    def test_service_initialization(self):
        """Test service initialization."""
        service = DetectionService()
        assert service.detectors == {}
        assert service.default_model is None
    
    def test_service_get_detector_empty(self):
        """Test getting detector from empty service."""
        service = DetectionService()
        
        with pytest.raises(ValueError, match="No models loaded"):
            service.get_detector()
    
    def test_service_list_models_empty(self):
        """Test listing models from empty service."""
        service = DetectionService()
        models = service.list_models()
        assert models == []
    
    def test_service_get_stats(self):
        """Test getting service statistics."""
        service = DetectionService()
        stats = service.get_stats()
        
        assert isinstance(stats, dict)
        assert "models_loaded" in stats
        assert stats["models_loaded"] == 0
        assert "total_requests" in stats
        assert stats["total_requests"] == 0


class TestUtilityFunctions:
    """Test utility functions in server module."""
    
    def test_load_image_from_bytes(self, sample_image_bytes):
        """Test loading image from bytes."""
        from perceptra_detector.api.server import load_image_from_bytes
        
        image = load_image_from_bytes(sample_image_bytes)
        assert isinstance(image, np.ndarray)
        assert len(image.shape) == 3
        assert image.shape[2] == 3  # RGB
    
    def test_load_image_from_invalid_bytes(self):
        """Test loading image from invalid bytes."""
        from perceptra_detector.api.server import load_image_from_bytes
        
        invalid_bytes = b"not an image"
        with pytest.raises(ValueError):
            load_image_from_bytes(invalid_bytes)
    
    def test_result_to_response(self):
        """Test converting DetectionResult to response format."""
        from perceptra_detector.api.server import result_to_response
        from perceptra_detector import DetectionResult, Detection, BoundingBox, TaskType
        
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=200)
        detections = [
            Detection(bbox=bbox, confidence=0.9, class_id=0, class_name="person")
        ]
        
        result = DetectionResult(
            detections=detections,
            image_shape=(640, 640, 3),
            task_type=TaskType.DETECTION,
            model_name="test_model",
            inference_time=0.1
        )
        
        response = result_to_response(result)
        
        assert response.num_detections == 1
        assert response.inference_time == 0.1
        assert len(response.detections) == 1
        assert response.detections[0].class_name == "person"


class TestCORSMiddleware:
    """Test CORS middleware."""
    
    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.options("/", headers={"Origin": "http://example.com"})
        
        # CORS headers should be present
        assert "access-control-allow-origin" in response.headers


class TestRequestValidation:
    """Test request validation."""
    
    def test_confidence_threshold_validation(self, client, sample_image_bytes):
        """Test confidence threshold validation."""
        files = {"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
        
        # Test invalid confidence (> 1.0)
        data = {"confidence_threshold": 1.5}
        response = client.post("/detect", files=files, data=data)
        assert response.status_code == 422
        
        # Test invalid confidence (< 0.0)
        data = {"confidence_threshold": -0.5}
        response = client.post("/detect", files=files, data=data)
        assert response.status_code == 422
    
    def test_iou_threshold_validation(self, client, sample_image_bytes):
        """Test IoU threshold validation."""
        files = {"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
        
        # Test invalid IoU (> 1.0)
        data = {"iou_threshold": 1.5}
        response = client.post("/detect", files=files, data=data)
        assert response.status_code == 422
        
        # Test invalid IoU (< 0.0)
        data = {"iou_threshold": -0.5}
        response = client.post("/detect", files=files, data=data)
        assert response.status_code == 422


class TestProcessTimeHeader:
    """Test X-Process-Time header middleware."""
    
    def test_process_time_header(self, client):
        """Test that X-Process-Time header is added to responses."""
        response = client.get("/health")
        
        assert "x-process-time" in response.headers
        # Should be a valid float string
        process_time = float(response.headers["x-process-time"])
        assert process_time >= 0


class TestErrorHandling:
    """Test error handling."""
    
    def test_404_error(self, client):
        """Test 404 error handling."""
        response = client.get("/nonexistent-endpoint")
        assert response.status_code == 404
    
    def test_method_not_allowed(self, client):
        """Test method not allowed."""
        # GET on POST-only endpoint
        response = client.get("/detect")
        assert response.status_code == 405


class TestAPIDocumentation:
    """Test API documentation endpoints."""
    
    def test_openapi_schema(self, client):
        """Test OpenAPI schema is available."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema
    
    def test_docs_endpoint(self, client):
        """Test Swagger UI documentation endpoint."""
        response = client.get("/docs")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    
    def test_redoc_endpoint(self, client):
        """Test ReDoc documentation endpoint."""
        response = client.get("/redoc")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]


class TestResponseModels:
    """Test response model validation."""
    
    def test_health_response_model(self):
        """Test HealthResponse model."""
        from perceptra_detector.api.models import HealthResponse
        
        response = HealthResponse(
            status="healthy",
            version="0.1.0",
            models_loaded=2,
            gpu_available=True
        )
        
        assert response.status == "healthy"
        assert response.version == "0.1.0"
        assert response.models_loaded == 2
        assert response.gpu_available is True
    
    def test_models_list_response(self):
        """Test ModelsListResponse model."""
        from perceptra_detector.api.models import ModelsListResponse
        
        response = ModelsListResponse(
            models=["yolo", "detr"],
            default_model="yolo"
        )
        
        assert len(response.models) == 2
        assert response.default_model == "yolo"
    
    def test_bbox_response_model(self):
        """Test BoundingBoxResponse model."""
        from perceptra_detector.api.models import BoundingBoxResponse
        
        bbox = BoundingBoxResponse(
            x1=10.0,
            y1=20.0,
            x2=100.0,
            y2=200.0,
            width=90.0,
            height=180.0
        )
        
        assert bbox.x1 == 10.0
        assert bbox.width == 90.0
    
    def test_detection_response_model(self):
        """Test DetectionResponse model."""
        from perceptra_detector.api.models import DetectionResponse, BoundingBoxResponse
        
        bbox = BoundingBoxResponse(
            x1=10.0, y1=20.0, x2=100.0, y2=200.0,
            width=90.0, height=180.0
        )
        
        detection = DetectionResponse(
            bbox=bbox,
            confidence=0.95,
            class_id=0,
            class_name="person",
            has_mask=False
        )
        
        assert detection.confidence == 0.95
        assert detection.class_name == "person"
        assert detection.has_mask is False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])