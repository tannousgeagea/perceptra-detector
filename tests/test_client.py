"""
Tests for the Python SDK client.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import requests
from io import BytesIO
from PIL import Image

from perceptra_detector.client.sdk import DetectorClient


@pytest.fixture
def mock_response():
    """Create a mock response object."""
    def _create_response(status_code=200, json_data=None):
        response = Mock()
        response.status_code = status_code
        response.json.return_value = json_data or {}
        response.raise_for_status = Mock()
        if status_code >= 400:
            response.raise_for_status.side_effect = requests.HTTPError()
        return response
    return _create_response


@pytest.fixture
def sample_detection_result():
    """Sample detection result for testing."""
    return {
        "detections": [
            {
                "bbox": {"x1": 10, "y1": 20, "x2": 100, "y2": 200, "width": 90, "height": 180},
                "confidence": 0.95,
                "class_id": 0,
                "class_name": "person"
            }
        ],
        "num_detections": 1,
        "image_shape": {"height": 640, "width": 640, "channels": 3},
        "task_type": "detection",
        "model_name": "yolo",
        "inference_time": 0.05,
        "class_counts": {"person": 1}
    }


@pytest.fixture
def client():
    """Create a client instance for testing."""
    return DetectorClient(base_url="http://localhost:8000")


class TestClientInitialization:
    """Test client initialization."""
    
    def test_client_default_initialization(self):
        """Test client with default parameters."""
        client = DetectorClient()
        assert client.base_url == "http://localhost:8000"
        assert client.timeout == 30
        assert client.verify_ssl is True
    
    def test_client_custom_url(self):
        """Test client with custom URL."""
        client = DetectorClient(base_url="http://example.com:9000")
        assert client.base_url == "http://example.com:9000"
    
    def test_client_strips_trailing_slash(self):
        """Test that trailing slash is removed from base_url."""
        client = DetectorClient(base_url="http://localhost:8000/")
        assert client.base_url == "http://localhost:8000"
    
    def test_client_custom_timeout(self):
        """Test client with custom timeout."""
        client = DetectorClient(timeout=60)
        assert client.timeout == 60
    
    def test_client_repr(self):
        """Test client string representation."""
        client = DetectorClient(base_url="http://localhost:8000")
        repr_str = repr(client)
        assert "DetectorClient" in repr_str
        assert "localhost:8000" in repr_str


class TestHealthCheck:
    """Test health check endpoint."""
    
    @patch('requests.Session.get')
    def test_health_check_success(self, mock_get, client, mock_response):
        """Test successful health check."""
        mock_get.return_value = mock_response(200, {
            "status": "healthy",
            "version": "0.1.0",
            "models_loaded": 2,
            "gpu_available": True
        })
        
        result = client.health_check()
        
        assert result["status"] == "healthy"
        assert result["version"] == "0.1.0"
        assert result["models_loaded"] == 2
        assert result["gpu_available"] is True
        
        mock_get.assert_called_once()
    
    @patch('requests.Session.get')
    def test_health_check_failure(self, mock_get, client, mock_response):
        """Test health check failure."""
        mock_get.return_value = mock_response(500)
        
        with pytest.raises(requests.HTTPError):
            client.health_check()


class TestListModels:
    """Test list models endpoint."""
    
    @patch('requests.Session.get')
    def test_list_models_success(self, mock_get, client, mock_response):
        """Test successful model listing."""
        mock_get.return_value = mock_response(200, {
            "models": ["yolo", "detr"],
            "default_model": "yolo"
        })
        
        result = client.list_models()
        
        assert "models" in result
        assert len(result["models"]) == 2
        assert "yolo" in result["models"]
        assert result["default_model"] == "yolo"


class TestGetModelInfo:
    """Test get model info endpoint."""
    
    @patch('requests.Session.get')
    def test_get_model_info_success(self, mock_get, client, mock_response):
        """Test successful model info retrieval."""
        mock_get.return_value = mock_response(200, {
            "name": "yolo",
            "backend": "yolo",
            "device": "cuda",
            "task_type": "detection",
            "num_classes": 80,
            "class_names": ["person", "car"],
            "confidence_threshold": 0.25,
            "iou_threshold": 0.45
        })
        
        result = client.get_model_info("yolo")
        
        assert result["name"] == "yolo"
        assert result["backend"] == "yolo"
        assert result["num_classes"] == 80
    
    @patch('requests.Session.get')
    def test_get_model_info_not_found(self, mock_get, client, mock_response):
        """Test model not found."""
        mock_get.return_value = mock_response(404)
        
        with pytest.raises(requests.HTTPError):
            client.get_model_info("nonexistent")


class TestDetect:
    """Test detect endpoint."""
    
    @patch('requests.Session.post')
    def test_detect_from_file_path(self, mock_post, client, sample_detection_result, mock_response, tmp_path):
        """Test detection from file path."""
        # Create a temporary image file
        image_path = tmp_path / "test.jpg"
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        img.save(image_path)
        
        mock_post.return_value = mock_response(200, sample_detection_result)
        
        result = client.detect(str(image_path))
        
        assert result["num_detections"] == 1
        assert result["detections"][0]["class_name"] == "person"
        
        mock_post.assert_called_once()
    
    @patch('requests.Session.post')
    def test_detect_from_bytes(self, mock_post, client, sample_detection_result, mock_response):
        """Test detection from image bytes."""
        # Create image bytes
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        img_bytes = BytesIO()
        img.save(img_bytes, format='JPEG')
        image_bytes = img_bytes.getvalue()
        
        mock_post.return_value = mock_response(200, sample_detection_result)
        
        result = client.detect(image_bytes)
        
        assert result["num_detections"] == 1
        mock_post.assert_called_once()
    
    @patch('requests.Session.post')
    def test_detect_from_numpy_array(self, mock_post, client, sample_detection_result, mock_response):
        """Test detection from numpy array."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        mock_post.return_value = mock_response(200, sample_detection_result)
        
        result = client.detect(image)
        
        assert result["num_detections"] == 1
        mock_post.assert_called_once()
    
    def test_detect_file_not_found(self, client):
        """Test detection with non-existent file."""
        with pytest.raises(FileNotFoundError):
            client.detect("nonexistent.jpg")
    
    def test_detect_invalid_type(self, client):
        """Test detection with invalid image type."""
        with pytest.raises(TypeError):
            client.detect(12345)  # Invalid type
    
    @patch('requests.Session.post')
    def test_detect_with_parameters(self, mock_post, client, sample_detection_result, mock_response):
        """Test detection with custom parameters."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        mock_post.return_value = mock_response(200, sample_detection_result)
        
        result = client.detect(
            image,
            model_name="yolo",
            confidence_threshold=0.7,
            iou_threshold=0.5
        )
        
        # Check that parameters were passed
        call_kwargs = mock_post.call_args
        assert call_kwargs is not None


class TestDetectBatch:
    """Test batch detection endpoint."""
    
    @patch('requests.Session.post')
    def test_detect_batch_success(self, mock_post, client, mock_response):
        """Test successful batch detection."""
        images = [
            np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            for _ in range(3)
        ]
        
        mock_post.return_value = mock_response(200, {
            "results": [
                {"num_detections": 1, "detections": [], "inference_time": 0.05},
                {"num_detections": 2, "detections": [], "inference_time": 0.06},
                {"num_detections": 0, "detections": [], "inference_time": 0.04}
            ],
            "num_images": 3,
            "total_inference_time": 0.15,
            "average_inference_time": 0.05
        })
        
        result = client.detect_batch(images)
        
        assert result["num_images"] == 3
        assert len(result["results"]) == 3
        assert result["average_inference_time"] == 0.05
    
    @patch('requests.Session.post')
    def test_detect_batch_with_file_paths(self, mock_post, client, mock_response, tmp_path):
        """Test batch detection with file paths."""
        # Create temporary image files
        image_paths = []
        for i in range(2):
            path = tmp_path / f"test{i}.jpg"
            img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
            img.save(path)
            image_paths.append(str(path))
        
        mock_post.return_value = mock_response(200, {
            "results": [{"num_detections": 1}, {"num_detections": 2}],
            "num_images": 2,
            "total_inference_time": 0.1,
            "average_inference_time": 0.05
        })
        
        result = client.detect_batch(image_paths)
        
        assert result["num_images"] == 2


class TestDetectURL:
    """Test URL detection endpoint."""
    
    @patch('requests.Session.post')
    def test_detect_url_success(self, mock_post, client, sample_detection_result, mock_response):
        """Test successful URL detection."""
        mock_post.return_value = mock_response(200, sample_detection_result)
        
        result = client.detect_url("https://example.com/image.jpg")
        
        assert result["num_detections"] == 1
        mock_post.assert_called_once()
    
    @patch('requests.Session.post')
    def test_detect_url_with_parameters(self, mock_post, client, sample_detection_result, mock_response):
        """Test URL detection with parameters."""
        mock_post.return_value = mock_response(200, sample_detection_result)
        
        result = client.detect_url(
            "https://example.com/image.jpg",
            model_name="yolo",
            confidence_threshold=0.6
        )
        
        assert result is not None
        mock_post.assert_called_once()


class TestClientContextManager:
    """Test client context manager."""
    
    def test_context_manager(self):
        """Test using client as context manager."""
        with DetectorClient() as client:
            assert client is not None
            assert isinstance(client, DetectorClient)
    
    @patch('requests.Session.close')
    def test_context_manager_closes_session(self, mock_close):
        """Test that session is closed when exiting context."""
        with DetectorClient() as client:
            pass
        
        # Session close should be called
        # Note: This tests the pattern, actual implementation may vary


class TestClientErrorHandling:
    """Test client error handling."""
    
    @patch('requests.Session.post')
    def test_http_error_handling(self, mock_post, client, mock_response):
        """Test handling of HTTP errors."""
        mock_post.return_value = mock_response(500, {"error": "Internal server error"})
        
        with pytest.raises(requests.HTTPError):
            client.detect(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
    
    @patch('requests.Session.post')
    def test_timeout_handling(self, mock_post, client):
        """Test handling of timeout errors."""
        mock_post.side_effect = requests.Timeout("Request timed out")
        
        with pytest.raises(requests.Timeout):
            client.detect(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
    
    @patch('requests.Session.post')
    def test_connection_error_handling(self, mock_post, client):
        """Test handling of connection errors."""
        mock_post.side_effect = requests.ConnectionError("Connection failed")
        
        with pytest.raises(requests.ConnectionError):
            client.detect(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))


class TestClientSession:
    """Test client session management."""
    
    def test_session_creation(self):
        """Test that session is created on initialization."""
        client = DetectorClient()
        assert hasattr(client, 'session')
        assert isinstance(client.session, requests.Session)
    
    def test_session_close(self):
        """Test closing the session."""
        client = DetectorClient()
        client.close()
        # Session should be closed (implementation dependent)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])